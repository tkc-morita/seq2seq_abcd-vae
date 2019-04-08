# encoding: utf-8

import torch
from torchvision.transforms import Compose
import numpy as np
import pandas as pd
import librosa
import scipy.io.wavfile as spw
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
from modules import model, data_utils
import encode
import os, argparse, itertools


class Decoder(encode.Encoder):
	def decode(self, features, to_numpy=True, is_param = True, max_length=15000, offset_threshold=0.5, take_mean=False):
		with torch.no_grad():
			if is_param:
				if take_mean:
					features = features['mean']
				else:
					features = self.feature_sampler(**features)
			features = features.to(self.device).view(1,-1) # Add batch dimension
			max_length = torch.tensor([max_length]).to(self.device)
			_,offset_prediction,out = self.decoder(features, max_length, self.device) # The output is flattened and thus the batch dimension doesn't exist.
			offset_probs = torch.nn.Sigmoid()(offset_prediction)
			for ix,p in enumerate(offset_probs):
				if offset_threshold < p:
					out = out[:ix+1,:]
					break
		if to_numpy:
			out = out.data.numpy()
		return out


	def decode_dataset(self, dataset, to_numpy=True, is_param = True, max_length=15000, offset_threshold=0.5, take_mean=False):
		if take_mean:
			self.decoder.sampler2mean()
		decoded = []
		for data in dataset:
			decoded.append(self.decode(data, to_numpy=to_numpy, is_param=is_param, max_length=max_length, offset_threshold=offset_threshold, take_mean=take_mean))
		return decoded

def exp_istft(log_spectra, hop_length=None, win_length=None, window='hann', center=True, normalizer=1.0, epsilon=0.0):
	wavs = []
	for ls in log_spectra:
		spectrum = np.exp(ls * normalizer) - epsilon # Inverse "log(spectrum_amplitude + epsilon) / normalizer".
		spectrum = spectrum.transpose(1,0) # freqs x time
		# spectrum = spectrum[...,0] + 1j * spectrum[...,1] # To complex
		wavs.append(librosa.core.istft(spectrum, hop_length=hop_length, win_length=win_length, window=window, center=center))
	return wavs

def to_spectrogram(log_spectra, fs, hop_length, normalizer=1.0, epsilon=0.0, onset=0, offset=None, save_path_template='{ix}.png'):
	for ix,ls in enumerate(log_spectra):
		if offset is None:
			ls = ls[onset:]
		else:
			ls = ls[onset:offset+1]
		spectrum = np.exp(ls * normalizer) - epsilon # Inverse "log(spectrum_amplitude + epsilon) / normalizer".
		spectrum /= 2**15
		spectrum = spectrum.transpose(1,0) # freqs x time
		log_baseline = np.log10(20) - 6
		# spectrum /= log_baseline
		spectrum = 20 * (np.log10(spectrum) - log_baseline)
		ax = sns.heatmap(spectrum, cmap='binary')
		ax.invert_yaxis()
		ax.set_ylabel('Frequency [Hz]')
		xticklabels = ax.get_xticklabels()
		ax.set_xticklabels([float(l.get_text()) * hop_length / fs for l in xticklabels], rotation=90)
		ax.set_xlabel('Time [sec]')
		plt.savefig(save_path_template.format(ix=ix), bbox_inches="tight")
		plt.clf()
		# plt.show()


def compare_spectrogram_with_original(log_spectra, df_path, root_dir, hop_length, win_length, window='hann', center=True, normalizer=1.0, epsilon=0.0, save_path_template='{ix}.png', onset=0):
	for ix,(ls,row) in enumerate(zip(log_spectra,df_path.itertuples())):
		# Read the original wav.
		fs,wav = spw.read(os.path.join(root_dir,row.input_path))
		onset_ix = int(np.ceil(row.onset * fs))
		offset_ix = int(np.floor(row.offset * fs))
		wav = wav[onset_ix:offset_ix].astype(np.float32)
		wav /= 2**15
		spectrum_original = librosa.core.stft(wav, n_fft=win_length, hop_length=hop_length, window=window, center=center)
		spectrum_original = np.abs(spectrum_original)
		num_frames = spectrum_original.shape[1]

		# The decoded log(spectrum) (ls) -> speectrum
		ls = ls[onset:onset+num_frames]
		spectrum = (np.exp(ls * normalizer) - epsilon) # Inverse "log(spectrum_amplitude + epsilon) / normalizer".
		spectrum /= 2**15
		spectrum = spectrum.transpose(1,0) # freqs x time


		log_baseline = np.log10(2) - 5
		# spectrum /= log_baseline
		spectrum = 20 * (np.log10(spectrum) - log_baseline)
		spectrum_original = 20 * (np.log10(spectrum_original) - log_baseline)

		fig,axes = plt.subplots(2,1)
		ax = sns.heatmap(spectrum, cmap='binary', ax=axes[0], vmin=0, vmax=120)
		ax.invert_yaxis()
		ax.set_ylabel('Frequency')
		xticklabels = ax.get_xticklabels()
		ax.set_xticklabels([float(l.get_text()) * hop_length / fs for l in xticklabels], rotation=90)
		ax.set_xlabel('Time [sec]')
		ax.set_title('Decoded')

		ax = sns.heatmap(spectrum_original, cmap='binary', ax=axes[1], vmin=0, vmax=120)
		ax.invert_yaxis()
		ax.set_ylabel('Frequency')
		xticklabels = ax.get_xticklabels()
		ax.set_xticklabels([float(l.get_text()) * hop_length / fs for l in xticklabels], rotation=90)
		ax.set_xlabel('Time [sec]')
		ax.set_title('Original')

		plt.subplots_adjust(hspace=0.7)
		plt.savefig(save_path_template.format(ix=ix), bbox_inches="tight")
		plt.clf()
		plt.close()
		# plt.show()


def get_parameters():
	par_parser = argparse.ArgumentParser()
	par_parser.add_argument('model_path', type=str, help='Path to the configuration file of a trained model.')
	par_parser.add_argument('data', type=str, help='Path to the data csv file.')
	par_parser.add_argument('data_normalizer', type=float, help='(Reverse-)Normalizing constant multiplied to the output.')
	par_parser.add_argument('-d', '--device', type=str, default='cpu', help='Computing device.')
	par_parser.add_argument('-S', '--save_dir', type=str, default=None, help='Path to the directory where results are saved.')
	par_parser.add_argument('-F','--sampling_rate', type=int, default=32000, help='Sampling rate of the output wav file.')
	par_parser.add_argument('--fft_frame_length', type=float, default=0.008, help='FFT frame length in sec.')
	par_parser.add_argument('--fft_step_size', type=float, default=0.004, help='FFT step size in sec.')
	par_parser.add_argument('--fft_window_type', type=str, default='hann', help='Window type for FFT. "hann_window" by default.')
	par_parser.add_argument('--fft_no_centering', action='store_true', help='If selected, no centering in FFT.')
	par_parser.add_argument('-m', '--take_mean', action='store_true', help='If selected, take the mean of probabilistic distributions instead of sampling from them.')
	par_parser.add_argument('-E', '--epsilon', type=float, default=2**(-15), help='Small positive real number to add to avoid log(0).')
	par_parser.add_argument('-V', '--visualize_spectra', action='store_true', help='If selected, draw spefctra as heatmaps instead of saving wavs.')
	par_parser.add_argument('-M', '--max_duration', type=float, default=10.0, help='Maximum duration in sec.')
	par_parser.add_argument('-O', '--compare_original', action='store_true', help='Compare decoded spectrogram with original.')
	par_parser.add_argument('-r','--original_root', type=str, help='Path to the root directory under which original wavs are located.')
	return par_parser.parse_args()



if __name__ == '__main__':
	parameters = get_parameters()

	save_dir = parameters.save_dir
	if save_dir is None:
		save_dir = os.path.join(os.path.splitext(parameters.data)[0], 'decoded')
	if not os.path.isdir(save_dir):
		os.makedirs(save_dir)
	if parameters.take_mean:
		save_path = os.path.join(save_dir, 'decoded_data-ix-{ix}_MEAN-BASED.wav')
	else:
		save_path = os.path.join(save_dir, 'decoded_data-ix-{ix}.wav')

	df_data = pd.read_csv(parameters.data).sort_values(['data_ix', 'feature_dim'])
	dataset = [{p_name:torch.from_numpy(p_gp.parameter_value.values.astype(np.float32)) for  p_name,p_gp in gp.groupby('parameter_name')} for data_ix,gp in df_data.groupby('data_ix')]

	# Get a model.
	decoder = Decoder(parameters.model_path, device=parameters.device)

	fft_frame_length = int(np.floor(parameters.fft_frame_length * parameters.sampling_rate))
	fft_step_size = int(np.floor(parameters.fft_step_size * parameters.sampling_rate))

	max_length = int(np.floor(0.1 * parameters.sampling_rate / fft_step_size))
	decoded = decoder.decode_dataset(dataset, take_mean=parameters.take_mean, max_length=max_length)


	if parameters.visualize_spectra:
		if parameters.compare_original:
			df_path = df_data.drop_duplicates(subset=['data_ix','input_path','onset','offset']).sort_values('data_ix')
			compare_spectrogram_with_original(decoded, df_path, parameters.original_root, fft_step_size, fft_frame_length, window=parameters.fft_window_type, center=not parameters.fft_no_centering, normalizer = parameters.data_normalizer, epsilon=parameters.epsilon, save_path_template=save_path.replace('.wav','.png'))
		else:
			to_spectrogram(decoded, parameters.sampling_rate, fft_step_size, normalizer = parameters.data_normalizer, save_path_template=save_path.replace('.wav','.png'), epsilon=parameters.epsilon)
	else:
		wavs = exp_istft(decoded, hop_length=fft_step_size, win_length=fft_frame_length, window=parameters.fft_window_type, center=not parameters.fft_no_centering, normalizer = parameters.data_normalizer, epsilon=parameters.epsilon)
		for ix, w in enumerate(wavs):
			spw.write(save_path.format(ix=ix), parameters.sampling_rate, w)
