# encoding: utf-8

import torch
from torchvision.transforms import Compose
import numpy as np
import pandas as pd
import librosa
import scipy.io.wavfile as spw
import matplotlib.pyplot as plt
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

def exp_istft(log_spectra, hop_length=None, win_length=None, window='hann', center=True, normalizer=1.0):
	wavs = []
	for ls in log_spectra:
		spectrum = np.exp(ls * normalizer) # Inverse "log(spectrum amplitude) / normalizer".
		spectrum = spectrum.reshape(spectrum.shape[0], -1, 2).transpose(1,0,2) # freqs x time x real_imag
		spectrum = spectrum[...,0] + 1j * spectrum[...,1] # To complex
		wavs.append(librosa.core.istft(spectrum, hop_length=hop_length, win_length=win_length, window=window, center=center))
	return wavs

def to_spectgram(log_abs_spectra_and_signs):
	wavs = []
	for spectrum,sign in log_abs_spectra_and_signs:
		spectrum = np.exp(spectrum * normalizer) * sign # Inverse "log(abs(spectrum)) / normalizer".
		spectrum = spectrum.reshape(spectrum.shape[0], -1, 2).transpose(1,0,2) # freqs x time x real_imag
		spectrum = spectrum.pow(2).sum(-1).sqrt() # To amplitude.
		baseline = np.log10(2) - 5
		spectrum /= baseline
		spectrum = 20 * np.log10(spectrum)
		sns.heatmap(spectrum)
		plt.show()


def get_parameters():
	par_parser = argparse.ArgumentParser()
	par_parser.add_argument('model_dir', type=str, help='Path to the directory containing learning info.')
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
	decoder = Decoder(parameters.model_dir, device=parameters.device)

	decoded = decoder.decode_dataset(dataset, take_mean=parameters.take_mean)

	fft_frame_length = int(np.floor(parameters.fft_frame_length * parameters.sampling_rate))
	fft_step_size = int(np.floor(parameters.fft_step_size * parameters.sampling_rate))

	# wavs = exp_istft(decoded, hop_length=fft_step_size, win_length=fft_frame_length, window=parameters.fft_window_type, center=not parameters.fft_no_centering, normalizer = parameters.data_normalizer)

	# for ix, w in enumerate(wavs):
		# spw.write(save_path.format(ix=ix), parameters.sampling_rate, w)

	to_spectgram(log_abs_spectra_and_signs, normalizer = parameters.data_normalizer)
