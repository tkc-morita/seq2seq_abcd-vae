# coding: utf-8

from numpy.core.fromnumeric import std
import torch
from modules.data_utils import Compose
import numpy as np
import pandas as pd
from modules import data_utils
import learning
import parselmouth
import os, argparse

class Encoder(learning.Learner):
	def __init__(self, model_config_path, device = 'cpu'):
		self.device = torch.device(device)
		self.retrieve_model(checkpoint_path = model_config_path, device=device)
		for param in self.parameters():
			param.requires_grad = False
		self.encoder.eval() # Turn off dropout
		self.feature_sampler.eval()
		self.decoder.eval()
		# self.counter = 0

	def _change_duration(self, wav_as_np, perturbed_duration_ratio, fs, minimum_pitch, maximum_pitch, transform=None):
		# margin = np.zeros(fs//100) # 1 sec margin on each side of the wave.
		wav_as_np = wav_as_np*(2**-15)
		# margin = np.array([])
		# wav_as_np = np.concatenate([margin,wav_as_np,margin])
		snd = parselmouth.Sound(values=wav_as_np, sampling_frequency=fs)
		# snd.save(os.path.expanduser('~/Documents/{}_orig.wav'.format(self.counter)), "WAV")
		snd = snd.lengthen(factor=perturbed_duration_ratio,minimum_pitch=minimum_pitch, maximum_pitch=maximum_pitch)
		# snd.save(os.path.expanduser('~/Documents/{}.wav'.format(self.counter)), "WAV")
		# self.counter += 1
		# perturbed_margin_len = int(np.round(margin.size*perturbed_duration_ratio))
		perturbed = snd.values[0,:] # [0,perturbed_margin_len:-perturbed_margin_len]
		perturbed = torch.from_numpy(perturbed.astype(np.float32))
		# perturbed = torch.from_numpy(wav_as_np.astype(np.float32))
		perturbed = perturbed*(2**15)
		# print(perturbed_margin_len,perturbed.shape,snd.values.shape)
		if not transform is None:
			perturbed = transform(perturbed)
		return perturbed


	def perturb(self, data, perturbed_duration_ratio, is_packed = False, **kwargs):
		if is_packed:
			data,lengths = torch.nn.utils.rnn.pad_packed_sequence(data,batch_first=True)
			data = [self._change_duration(wav[:l].numpy(), r, **kwargs) for wav,l,r in zip(data,lengths,perturbed_duration_ratio.tolist())]
		else:
			data = [self._change_duration(wav.numpy(), r, **kwargs) for wav,r in zip(data,perturbed_duration_ratio.tolist())]
		data = torch.nn.utils.rnn.pack_sequence(data)
		return data

	def encode(self, data, to_numpy = True, **perturb_kwargs):
		data = self.perturb(data, **perturb_kwargs)
		with torch.no_grad():
			data = data.to(self.device)
			last_hidden = self.encoder(data)
			logits = self.feature_sampler(last_hidden)
			probs = torch.nn.functional.softmax(logits, -1)
		if to_numpy:
			probs = (p.data.cpu().numpy() for p in probs)
		return probs

	def get_duration_and_quantiles(self,data,fs,coefficient_of_variance,qs):
		_,lengths = torch.nn.utils.rnn.pad_packed_sequence(data)
		durations_in_sec = lengths.float() / fs
		stds = durations_in_sec * coefficient_of_variance
		gaussian = torch.distributions.normal.Normal(durations_in_sec, stds)
		quantiles = gaussian.icdf(qs[:,None])
		return durations_in_sec, quantiles

	def encode_dataset(self, dataset, fs, save_path, to_numpy = True, batch_size=1,
							transform=None,
							coefficient_of_variance_in_percent=3.7,
							quantile_range_in_percent=95.0,
							quantile_step_in_percent=5.0,
							minimum_pitch=75.0, 
							maximum_pitch=600.0,):
		dataloader = data_utils.DataLoader(dataset, batch_size=batch_size)
		rename_existing_file(save_path)
		if 'label' in dataset.df_annotation.columns:
			df_ann = dataset.df_annotation.drop(columns=['onset_ix','offset_ix','length'])
		else:
			df_ann = None
		min_q = (100.0-quantile_range_in_percent)*0.5
		max_q_plus_one_step = min_q+quantile_range_in_percent+quantile_step_in_percent
		qs = torch.arange(min_q,max_q_plus_one_step,quantile_step_in_percent)/100.0
		coefficient_of_variance = coefficient_of_variance_in_percent/100.0
		for data, _, _, ix_in_list in dataloader:
			durations_in_sec, quantiles = self.get_duration_and_quantiles(data, fs, coefficient_of_variance, qs)
			for perturbed_duration,q in zip(quantiles,qs):
				probs = self.encode(data, perturbed_duration_ratio=perturbed_duration/durations_in_sec,
										transform=transform, is_packed=True, to_numpy=to_numpy,fs=fs,
										minimum_pitch=minimum_pitch, maximum_pitch=maximum_pitch)
				df_encoded = pd.DataFrame(probs)
				df_encoded.loc[:,'data_ix'] = ix_in_list
				df_encoded['perturbed_duration_in_sec'] = perturbed_duration.numpy()
				df_encoded = df_encoded.melt(id_vars=['data_ix','perturbed_duration_in_sec'], var_name='category_ix', value_name='prob')
				df_encoded['q_under_gaussian'] = q.item()
				if not df_ann is None:
					df_encoded = df_encoded.merge(df_ann, how='left', left_on='data_ix', right_index=True)
				# Only save MAP; otherwise, the file becomes too large.
				argmax_idxs = df_encoded.groupby('data_ix').prob.idxmax()
				df_encoded = df_encoded.loc[argmax_idxs,:]
				if os.path.isfile(save_path):
					df_encoded.to_csv(save_path, index=False, mode='a', header=False)
				else:
					df_encoded.to_csv(save_path, index=False)

def rename_existing_file(filepath):
	if os.path.isfile(filepath):
		new_path = filepath+'.prev'
		rename_existing_file(new_path)
		os.rename(filepath, new_path)

def get_parameters():
	par_parser = argparse.ArgumentParser()

	par_parser.add_argument('model_path', type=str, help='Path to the configuration file of a trained model.')
	par_parser.add_argument('input_root', type=str, help='Path to the root directory under which inputs are located.')
	par_parser.add_argument('annotation_file', type=str, help='Path to the annotation csv file.')
	par_parser.add_argument('data_normalizer', type=float, help='Normalizing constant to devide the data.')
	par_parser.add_argument('--annotation_sep', type=str, default=',', help='Separator symbol of the annotation file. Comma "," by default (i.e., csv).')
	par_parser.add_argument('-d', '--device', type=str, default='cpu', help='Computing device.')
	par_parser.add_argument('-S', '--save_path', type=str, default=None, help='Path to the file where results are saved.')
	par_parser.add_argument('--fft_frame_length', type=float, default=0.008, help='FFT frame length in sec.')
	par_parser.add_argument('--fft_step_size', type=float, default=0.004, help='FFT step size in sec.')
	par_parser.add_argument('--fft_window_type', type=str, default='hann_window', help='Window type for FFT. "hann_window" by default.')
	par_parser.add_argument('--fft_no_centering', action='store_true', help='If selected, no centering in FFT.')
	par_parser.add_argument('--channel', type=int, default=0, help='Channel ID # (starting from 0) of multichannel recordings to use.')
	par_parser.add_argument('-E','--epsilon', type=float, default=2**(-15), help='Small positive real number to add to avoid log(0).')
	par_parser.add_argument('-b', '--batch_size', type=int, default=1, help='Batch size.')
	par_parser.add_argument('--coefficient_of_variance_in_percent', type=float, default=3.7, help='Coefficient of variance (std/mean) assumed for the Gaussian duration noise around the observed values.')
	par_parser.add_argument('--quantile_range_in_percent', type=float, default=95.0, help='Range of duration quantiles to compute.')
	par_parser.add_argument('--quantile_step_in_percent', type=float, default=2.5, help='Step of duration quantiles to compute.')
	par_parser.add_argument('--maximum_pitch', type=float, default=10000.0, help='Maximum pitch frequency.')
	par_parser.add_argument('--minimum_pitch', type=float, default=301.0, help='Minimum pitch frequency.')

	return par_parser.parse_args()

if __name__ == '__main__':
	parameters = get_parameters()

	save_path = parameters.save_path
	if save_path is None:
		save_path = os.path.join(parameters.input_root, 'autoencoded.csv')
	save_dir = os.path.dirname(save_path)
	if not os.path.isdir(save_dir):
		os.makedirs(save_dir)

	data_parser = data_utils.Data_Parser(parameters.input_root, parameters.annotation_file, annotation_sep=parameters.annotation_sep)
	fs = data_parser.get_sample_freq() # Assuming all the wav files have the same fs, get the 1st file's.

	fft_frame_length = int(np.floor(parameters.fft_frame_length * fs))
	fft_step_size = int(np.floor(parameters.fft_step_size * fs))

	# Get a model.
	encoder = Encoder(parameters.model_path, device=parameters.device)

	to_tensor = data_utils.ToTensor()
	stft = data_utils.STFT(fft_frame_length, fft_step_size, window=parameters.fft_window_type, centering=not parameters.fft_no_centering)
	log = data_utils.Transform(lambda x: (x + parameters.epsilon).log())
	normalize = data_utils.Transform(lambda x: x / parameters.data_normalizer)

	dataset = data_parser.get_data(transform=to_tensor, channel=parameters.channel)

	transform = Compose([stft,log,normalize])

	encoder.encode_dataset(dataset, fs, save_path, batch_size=parameters.batch_size, transform=transform,
							coefficient_of_variance_in_percent=parameters.coefficient_of_variance_in_percent,
							quantile_range_in_percent=parameters.quantile_range_in_percent,
							quantile_step_in_percent=parameters.quantile_step_in_percent,
							minimum_pitch=parameters.minimum_pitch, 
							maximum_pitch=parameters.maximum_pitch,
							)

