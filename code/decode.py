# encoding: utf-8

import torch
from torchvision.transforms import Compose
import numpy as np
import pandas as pd
import librosa
import scipy.io.wavfile as spw
from modules import model, data_utils
import encode
import os, argparse, itertools


class Decoder(encode.Encoder):
	def decode(self, features, to_numpy=True, is_param = True, max_length=15000, offset_threshold=0.5):
		with torch.no_grad():
			if is_param:
				features = self.feature_sampler(**features)
			features = features.to(self.device).view(1,-1) # Add batch dimension
			max_length = torch.tensor([max_length]).to(self.device)
			_,offset_prediction,out = self.decoder(features, max_length, self.device) # The output is flattened and thus the batch dimension doesn't exist.
			offset_probs = torch.nn.Softmax(dim=-1)(offset_prediction)[:,1]
			for ix,p in enumerate(offset_probs):
				print(ix,p)
				if offset_threshold < p:
					out = out[:ix+1,:]
					break
			print(out)
		if to_numpy:
			out = out.data.numpy()
		return out


	def decode_dataset(self, dataset, to_numpy=True, is_param = True, max_length=15000, offset_threshold=0.5):
		decoded = []
		for data in dataset:
			decoded.append(self.decode(data, to_numpy=to_numpy, is_param=is_param, max_length=max_length, offset_threshold=offset_threshold))
		return decoded

def istft(spectra, hop_length=None, win_length=None, window='hann', center=True):
	wavs = []
	for s in spectra:
		s = s.reshape(s.shape[0], -1, 2).transpose(1,0,2) # freqs x time x real_imag
		s = s[...,0] + 1j * s[...,1] # To complex
		wavs.append(librosa.core.istft(s, hop_length=hop_length, win_length=win_length, window=window, center=center))
	return wavs


def get_parameters():
	par_parser = argparse.ArgumentParser()

	par_parser.add_argument('model_dir', type=str, help='Path to the directory containing learning info.')
	par_parser.add_argument('data', type=str, help='Path to the data csv file.')
	par_parser.add_argument('-d', '--device', type=str, default='cpu', help='Computing device.')
	par_parser.add_argument('-S', '--save_dir', type=str, default=None, help='Path to the directory where results are saved.')
	par_parser.add_argument('-F','--sampling_rate', type=int, default=32000, help='Sampling rate of the output wav file.')
	par_parser.add_argument('--fft_frame_length', type=float, default=0.008, help='FFT frame length in sec.')
	par_parser.add_argument('--fft_step_size', type=float, default=0.004, help='FFT step size in sec.')
	par_parser.add_argument('--fft_window_type', type=str, default='hann', help='Window type for FFT. "hann_window" by default.')
	par_parser.add_argument('--fft_no_centering', action='store_true', help='If selected, no centering in FFT.')

	return par_parser.parse_args()



if __name__ == '__main__':
	parameters = get_parameters()

	save_dir = parameters.save_dir
	if save_dir is None:
		save_dir = os.path.join(os.path.splitext(parameters.data)[0], 'decoded')
	if not os.path.isdir(save_dir):
		os.makedirs(save_dir)
	save_path = os.path.join(save_dir, 'decoded_data-ix-{ix}.wav')

	df_data = pd.read_csv(parameters.data).sort_values(['data_ix', 'feature_dim'])
	dataset = [{p_name:torch.from_numpy(p_gp.parameter_value.values.astype(np.float32)) for  p_name,p_gp in gp.groupby('parameter_name')} for data_ix,gp in df_data.groupby('data_ix')]

	# Get a model.
	decoder = Decoder(parameters.model_dir, device=parameters.device)

	decoded = decoder.decode_dataset(dataset)

	fft_frame_length = int(np.floor(parameters.fft_frame_length * parameters.sampling_rate))
	fft_step_size = int(np.floor(parameters.fft_step_size * parameters.sampling_rate))

	wavs = istft(decoded, hop_length=fft_step_size, win_length=fft_frame_length, window=parameters.fft_window_type, center=not parameters.fft_no_centering)

	for ix, w in enumerate(wavs):
		spw.write(save_path.format(ix=ix), parameters.sampling_rate, w)
