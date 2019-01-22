# encoding: utf-8

import torch
from torchvision.transforms import Compose
import numpy as np
import pandas as pd
from modules import model, data_utils
import learning
import os, argparse, itertools


class Encoder(learning.Learner):
	def __init__(self, model_parameters_dir, device = 'cpu'):
		self.device = torch.device(device)
		self.retrieve_model(dir_path = model_parameters_dir)
		for param in itertools.chain(self.encoder.parameters(), self.decoder.parameters()):
			param.requires_grad = False
		self.encoder.to(self.device)
		self.encoder.eval() # Turn off dropout
		self.decoder.to(self.device)
		self.decoder.eval()


	def encode(self, data, is_packed = False, to_numpy = True):
		if not is_packed:
			data = torch.nn.utils.rnn.torch.nn.utils.rnn.pack_sequence([data])
		with torch.no_grad():
			data = data.to(self.device)
			hidden = self.encoder.init_hidden(batched_input.batch_sizes[0])
			h,c = self.encoder(batched_input, hidden)
		# Remove batch dimension.
		h = h.view(h.size(0), h.size(2))
		c = c.view(c.size(0), c.size(2))
		if to_numpy:
			h = h.data.numpy()
			c = c.data.numpy()
		return h,c
	
	def encode_dataset(self, dataset, to_numpy = True):
		dataloader = data_utils.DataLoader(dataset, batch_size=1)
		df_encoded = pd.DataFrame(columns=['data_ix', 'layer_ix', 'unit_ix', 'hidden', 'cell'])
		layer_ixs = np.tile(np.arange(self.encoder.num_layers), (1,self.encoder.hidden_size)).reshape(-1)
		hidden_unit_ix = np.tile(np.arange(self.encoder.hidden_size), (self.encoder.num_layers,1)).reshape(-1)
		for data, _, _, ix_in_list in dataloader:
			h,c = self.encode(data, is_packed=True, to_numpy=to_numpy)
			sub_df = pd.DataFrame()
			sub_df['hidden'] = h.reshape(-1)
			sub_df['cell'] = c.reshape(-1)
			sub_df['data_ix'] = ix_in_list[0]
			sub_df['layer_ix'] = layer_ixs
			sub_df['unit_ix'] = hidden_unit_ix
			df_encoded = df_encoded.append(sub_df, ignore_index=True, sort=False)
		return df_encoded

def get_parameters():
	par_parser = argparse.ArgumentParser()

	par_parser.add_argument('input_root', type=str, help='Path to the root directory under which inputs are located.')
	par_parser.add_argument('annotation_file', type=str, help='Path to the annotation csv file.')
	par_parser.add_argument('-d', '--device', type=str, default='cpu', help='Computing device.')
	par_parser.add_argument('-S', '--save_path', type=str, default=None, help='Path to the file where results are saved.')
	par_parser.add_argument('--fft_frame_length', type=float, default=0.008, help='FFT frame length in sec.')
	par_parser.add_argument('--fft_step_size', type=float, default=0.004, help='FFT step size in sec.')
	par_parser.add_argument('--fft_window_type', type=str, default='hann_window', help='Window type for FFT. "hann_window" by default.')
	par_parser.add_argument('--fft_no_centering', action='store_true', help='If selected, no centering in FFT.')

	return par_parser.parse_args()

if __name__ == '__main__':
	parameters = get_parameters()

	save_path = parameters.save_path
	if save_path is None:
		save_path = os.path.join(parameters.input_root, 'autoencoded.csv')

	data_parser = data_utils.Data_Parser(parameters.input_root, parameters.annotation_file)
	fs = data_parser.get_sample_freq() # Assuming all the wav files have the same fs, get the 1st file's.

	fft_frame_length = int(np.floor(parameters.fft_frame_length * fs))
	fft_step_size = int(np.floor(parameters.fft_step_size * fs))

	# Get a model.
	encoder = Encoder(parameters.model_dir, device=parameters.device)

	to_tensor = data_utils.ToTensor()
	stft = data_utils.STFT(fft_frame_length, fft_step_size, encoder.device, window=parameters.fft_window_type, centering=not parameters.fft_no_centering)
	logger.info("Sampling frequency of data: {fs}".format(fs=fs))
	logger.info("STFT window type: {fft_window}".format(fft_window=parameters.fft_window_type))
	logger.info("STFT frame lengths: {fft_frame_length_in_sec} sec".format(fft_frame_length_in_sec=parameters.fft_frame_length))
	logger.info("STFT step size: {fft_step_size_in_sec} sec".format(fft_step_size_in_sec=parameters.fft_step_size))


	dataset = data_parser.get_data(transform=Compose([to_tensor,stft]))

	df_encoded = encoder.encode_dataset(dataset)

	df_encoded.to_csv(save_path, index=False)

