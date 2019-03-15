# encoding: utf-8

import torch
from torchvision.transforms import Compose
import numpy as np
import pandas as pd
from modules import data_utils
import learning
import os, argparse, itertools


class Encoder(learning.Learner):
	def __init__(self, model_config_path, device = 'cpu'):
		self.device = torch.device(device)
		self.retrieve_model(checkpoint_path = model_config_path, device=device)
		for param in itertools.chain(self.encoder.parameters(), self.decoder.parameters()):
			param.requires_grad = False
		self.encoder.to(self.device)
		self.encoder.eval() # Turn off dropout
		self.decoder.to(self.device)
		self.decoder.eval()
		self.bag_of_data_decoder.to(self.device)
		self.bag_of_data_decoder.eval()


	def encode(self, data, is_packed = False, to_numpy = True):
		if not is_packed:
			data = torch.nn.utils.rnn.torch.nn.utils.rnn.pack_sequence([data])
		with torch.no_grad():
			data = data.to(self.device)
			params = self.encoder(data)
		if to_numpy:
			params = (p.data.numpy() for p in params)
		return params


	def encode_dataset(self, dataset, to_numpy = True, parameter_ix2name=None):
		if parameter_ix2name is None:
			parameter_ix2name = {}
		dataloader = data_utils.DataLoader(dataset, batch_size=1)
		df_encoded = pd.DataFrame()
		for data, _, ix_in_list in dataloader:
			params = self.encode(data, is_packed=True, to_numpy=to_numpy)
			for parameter_ix,p in enumerate(params):
				sub_df = pd.DataFrame()
				sub_df['parameter_value'] = p.reshape(-1)
				sub_df['data_ix'] = ix_in_list[0]
				if parameter_ix in parameter_ix2name:
					parameter_name = parameter_ix2name[parameter_ix]
				else:
					parameter_name = parameter_ix
				sub_df['parameter_name'] = parameter_name
				sub_df['feature_dim'] = sub_df.index
				df_encoded = df_encoded.append(sub_df, ignore_index=True, sort=False)
		return df_encoded

def get_parameters():
	par_parser = argparse.ArgumentParser()

	par_parser.add_argument('model_dir', type=str, help='Path to the directory containing learning info.')
	par_parser.add_argument('input_root', type=str, help='Path to the root directory under which inputs are located.')
	par_parser.add_argument('annotation_file', type=str, help='Path to the annotation csv file.')
	par_parser.add_argument('data_normalizer', type=float, help='Normalizing constant to devide the data.')
	par_parser.add_argument('-d', '--device', type=str, default='cpu', help='Computing device.')
	par_parser.add_argument('-S', '--save_path', type=str, default=None, help='Path to the file where results are saved.')
	par_parser.add_argument('--fft_frame_length', type=float, default=0.008, help='FFT frame length in sec.')
	par_parser.add_argument('--fft_step_size', type=float, default=0.004, help='FFT step size in sec.')
	par_parser.add_argument('--fft_window_type', type=str, default='hann_window', help='Window type for FFT. "hann_window" by default.')
	par_parser.add_argument('--fft_no_centering', action='store_true', help='If selected, no centering in FFT.')
	par_parser.add_argument('-p', '--parameter_names', type=str, default=None, help='Comma-separated parameter names.')
	par_parser.add_argument('-E','--epsilon', type=float, default=1e-15, help='Small positive real number to add to avoid log(0).')
	

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
	stft = data_utils.STFT(fft_frame_length, fft_step_size, window=parameters.fft_window_type, centering=not parameters.fft_no_centering)
	log_and_normalize = data_utils.Transform(lambda x: x.log() / parameters.data_normalizer)

	dataset = data_parser.get_data(transform=Compose([to_tensor,stft,log_and_normalize]))

	if parameters.parameter_names is None:
		parameter_ix2name = {}
	else:
		parameter_ix2name = dict(enumerate(parameters.parameter_names.split(',')))
	df_encoded = encoder.encode_dataset(dataset, parameter_ix2name=parameter_ix2name)
	df_encoded = df_encoded.sort_values('data_ix')
	if 'label' in data_parser.df_annotation.columns:
		df_encoded = df_encoded.merge(data_parser.df_annotation, how='left', left_on='data_ix', right_index=True)
	df_encoded.to_csv(save_path, index=False)

