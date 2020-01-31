# coding: utf-8

import torch
from modules.data_utils import Compose
import numpy as np
import pandas as pd
from modules import data_utils
import learning
import os, argparse, itertools


class Encoder(learning.Learner):
	def __init__(self, model_config_path, device = 'cpu'):
		self.device = torch.device(device)
		self.retrieve_model(checkpoint_path = model_config_path, device=device)
		for param in self.parameters():
			param.requires_grad = False
		self.encoder.eval() # Turn off dropout
		self.feature_sampler.eval()
		self.decoder.eval()


	def encode(self, data, is_packed = False, to_numpy = True):
		if not is_packed:
			if not isinstance(data, list):
				data = [data]
			data = torch.nn.utils.rnn.pack_sequence(data)
		with torch.no_grad():
			data = data.to(self.device)
			last_hidden = self.encoder(data)
			logits = self.feature_sampler(last_hidden)
			probs = torch.nn.functional.softmax(logits, -1)
		if to_numpy:
			probs = (p.data.cpu().numpy() for p in probs)
		return probs


	def encode_dataset(self, dataset, save_path, to_numpy = True, batch_size=1):
		dataloader = data_utils.DataLoader(dataset, batch_size=batch_size)
		rename_existing_file(save_path)
		if 'label' in dataset.df_annotation.columns:
			df_ann = dataset.df_annotation.drop(columns=['onset_ix','offset_ix','length'])
		else:
			df_ann = None
		for data, _, _, ix_in_list in dataloader:
			probs = self.encode(data, is_packed=True, to_numpy=to_numpy)
			df_encoded = pd.DataFrame(probs)
			df_encoded.loc[:,'data_ix'] = ix_in_list
			df_encoded = df_encoded.melt(id_vars=['data_ix'], var_name='category_ix', value_name='prob')
			if not df_ann is None:
				df_encoded = df_encoded.merge(df_ann, how='left', left_on='data_ix', right_index=True)
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
	log_and_normalize = data_utils.Transform(lambda x: (x + parameters.epsilon).log() / parameters.data_normalizer)

	dataset = data_parser.get_data(transform=Compose([to_tensor,stft,log_and_normalize]), channel=parameters.channel)

	encoder.encode_dataset(dataset, save_path, batch_size=parameters.batch_size)

