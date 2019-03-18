# coding :utf-8

import numpy as np
import pandas as pd
import torch
from torchvision.transforms import Compose
import sys
sys.path.append('../modules')
from modules import data_utils
import argparse, os.path
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns

def hist_data(dataset):
	dataloader = data_utils.DataLoader(dataset)
	listed_data = []
	for batched_input, _, _ in dataloader:
		listed_data += batched_input.data.view(-1).tolist()
	sns.distplot(listed_data, kde=False)
	plt.show()

def get_quantiles(dataset):
	dataloader = data_utils.DataLoader(dataset, batch_size=526)
	listed_data = []
	for batched_input, _, ixs in dataloader:
		listed_data += batched_input.data.view(-1).tolist()
	srs = pd.Series(listed_data)
	print('# of zeros')
	print('min', srs.min())
	print('0.05',srs.quantile(q=0.05))
	print('0.10',srs.quantile(q=0.1))
	print('0.25',srs.quantile(q=0.25))
	print('0.50',srs.quantile(q=0.5))
	print('0.75',srs.quantile(q=0.75))
	print('0.90',srs.quantile(q=0.9))
	print('0.95',srs.quantile(q=0.95))
	print('max', srs.max())

def search_zero(dataset):
	dataloader = data_utils.DataLoader(dataset, batch_size=1)
	print('Ixs of data containing zeros.')
	for batched_input, _, ixs in dataloader:
		if (batched_input.data==0).sum():
			print(ixs)


def get_parameters():
	par_parser = argparse.ArgumentParser()
	par_parser.add_argument('input_root', type=str, help='Path to the root directory under which inputs are located.')
	par_parser.add_argument('annotation_file', type=str, help='Path to the annotation csv file.')
	par_parser.add_argument('--fft_frame_length', type=float, default=0.008, help='FFT frame length in sec.')
	par_parser.add_argument('--fft_step_size', type=float, default=0.004, help='FFT step size in sec.')
	par_parser.add_argument('--fft_window_type', type=str, default='hann_window', help='Window type for FFT. "hann_window" by default.')
	par_parser.add_argument('--fft_no_centering', action='store_true', help='If selected, no centering in FFT.')
	return par_parser.parse_args()

if __name__ == "__main__":
	parameters = get_parameters()
	data_parser = data_utils.Data_Parser(parameters.input_root, parameters.annotation_file)
	fs = data_parser.get_sample_freq() # Assuming all the wav files have the same fs, get the 1st file's.

	fft_frame_length = int(np.floor(parameters.fft_frame_length * fs))
	fft_step_size = int(np.floor(parameters.fft_step_size * fs))

	to_tensor = data_utils.ToTensor()
	stft = data_utils.STFT(fft_frame_length, fft_step_size, window=parameters.fft_window_type, centering=not parameters.fft_no_centering)
	# take_log = data_utils.Transform(lambda x: x.log())

	# dataset = data_parser.get_data(data_type='train', transform=Compose([to_tensor,stft,take_log]))
	# dataset = data_parser.get_data(data_type='valid', transform=Compose([to_tensor,stft,take_log]))
	dataset = data_parser.get_data(transform=Compose([to_tensor,stft]))

	# hist_data(dataset)
	# get_quantiles(dataset)
	search_zero(dataset)