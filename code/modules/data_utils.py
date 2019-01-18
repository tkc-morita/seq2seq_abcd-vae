# coding: utf-8

# coding: utf-8

import torch
import torch.utils.data
import pandas as pd
import numpy as np
import scipy.io.wavfile as spw
import os.path

class Data_Parser(object):
	def __init__(self, input_root, annotation_file, data_type_col_name = 'data_type'):
		self.df_annotation = pd.read_csv(annotation_file)
		self.input_root = input_root
		self.data_type_col_name = data_type_col_name

	def get_data(self, data_type, transform = None,):
		return Dataset(
						self.df_annotation[self.df_annotation[self.data_type_col_name]==data_type].reset_index(drop=True),
						self.input_root,
						transform=transform,
						)


	def get_sample_freq(self, input_path = None):
		if input_path is None: # Return the first wav file's fs.
			input_path = self.df_annotation.input_path.ix[0]
		fs, _ = spw.read(os.path.join(self.input_root, input_path))
		return fs


class Dataset(torch.utils.data.Dataset):
	def __init__(self, df_annotation, input_root, transform = None):
		self.df_annotation = df_annotation
		self.input_root = input_root
		self.transform = transform
		self.get_discrete_bounds()

	def get_discrete_bounds(self):
		for input_path,sub_df in self.df_annotation.groupby('input_path'):
			fs, input_data = spw.read(os.path.join(self.input_root, input_path))
			onset_ix = sub_df.onset.map(lambda sec: int(np.ceil(sec * fs)))
			offset_ix = sub_df.offset.map(lambda sec: int(np.floor(sec * fs)))
			self.df_annotation.loc[sub_df.index, 'onset_ix'] = onset_ix
			self.df_annotation.loc[sub_df.index, 'offset_ix'] = offset_ix
			self.df_annotation.loc[sub_df.index, 'length'] = offset_ix - onset_ix


	def sort_by_length(self):
		return self.df_annotation.sort_values('length', ascending=False)

	def __len__(self):
		"""Return # of data points."""
		return self.df_annotation.shape[0]

	def __getitem__(self, ix):
		"""Return """
		input_path = self.df_annotation.loc[ix, 'input_path']
		_, input_data = spw.read(os.path.join(self.input_root, input_path))
		if input_data.ndim > 1:
			input_data = input_data[:,0] # Use the 1st ch.
		input_data = input_data[self.df_annotation.loc[ix, 'onset_ix']:self.df_annotation.loc[ix, 'offset_ix']].astype(np.float32)


		if self.transform:
			input_data = self.transform(input_data)

		return input_data


class ToTensor(object):
	"""Convert ndarrays to Tensors."""
	def __init__(self):
		pass


	def __call__(self, input_data):
		return torch.from_numpy(input_data)


class Transform(object):
	def __init__(self, in_trans):
		self.in_trans = in_trans
		
	def __call__(self, input_data, output_data):
		in_transformed = self.in_trans(input_data)
		return in_transformed

class STFT(object):
	def __init__(self, frame_length, step_size, device, window='hann_window', centering=True):
		self.frame_length = frame_length
		self.step_size = step_size
		self.window = getattr(torch, window)(frame_length).to(device)
		self.centering = centering

	def __call__(self, input_data):
		transformed = torch.stft(
							input_data, self.frame_length, hop_length=self.step_size, window=self.window, center=self.centering
							)
		transformed = transformed.transpose(
							0,1 # Make the 0th dim represent time.
						).view(
							transformed.size(1),
							-1 # freq and real_vs_imag dimensions merged into one.
						).contiguous()
		return transformed


class DataLoader(object):
	def __init__(self, dataset, batch_size=1, shuffle=False):
		self.dataset = dataset
		self.split_into_batches(batch_size)
		self.shuffle = shuffle

	def split_into_batches(self,batch_size):
		df_sorted = self.dataset.sort_by_length()
		datasize = len(self.dataset)
		self.batches = [df_sorted.index[batch_start:min(batch_start+batch_size, datasize)] for batch_start in range(0,datasize,batch_size)]

	def __iter__(self):
		self._counter = 0
		self._stop = len(self.batches)
		if self.shuffle:
			np.random.shuffle(self.batches)
		return self

	def __next__(self):
		if self._counter >= self._stop:
			raise StopIteration
		ixs = self.batches[self._counter]
		self._counter += 1
		batched_input = []
		pseudo_input = [] # Input for the decoder.
		is_offset = []
		for ix in ixs:
			seq = self.dataset[ix]
			batched_input.append(seq)
			pseudo_input.append(torch.zeros_like(seq))
			l = seq.size(0)
			is_offset.append(torch.tensor([0]*(l-1)+[1]))
		batched_input = torch.nn.utils.rnn.torch.nn.utils.rnn.pack_sequence(batched_input)
		pseudo_input = torch.nn.utils.rnn.torch.nn.utils.rnn.pack_sequence(pseudo_input)
		is_offset = torch.nn.utils.rnn.torch.nn.utils.rnn.pack_sequence(is_offset)
		return batched_input, pseudo_input, is_offset

	def get_num_batches(self):
		return len(self.batches)