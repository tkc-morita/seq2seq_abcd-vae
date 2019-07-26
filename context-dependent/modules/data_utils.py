# coding: utf-8

import torch
import torch.utils.data
import pandas as pd
import numpy as np
import scipy.io.wavfile as spw
import os.path
import warnings
warnings.simplefilter("error")

class Data_Parser(object):
	def __init__(self, input_root, annotation_file, data_type_col_name = 'data_type', annotation_sep=','):
		self.df_annotation = pd.read_csv(annotation_file, sep=annotation_sep)
		self.input_root = input_root
		self.data_type_col_name = data_type_col_name
		self.index_speakers()

	def index_speakers(self):
		if 'speaker' in self.df_annotation.columns:
			speaker2ix = {spk:ix for ix,spk in enumerate(self.df_annotation.speaker.unique())}
			self.df_annotation.loc[:,'speaker'] = self.df_annotation.speaker.map(speaker2ix)
		else:
			self.df_annotation['speaker'] = float('nan')

	def get_num_speakers(self):
		return len(self.df_annotation.speaker.unique())

	def get_data(self, data_type = None, transform = None, channel=0):
		if data_type is None:
			sub_df = self.df_annotation.copy()
		else:
			sub_df = self.df_annotation[self.df_annotation[self.data_type_col_name]==data_type].copy()
		return Dataset(
						sub_df,
						self.input_root,
						transform=transform,
						channel=channel
						)


	def get_sample_freq(self, input_path = None):
		if input_path is None: # Return the first wav file's fs.
			input_path = self.df_annotation.loc[0,'input_path']
		fs, _ = spw.read(os.path.join(self.input_root, input_path))
		return fs


class Dataset(torch.utils.data.Dataset):
	def __init__(self, df_annotation, input_root, transform = None, channel=0, context_length_in_sec=1.0):
		self.df_annotation = df_annotation
		self.input_root = input_root
		self.transform = transform
		self.channel = channel
		self.get_discrete_bounds(context_length_in_sec)

	def get_discrete_bounds(self, context_length_in_sec):
		self.max_abs = 0.0
		for input_path,sub_df in self.df_annotation.groupby('input_path'):
			fs, _ = spw.read(os.path.join(self.input_root, input_path))
			onset_ix = (sub_df.onset * fs).map(np.ceil)
			offset_ix = (sub_df.offset * fs).map(np.floor)
			self.df_annotation.loc[sub_df.index, 'onset_ix'] = onset_ix
			self.df_annotation.loc[sub_df.index, 'offset_ix'] = offset_ix
		self.df_annotation.loc[:, 'onset_ix'] = self.df_annotation.loc[:, 'onset_ix'].astype(int)
		self.df_annotation.loc[:, 'offset_ix'] = self.df_annotation.loc[:, 'offset_ix'].astype(int)
		self.df_annotation.loc[:, 'length'] = self.df_annotation.loc[:, 'offset_ix'] - self.df_annotation.loc[:, 'onset_ix']
		self.context_length = np.floor(context_length_in_sec * fs).astype(int)

	def sort_indices_by_length(self, ixs):
		return self.df_annotation.iloc[ixs,:].sort_values('length', ascending=False).index

	def __len__(self):
		"""Return # of data strings."""
		return self.df_annotation.shape[0]

	def __getitem__(self, ix):
		"""Return """
		input_path = self.df_annotation.loc[ix, 'input_path']
		_, input_data = spw.read(os.path.join(self.input_root, input_path))
		if input_data.ndim > 1:
			input_data = input_data[:,self.channel] # Use only one channel.
		onset_ix = self.df_annotation.loc[ix, 'onset_ix']
		offset_ix = self.df_annotation.loc[ix, 'offset_ix']
		input_data = input_data[onset_ix:offset_ix].astype(np.float32)
		prefix = input_data[max(0, onset_ix-self.context_length):onset_ix]
		suffix = input_data[offset_ix:min(input_data.size, offset_ix*self.context_length)]

		speaker = self.df_annotation.loc[ix, 'speaker']

		if self.transform:
			input_data = self.transform(input_data)
			prefix = self.transform(prefix)
			suffix = self.transform(suffix)
		return input_data, prefix, suffix, speaker


class ToTensor(object):
	"""Convert ndarrays to Tensors."""
	def __init__(self):
		pass


	def __call__(self, input_data):
		return torch.from_numpy(input_data)


class Transform(object):
	def __init__(self, in_trans):
		self.in_trans = in_trans
		
	def __call__(self, input_data):
		in_transformed = self.in_trans(input_data)
		return in_transformed

class STFT(object):
	def __init__(self, frame_length, step_size, window='hann_window', centering=True):
		self.frame_length = frame_length
		self.step_size = step_size
		self.window = getattr(torch, window)(frame_length)
		self.centering = centering

	def __call__(self, input_data):
		transformed = input_data.stft(
							self.frame_length, hop_length=self.step_size, window=self.window, center=self.centering
							)
		transformed = transformed.pow(2).sum(-1).sqrt() # Get the amplitude.
		transformed = transformed.transpose(
							0,1 # Make the 0th dim represent time.
						).contiguous()
		return transformed

class Compose(object):
	def __init__(self, transforms):
		self.transforms = transforms

	def __call__(self, data):
		for trans in self.transforms:
			data = trans(data)
		return data

class DataLoader(object):
	def __init__(self, dataset, batch_size=1, shuffle=False):
		self.dataset = dataset
		self.shuffle = shuffle
		if shuffle:
			sampler = torch.utils.data.RandomSampler(self.dataset, replacement=False)
		else:
			sampler = torch.utils.data.SequentialSampler(self.dataset)
		self.batch_sampler = torch.utils.data.BatchSampler(sampler, batch_size, drop_last=False)


	def __iter__(self):
		self.batches = list(self.batch_sampler)
		return self

	def __next__(self):
		if not self.batches:
			raise StopIteration
		ixs = self.batches.pop()
		ixs = self.dataset.sort_indices_by_length(ixs)
		batched_input = []
		prefixes = []
		suffixes = []
		speakers = []
		is_offset = []
		for ix in ixs:
			seq,prefix,suffix,spk = self.dataset[ix]
			batched_input.append(seq)
			prefixes.append(prefix)
			suffixes.append(suffixes)
			speakers.append(spk)
			l = seq.size(0)
			is_offset.append(torch.tensor([0.0]*(l-1)+[1.0]))
		batched_input = torch.nn.utils.rnn.torch.nn.utils.rnn.pack_sequence(batched_input)
		prefixes = torch.nn.utils.rnn.torch.nn
		is_offset = torch.nn.utils.rnn.torch.nn.utils.rnn.pack_sequence(is_offset)
		speakers = torch.tensor(speakers)
		return batched_input, is_offset, speakers, ixs

	def get_num_batches(self):
		return len(self.batch_sampler)