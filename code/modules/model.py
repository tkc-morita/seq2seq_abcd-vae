# coding: utf-8

import torch
import numpy as np

class LSTM(torch.nn.Module):
	def __init__(self, input_dims, hidden_size, num_layers=1, dropout = 0.0, is_decoder=False):
		super(LSTM, self).__init__()
		self.num_layers = num_layers
		self.drop = torch.nn.Dropout(dropout)
		self.hidden_size = hidden_size
		self.lstm = torch.nn.LSTM(
						input_dims, # FFT output size
						hidden_size,
						num_layers,
						batch_first=True
						)
		self.is_decoder = is_decoder
		if self.is_decoder:
			self.feature2output = torch.nn.Linear(hidden_size, input_dims)
			self.offset_detector = torch.nn.Linear(hidden_size, 2)


	def forward(self, packed_input, init_hidden):
		"""
		packed_input is an instance of torch.nn.utils.rnn.PackedSequence.
		"""
		# Note that it is prohibited to directly instantiate torch.nn.utils.rnn.PackedSequence, even though it would lead to cleaner code...
		# Accordingly, the returned outputs are "flatten", meaning that the batch and sequence dimensions are merged, and the dropout applies to the padded tensor.
		padded_input, lengths = torch.nn.utils.rnn.pad_packed_sequence(packed_input, batch_first=True)
		padded_input = self.drop(padded_input) # Padding would be unnecessary if seld.drop could apply to packed_input.data and torch.nn.utils.rnn.PackedSequence(dropped, batch_sizes=packed_input.batch_sizes) is direct instantiated, but this is currently prohibited.
		packed_input = torch.nn.utils.rnn.pack_padded_sequence(padded_input, lengths, batch_first=True)
		features, last_hidden = self.lstm(packed_input, init_hidden)
		if self.is_decoder:
			flatten_features = features.data # Padding would affect the autograd, so we avoid it.
			flatten_features = self.drop(flatten_features)
			flatten_output = self.feature2output(flatten_features)
			flatten_offset_weights = self.offset_detector(flatten_features)
			return flatten_output, last_hidden, flatten_offset_weights
		else:
			return last_hidden



	def init_hidden(self, batch_size):
		weight = next(self.parameters()).data
		return (
				torch.zeros(self.num_layers, batch_size, self.hidden_size), # Hidden state
				torch.zeros(self.num_layers, batch_size, self.hidden_size) # Cell
				)

	def repackage_hidden(self, hidden):
		"""
		Delink hidden from the propagation chain.
		Input is a tuple of hidden states and cells.
		"""
		return tuple(torch.tensor(v.data) for v in hidden)

