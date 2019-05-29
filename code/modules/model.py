# coding: utf-8

import torch
import math

def choose_distribution(distribution_name):
	distributions = {"isotropic_gaussian":(sample_from_isotropic_gaussian,log_pdf_isotropic_gaussian,kl_isotropic_to_standard_gaussian)}
	return distributions[distribution_name]

def sample_from_isotropic_gaussian(mean, log_variance):
	noise = torch.randn_like(mean)
	return mean + (0.5 * log_variance).exp() * noise

def kl_isotropic_to_standard_gaussian(mean, log_variance, sum_only_over_val_dim = False):
	"""
	Compute the KL divergence of N(mean, std*I) to N(0, I).
	References:
	Kingma and Willing 2014. Auto-Encoding Variational Bayes.
	"""
	if sum_only_over_val_dim:
		sum_dims = -1
	else:
		sum_dims = ()
	return  - 0.5 * (1 + log_variance - mean.pow(2) - log_variance.exp()).sum(dim=sum_dims)

def log_pdf_isotropic_gaussian(value, mean, log_variance, sum_only_over_val_dim = False):
	if sum_only_over_val_dim:
		sum_dims = -1
	else:
		sum_dims = ()
	value_mean_diff = value - mean
	return - 0.5 * (
				math.log(2 * math.pi)
				+ log_variance
				+ value_mean_diff * (-log_variance).exp() * value_mean_diff
				).sum(dim=sum_dims)


class RNN_Variational_Encoder(torch.nn.Module):
	"""
	Encoder module for RNN-VAE assuming a feature noise parameterized by 2 vectors
	(e.g. location and scale for Gaussian, Logistic, Student's t, Uniform, Laplace, Elliptical, Triangular).
	References:
	Kingma and Willing 2014. Auto-Encoding Variational Bayes.
	Bowman et al. 2016. Generating Sentences from a Continuous Space.
	"""
	def __init__(self, input_size, rnn_hidden_size, mlp_hidden_size, parameter_size, rnn_type='LSTM', rnn_layers=1, hidden_dropout = 0.0, bidirectional=True):
		super(RNN_Variational_Encoder, self).__init__()
		if rnn_type == 'ESN':
			pass
		else:
			self.rnn = getattr(torch.nn, model_type)(input_size, rnn_hidden_size, rnn_layers, dropout=hidden_dropout, bidirectional=bidirectional, batch_first=True)
		hidden_size_total = rnn_layers * rnn_hidden_size
		if bidirectional:
			hidden_size_total *= 2
		if rnn_type == 'LSTM':
			hidden_size_total *= 2
		self.to_parameters = MLP_To_k_Vecs(hidden_size_total, mlp_hidden_size, parameter_size, 2)

	def forward(self, packed_input):
		_, last_hidden = self.rnn(packed_input)
		if self.rnn.rnn.mode == 'LSTM':
			last_hidden = torch.stack(last_hidden)
			batch_dim = 2
		else:
			batch_dim = 1
		# Flatten the last_hidden into batch_size x rnn_layers * hidden_size
		features = last_hidden.transpose(0,batch_dim).contiguous().view(last_hidden.size(batch_dim), -1)
		parameters = self.to_parameters(features)
		return parameters



class RNN_Variational_Decoder(torch.nn.Module):
	"""
	Decoder module for RNN-VAE assuming probabilistic emission parameterized by two vectors (e.g., location and scale).
	References:
	Kingma and Willing 2014. Auto-Encoding Variational Bayes.
	Bowman et al. 2016. Generating Sentences from a Continuous Space.
	"""
	def __init__(self, output_size, rnn_hidden_size, mlp_hidden_size, feature_size, rnn_type='LSTM', rnn_layers=1, dropout = 0.0, emission_sampler = None, self_feedback=True):
		super(RNN_Variational_Decoder, self).__init__()
		if not self_feedback:
			dropout = 1.0
		hidden_size_total = rnn_layers*rnn_hidden_size
		if rnn_type=='LSTM':
			hidden_size_total *= 2
			self.get_output = lambda hidden: hidden[0]
			self.reshape_hidden = lambda hidden: self._split_into_hidden_and_cell(self._reshape_hidden(hidden, (-1, rnn_hidden_size, 2)))
			self.shrink_hidden = lambda hidden, batch_size: (hidden[0][:batch_size], hidden[1][:batch_size])
		else:
			self.get_output = lambda hidden: hidden
			self.reshape_hidden = lambda hidden: self._reshape_hidden(hidden, (-1, rnn_hidden_size))
			self.shrink_hidden = lambda hidden, batch_size: hidden[:batch_size]
		self.feature2hidden = torch.nn.Linear(feature_size, hidden_size_total)
		self.to_parameters = MLP_To_k_Vecs(rnn_hidden_size, mlp_hidden_size, output_size, 2)
		self.offset_predictor = MLP(rnn_hidden_size, mlp_hidden_size, 1)
		assert rnn_layers==1, 'Only rnn_layers=1 is currently supported.'
		assert not emission_sampler is None, 'emission_sampler must be provided.'
		self.emission_sampler = emission_sampler
		self.rnn_cell = RNN_Cell(output_size, rnn_hidden_size, model_type=rnn_type, dropout=dropout)

	def forward(self, features, lengths, device):
		"""
		The output is "flatten", meaning that it's PackedSequence.data.
		This should be sufficient for the training purpose etc. while we can avoid padding, which would affect the autograd and thus requires masking in the loss calculation.
		"""
		hidden = self.feature2hidden(features)
		hidden = self.reshape_hidden(hidden)

		# Manual implementation of RNNs based on RNNCell.
		batch_sizes = self._length_to_batch_sizes(lengths)
		flatten_rnn_out = torch.tensor([]).to(device) # Correspond to PackedSequence.data.
		flatten_emission_param1 = torch.tensor([]).to(device)
		flatten_emission_param2 = torch.tensor([]).to(device)
		flatten_out = torch.tensor([]).to(device)
		# last_hidden = torch.tensor([])
		batched_input = torch.zeros(batch_sizes[0], self.rnn_cell.cell.input_size).to(device)
		for bs in batch_sizes:
			# last_hidden = torch.cat([hidden[bs:],last_hidden], dim=0) # hidden[bs:] is non-empty only if hidden.size(0) > bs.
			hidden = self.rnn_cell(batched_input[:bs], self.shrink_hidden(hidden,bs))
			rnn_out = self.get_output(hidden)
			emission_param1,emission_param2 = self.to_parameters(rnn_out)
			batched_input = self.emission_sampler(emission_param1, emission_param2)
			flatten_rnn_out = torch.cat([flatten_rnn_out,rnn_out], dim=0)
			flatten_emission_param1 = torch.cat([flatten_emission_param1, emission_param1], dim=0)
			flatten_emission_param2 = torch.cat([flatten_emission_param2, emission_param2], dim=0)
			flatten_out = torch.cat([flatten_out, batched_input], dim=0)
		# last_hidden = torch.cat([hidden,last_hidden], dim=0)
		flatten_offset_weights = self.offset_predictor(flatten_rnn_out).squeeze(-1) # Singleton list returned.
		return (flatten_emission_param1,flatten_emission_param2), flatten_offset_weights, flatten_out


	def _split_into_hidden_and_cell(self, reshaped_hidden):
		return (reshaped_hidden[...,0],reshaped_hidden[...,1])

	def _reshape_hidden(self, hidden, view_shape):
		return hidden.view(view_shape)

	def _length_to_batch_sizes(self, lengths):
		batch_sizes = [(lengths>t).sum() for t in range(lengths.max())]
		return batch_sizes

	def sampler2mean(self, mean_ix = 0):
		"""
		Replace the self.emission_sampler with a function that returns the mean of the distribution.
		Currently, the mean is assumed to be represented as one of the two arguments to the self.emission_sampler (selected by mean_ix).
		"""
		self._emission_sampler = self.emission_sampler
		self.emission_sampler = lambda emission_param1, emission_param2: emission_param1 if mean_ix==0 else emission_param2
		
	def mean2sampler(self):
		"""
		Reset self.emission_sampler as a sampler.
		"""
		self.emission_sampler = self._emission_sampler


class RNN(torch.nn.Module):
	def __init__(self, input_size, hidden_size, model_type='GRU', num_layers=1, dropout = 0.0, bidirectional=False):
		super(RNN, self).__init__()
		self.drop = torch.nn.Dropout(dropout)
		self.rnn = getattr(torch.nn, model_type)(input_size, hidden_size, num_layers, bidirectional=bidirectional, batch_first=True)


	def forward(self, packed_input, init_hidden=None):
		"""
		packed_input is an instance of torch.nn.utils.rnn.PackedSequence.
		"""
		# Note that it is prohibited to directly instantiate torch.nn.utils.rnn.PackedSequence, even though it would lead to cleaner code...
		padded_input, lengths = torch.nn.utils.rnn.pad_packed_sequence(packed_input, batch_first=True)
		padded_input = self.drop(padded_input) # Padding would be unnecessary if seld.drop could apply to packed_input.data and torch.nn.utils.rnn.PackedSequence(dropped, batch_sizes=packed_input.batch_sizes) is direct instantiated, but this is currently prohibited.
		packed_input = torch.nn.utils.rnn.pack_padded_sequence(padded_input, lengths, batch_first=True)

		output, last_hidden = self.rnn(packed_input)
		return output, last_hidden


class RNN_Cell(torch.nn.Module):
	def __init__(self, input_size, hidden_size, model_type='GRU', dropout = 0.0):
		super(RNN_Cell, self).__init__()
		self.drop = torch.nn.Dropout(dropout)
		self.cell = getattr(torch.nn, model_type+'Cell')(input_size, hidden_size)
	
	def forward(self, batched_input, init_hidden=None):
		batched_input = self.drop(batched_input)
		hidden = self.cell(batched_input, init_hidden)
		return hidden


class MLP_To_k_Vecs(torch.nn.Module):
	def __init__(self, input_size, hidden_size, output_size, k):
		super(MLP_To_k_Vecs, self).__init__()
		self.mlps = torch.nn.ModuleList([MLP(input_size, hidden_size, output_size) for ix in range(k)])
		
	def forward(self, batched_input):
		out = [mlp(batched_input) for mlp in self.mlps]
		return out


class MLP(torch.nn.Module):
	"""
	Multi-Layer Perceptron.
	"""
	def __init__(self, input_size, hidden_size, output_size):
		super(MLP, self).__init__()
		self.hidden_size = hidden_size
		self.whole_network = torch.nn.Sequential(
			torch.nn.Linear(input_size, hidden_size),
			torch.nn.Tanh(),
			torch.nn.Linear(hidden_size, output_size)
			)

	def forward(self, batched_input):
		return self.whole_network(batched_input)