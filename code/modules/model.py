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
				+ value_mean_diff * log_variance.exp() * value_mean_diff
				).sum(dim=sum_dims)


class RNN_Variational_Encoder(torch.nn.Module):
	"""
	Encoder module for RNN-VAE assuming a feature noise parameterized by 2 vectors
	(e.g. location and scale for Gaussian, Logistic, Student's t, Uniform, Laplace, Elliptical, Triangular).
	References:
	Kingma and Willing 2014. Auto-Encoding Variational Bayes.
	Bowman et al. 2016. Generating Sentences from a Continuous Space.
	"""
	def __init__(self, input_size, rnn_hidden_size, mlp_hidden_size, parameter_size, rnn_type='GRU', rnn_layers=1, dropout = 0.0, bidirectional=True):
		super(RNN_Variational_Encoder, self).__init__()
		self.rnn = RNN(input_size, rnn_hidden_size, num_layers=rnn_layers, dropout=dropout, bidirectional=bidirectional, model_type=rnn_type)
		hidden_size_total = rnn_layers * rnn_hidden_size
		if bidirectional:
			hidden_size_total *= 2
		if rnn_type == 'LSTM':
			hidden_size_total *= 2
		self.to_parameters = MLP_To_2_Vecs(hidden_size_total, mlp_hidden_size, parameter_size)

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
	def __init__(self, output_size, rnn_hidden_size, mlp_hidden_size, feature_size, rnn_type='GRU', rnn_layers=1, self_feedback = True, dropout = 0.0, emission_sampler = None):
		super(RNN_Variational_Decoder, self).__init__()
		hidden_size_total = rnn_layers*rnn_hidden_size
		if rnn_type=='LSTM':
			hidden_size_total *= 2
		self.feature2hidden = torch.nn.Linear(feature_size, hidden_size_total)
		self.to_parameters = MLP_To_2_Vecs(rnn_hidden_size, mlp_hidden_size, output_size)
		self.offset_predictor = torch.nn.Linear(rnn_hidden_size, 2)
		self.self_feedback = self_feedback
		self.reshape_hidden = self._get_reshape_hidden(rnn_type, self_feedback, rnn_layers, rnn_hidden_size)
		if self_feedback:
			assert rnn_layers==1, 'Only rnn_layers=1 is currently supported.'
			assert not emission_sampler is None, 'emission_sampler must be provided.'
			self.emission_sampler = emission_sampler
			self.rnn_cell = RNN_Cell(output_size, rnn_hidden_size, model_type=rnn_type, dropout=dropout)
			self.decode = self.rnn_with_self_feedback
			if rnn_type=='LSTM':
				self.get_output = lambda hidden: hidden[0]
			else:
				self.get_output = lambda hidden: hidden
		else:
			self.rnn = RNN(1, rnn_hidden_size, num_layers=rnn_layers, model_type=rnn_type)
			self.decode = self.rnn_WO_self_feedback


	def forward(self, features, lengths):
		"""
		The output is "flatten", meaning that it's PackedSequence.data.
		This should be sufficient for the training purpose etc. while we can avoid padding, which would affect the autograd and thus requires masking in the loss calculation.
		"""
		hidden = self.feature2hidden(features)
		hidden = self.reshape_hidden(hidden)
		flatten_emission_params, flatten_offset_weights = self.decode(hidden, lengths)
		return flatten_emission_params, flatten_offset_weights

	def rnn_WO_self_feedback(self, init_hidden, lengths):
		null_input = torch.nn.utils.rnn.pack_padded_sequence(torch.zeros((lengths.size(0),lengths.max().item(),1)), lengths, batch_first=True)
		packed_output, _ = self.rnn(null_input, init_hidden)
		flatten_rnn_out = packed_output.data
		flatten_emission_params = self.to_parameters(flatten_rnn_out)
		flatten_offset_weights = self.offset_predictor(flatten_rnn_out)
		return flatten_emission_params, flatten_offset_weights

	def rnn_with_self_feedback(self, hidden, lengths):
		batch_sizes = self._length_to_batch_sizes(lengths)
		# last_hidden = torch.tensor([])
		flatten_rnn_out = torch.tensor([]) # Correspond to PackedSequence.data.
		flatten_emission_param1 = torch.tensor([])
		flatten_emission_param2 = torch.tensor([])
		batched_input = torch.zeros(batch_sizes[0], self.rnn_cell.cell.input_size)
		for bs in batch_sizes:
			# last_hidden = torch.cat([hidden[bs:],last_hidden], dim=0) # hidden[bs:] is non-empty only if hidden.size(0) > bs.
			hidden = self.rnn_cell(batched_input[:bs])
			rnn_out = self.get_output(hidden)
			emission_param1,emission_param2 = self.to_parameters(rnn_out)
			batched_input = self.emission_sampler(emission_param1, emission_param2)
			flatten_rnn_out = torch.cat([flatten_rnn_out,rnn_out], dim=0)
			flatten_emission_param1 = torch.cat([flatten_emission_param1, emission_param1], dim=0)
			flatten_emission_param2 = torch.cat([flatten_emission_param2, emission_param2], dim=0)
		# last_hidden = torch.cat([hidden,last_hidden], dim=0)
		flatten_offset_weights = self.offset_predictor(flatten_rnn_out)
		return (flatten_emission_param1,flatten_emission_param2), flatten_rnn_out

	def _get_reshape_hidden(self, rnn_type, self_feedback, rnn_layers, rnn_hidden_size):
		def transpose_batch_and_layer_dims(reshape_hidden):
			"""
			reshape_hidden: num_batches x rnn_layers x rnn_hidden_size (x 2 if LSTM).
			"""
			return reshape_hidden.transpose(0,1).contiguous()
		def split_into_hidden_and_cell(reshaped_hidden):
			return (hidden[...,0],hidden[...,1])
		
		view_shape = [-1]
		operations = []
		if not self_feedback:
			view_shape.append(rnn_layers)
			operations.append(transpose_batch_and_layer_dims)
		view_shape.append(rnn_hidden_size)
		if rnn_type=='LSTM':
			view_shape.append(2)
			operations.append(split_into_hidden_and_cell)
		
		def reshape_hidden_base(hidden):
			return hidden.view(view_shape)

		operations = [reshape_hidden_base]+operations
		def reshape_hidden(hidden):
			for op in operations:
				hidden = op(hidden)
		return reshape_hidden


	def _length_to_batch_sizes(self, lengths):
		batch_sizes = [(lengths>t).sum() for t in range(lengths.max())]
		return batch_sizes


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


class MLP_To_2_Vecs(torch.nn.Module):
	def __init__(self, input_size, hidden_size1, output_size1, hidden_size2=None, output_size2=None):
		super(MLP_To_2_Vecs, self).__init__()
		if hidden_size2 is None:
			hidden_size2 = hidden_size1
		if output_size2 is None:
			output_size2 = output_size1
		self.mlp1 = MLP(input_size, hidden_size1, output_size1)
		self.mlp2 = MLP(input_size, hidden_size2, output_size2)
		
	def forward(self, batched_input):
		output1 = self.mlp1(batched_input)
		output2 = self.mlp2(batched_input)
		return output1, output2


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