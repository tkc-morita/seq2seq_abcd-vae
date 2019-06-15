# coding: utf-8

import torch
import math

def choose_distribution(distribution_name):
	distributions = {"isotropic_gaussian":(sample_from_isotropic_gaussian,log_pdf_isotropic_gaussian,kl_isotropic_to_standard_gaussian)}
	return distributions[distribution_name]

def sample_from_isotropic_gaussian(mean, log_variance):
	return mean + (0.5 * log_variance).exp() * torch.randn_like(mean)

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
	def __init__(self, input_size, rnn_hidden_size, mlp_hidden_size, parameter_size, rnn_type='LSTM', rnn_layers=1, hidden_dropout = 0.0, bidirectional=True, esn_leak=1.0):
		super(RNN_Variational_Encoder, self).__init__()
		if rnn_type == 'ESN':
			self.rnn = ESN(input_size, rnn_hidden_size, rnn_layers, dropout=hidden_dropout, bidirectional=bidirectional, batch_first=True, leak=esn_leak)
		else:
			self.rnn = getattr(torch.nn, rnn_type)(input_size, rnn_hidden_size, rnn_layers, dropout=hidden_dropout, bidirectional=bidirectional, batch_first=True)
		hidden_size_total = rnn_layers * rnn_hidden_size
		if bidirectional:
			hidden_size_total *= 2
		if rnn_type == 'LSTM':
			hidden_size_total *= 2
		self.to_parameters = MLP_To_k_Vecs(hidden_size_total, mlp_hidden_size, parameter_size, 2)

	def forward(self, packed_input):
		_, last_hidden = self.rnn(packed_input)
		if self.rnn.mode == 'LSTM':
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
	def __init__(self, output_size, rnn_hidden_size, mlp_hidden_size, feature_size, emission_sampler, rnn_type='LSTM', rnn_layers=1, input_dropout = 0.0, self_feedback=True, esn_leak=1.0):
		super(RNN_Variational_Decoder, self).__init__()
		if not self_feedback:
			input_dropout = 1.0
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
		self.emission_sampler = emission_sampler
		self.rnn_cell = RNN_Cell(output_size, rnn_hidden_size, model_type=rnn_type, input_dropout=input_dropout, esn_leak=esn_leak)

	def forward(self, features, lengths=None, batch_sizes=None):
		"""
		The output is "flatten", meaning that it's PackedSequence.data.
		This should be sufficient for the training purpose etc. while we can avoid padding, which would affect the autograd and thus requires masking in the loss calculation.
		"""
		hidden = self.feature2hidden(features)
		hidden = self.reshape_hidden(hidden)

		# Manual implementation of RNNs based on RNNCell.
		assert (not lengths is None) or (not batch_sizes is None), 'Either lengths or batch_sizes must be given.'
		if not lengths is None: # Mainly for the post training process.
			batch_sizes = self._length_to_batch_sizes(lengths)
		flatten_rnn_out = torch.tensor([]).to(features.device) # Correspond to PackedSequence.data.
		flatten_emission_param1 = torch.tensor([]).to(features.device)
		flatten_emission_param2 = torch.tensor([]).to(features.device)
		flatten_out = torch.tensor([]).to(features.device)
		batched_input = torch.zeros(batch_sizes[0], self.rnn_cell.cell.input_size).to(features.device)
		for bs in batch_sizes:
			hidden = self.rnn_cell(batched_input[:bs], self.shrink_hidden(hidden,bs))
			rnn_out = self.get_output(hidden)
			emission_param1,emission_param2 = self.to_parameters(rnn_out)
			batched_input = self.emission_sampler(emission_param1, emission_param2)
			flatten_rnn_out = torch.cat([flatten_rnn_out,rnn_out], dim=0)
			flatten_emission_param1 = torch.cat([flatten_emission_param1, emission_param1], dim=0)
			flatten_emission_param2 = torch.cat([flatten_emission_param2, emission_param2], dim=0)
			flatten_out = torch.cat([flatten_out, batched_input], dim=0)
		flatten_offset_weights = self.offset_predictor(flatten_rnn_out).squeeze(-1) # num_batches x 1 -> num_batches
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



class RNN_Cell(torch.nn.Module):
	def __init__(self, input_size, hidden_size, model_type='LSTM', input_dropout = 0.0, esn_leak=1.0):
		super(RNN_Cell, self).__init__()
		self.drop = torch.nn.Dropout(input_dropout)
		self.mode = model_type
		if model_type == 'ESN':
			self.cell = ESNCell(input_size, hidden_size, leak=esn_leak)
		else:
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


class ESN(torch.nn.Module):
	def __init__(self, input_size, hidden_size, num_layers, dropout=0.0, bidirectional=False, batch_first=True, bias=False, leak=1.0, q=0.95, sparsity=0.1):
		super(ESN, self).__init__()
		self.input_size = input_size
		self.mode = 'ESN'
		self.hidden_size = hidden_size
		self.num_layers = num_layers
		self.bidirectional = bidirectional
		self.dropout = dropout
		self.bias = bias
		self.batch_first = batch_first
		self.leak = leak


		internal_input_size = hidden_size
		if bidirectional:
			internal_input_size *= 2
			self._init_parameters(q, internal_input_size, sparsity, '_reverse')
		# 0th layer
		self._init_parameters(q, internal_input_size, sparsity)

		self.drop = torch.nn.Dropout(p=dropout)
		self.activation = torch.nn.Tanh()

	def _init_parameters(self, q, internal_input_size, sparsity, parname_suffix=''):
		# input2hidden matrix.
		# Either either -3.0/input_size or 3.0/input_size. 
		# tanh(x) almost ceils and floors at x=3 and -3, 
		# so the sum should stay in the range most of the time.
		# 0th layer
		import scipy.stats as spstats
		input_quantile = spstats.binom.ppf(q, self.input_size, 0.5).astype('float32')
		self.register_parameter(
				'weight_ih_l0'+parname_suffix,
				torch.nn.Parameter(
					torch.randint(2, (self.hidden_size, self.input_size), dtype=torch.float32)
					* (6.0 / input_quantile)
					- (3.0 / input_quantile),
					requires_grad=False)
				)

		# Intermediate layers.
		internal_quantile = spstats.binom.ppf(q, internal_input_size, 0.5).astype('float32')
		[self.register_parameter(
				'weight_ih_l{l}'.format(l=l)+parname_suffix,
				torch.nn.Parameter(
					torch.randint(2, (self.hidden_size, internal_input_size), dtype=torch.float32)
					* (6.0 / internal_quantile)
					- (3.0 / internal_quantile),
					requires_grad=False)
				) for l in range(1,self.num_layers)]

		# hidden2hidden matrix.
		for l in range(self.num_layers):
			self.register_parameter(
					'weight_hh_l{l}'.format(l=l)+parname_suffix,
					torch.nn.Parameter(
						torch.randn(self.hidden_size, self.hidden_size),
						requires_grad=False)
					)
			weight_hh = getattr(self, 'weight_hh_l{l}'.format(l=l)+parname_suffix)
			weight_hh.data = torch.nn.Dropout(p=1.0-sparsity)(weight_hh.data)
			eig_val,_ = torch.eig(weight_hh)
			weight_hh /= (eig_val.pow(2).sum(-1)).max().sqrt() / 0.99 # Adjust by the spectral radius.


	def forward(self, packed_input, hidden=None):
		flatten_input = packed_input.data
		last_hidden = torch.tensor([]).to(packed_input.data.device)
		for l in range(self.num_layers):
			flatten_hidden, last_hidden_l = self._forward_per_layer(flatten_input, packed_input.batch_sizes, 'weight_ih_l{l}'.format(l=l), 'weight_hh_l{l}'.format(l=l))
			last_hidden = torch.cat([last_hidden, last_hidden_l], dim=0)
			if self.bidirectional:
				flatten_hidden_back, last_hidden_l_back = self._forward_per_layer(flatten_input, packed_input.batch_sizes, 'weight_ih_l{l}_reverse'.format(l=l), 'weight_hh_l{l}_reverse'.format(l=l))
				flatten_hidden = torch.cat([flatten_hidden, flatten_hidden_back], dim=-1)
				last_hidden = torch.cat([last_hidden, last_hidden_l_back], dim=0)
			flatten_input = self.drop(flatten_hidden)
		return flatten_hidden, last_hidden


	def _forward_per_layer(self, flatten_input, batch_sizes, weight_ih_name, weight_hh_name):
		# input2hidden
		input2hidden_transposed = getattr(self, weight_ih_name).mm(flatten_input.t())
		# hidden2hidden
		hidden_transposed = self.init_hidden(batch_sizes[0]).to(input2hidden_transposed.device).t()
		flatten_hidden_transposed = torch.tensor([]).to(input2hidden_transposed.device)
		last_hidden_transposed = torch.tensor([]).to(input2hidden_transposed.device)
		batch_decreases = torch.cat([batch_sizes[:-1]-batch_sizes[1:], torch.tensor([batch_sizes[-1]])]).to(input2hidden_transposed.device)
		for bs,bd in zip(batch_sizes, batch_decreases):
			hidden_transposed = hidden_transposed[...,:bs]
			input2hidden_at_t_transposed = input2hidden_transposed[...,:bs]
			hidden2hidden_at_t_transposed = getattr(self, weight_hh_name).to_sparse().mm(hidden_transposed)
			hidden_transposed = (1.0 - self.leak) * hidden_transposed + self.leak * self.activation(input2hidden_at_t_transposed + hidden2hidden_at_t_transposed)
			flatten_hidden_transposed = torch.cat([flatten_hidden_transposed,hidden_transposed], dim=-1)
			input2hidden_transposed = input2hidden_transposed[...,bs:]
			last_hidden_transposed = torch.cat([hidden_transposed[...,bs-bd:], last_hidden_transposed], dim=-1)
		return flatten_hidden_transposed.t(), last_hidden_transposed.t().view(1,last_hidden_transposed.size(1),last_hidden_transposed.size(0))

	def init_hidden(self, batch_size):
		return torch.zeros((batch_size, self.hidden_size), requires_grad=False)

class ESNCell(torch.nn.Module):
	def __init__(self, input_size, hidden_size, bias=False, leak=1.0, q=0.95, sparsity=0.1):
		super(ESNCell, self).__init__()
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.bias = bias
		assert not bias, 'No bias is currently supported.'
		self.leak = leak

		# input2hidden matrix.
		# Either either -3.0/input_size or 3.0/input_size. 
		# tanh(x) almost ceils and floors at x=3 and -3, 
		# so the sum should stay in the range most of the time.
		self.register_parameter(
				'weight_ih',
				torch.nn.Parameter(
					torch.randint(2, (hidden_size, input_size), dtype=torch.float32),
					requires_grad=False)
				)
		import scipy.stats as spstats
		quantile = spstats.binom.ppf(q, input_size, 0.5).astype('float32')
		self.weight_ih *= 6.0 / quantile
		self.weight_ih -= 3.0 / quantile

		# hidden2hidden matrix.
		# Sparse tensor is under development in PyTorch and the autograd etc. are currently unsupported. 
		# This module will follow some computation requiring autograd,
		# so we will not use the sparse representation right now.
		self.register_parameter(
				'weight_hh',
				torch.nn.Parameter(
					torch.randn(hidden_size, hidden_size),
					requires_grad=False)
				)
		self.weight_hh.data = torch.nn.Dropout(p=1.0-sparsity)(self.weight_hh.data)
		eig_val,_ = torch.eig(self.weight_hh)
		self.weight_hh /= (eig_val.pow(2).sum(-1)).max().sqrt() / 0.99 # Adjust by the spectral radius.

		self.activation = torch.nn.Tanh()


	def forward(self, batched_input, hidden=None):
		if hidden is None:
			hidden = self.init_hidden(batched_input.size(0)).to(batched_input.device)
		update = self.activation(self.weight_ih.mm(batched_input.t()) + self.weight_hh.mm(hidden.t())).t()
		hidden = (1.0 - self.leak) * hidden + self.leak * update
		return hidden


	def init_hidden(self, batch_size):
		return torch.zeros((batch_size, self.hidden_size), requires_grad=False)