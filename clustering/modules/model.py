# coding: utf-8

import torch
import math

def choose_distribution(distribution_name):
	distributions = {"isotropic_gaussian":(sample_from_isotropic_gaussian,log_pdf_isotropic_gaussian,kl_isotropic_to_standard_gaussian)}
	return distributions[distribution_name]

@torch.jit.script
def sample_from_isotropic_gaussian(mean, log_variance):
	return mean + (0.5 * log_variance).exp() * torch.randn_like(mean)

@torch.jit.script
def kl_isotropic_to_standard_gaussian(mean, log_variance):
	"""
	Compute the KL divergence of N(mean, std*I) to N(0, I).
	References:
	Kingma and Willing 2014. Auto-Encoding Variational Bayes.
	"""
	return  - 0.5 * (1 + log_variance - mean.pow(2) - log_variance.exp()).sum()

@torch.jit.script
def log_pdf_isotropic_gaussian(value, mean, log_variance):
	value_mean_diff = value - mean
	return - 0.5 * (
				math.log(2 * math.pi)
				+ log_variance
				+ value_mean_diff * (-log_variance).exp() * value_mean_diff
				).sum()


class RNN_Variational_Encoder(torch.nn.Module):
	"""
	Encoder module for RNN-VAE assuming a feature noise parameterized by 2 vectors
	(e.g. location and scale for Gaussian, Logistic, Student's t, Uniform, Laplace, Elliptical, Triangular).
	References:
	Kingma and Willing 2014. Auto-Encoding Variational Bayes.
	Bowman et al. 2016. Generating Sentences from a Continuous Space.
	"""
	def __init__(self, input_size, rnn_hidden_size, rnn_type='LSTM', rnn_layers=1, hidden_dropout = 0.0, bidirectional=True, esn_leak=1.0):
		super(RNN_Variational_Encoder, self).__init__()
		if rnn_type == 'ESN':
			self.rnn = ESN(input_size, rnn_hidden_size, rnn_layers, dropout=hidden_dropout, bidirectional=bidirectional, batch_first=True, leak=esn_leak)
		else:
			self.rnn = getattr(torch.nn, rnn_type)(input_size, rnn_hidden_size, rnn_layers, dropout=hidden_dropout, bidirectional=bidirectional, batch_first=True)
		self.hidden_size_total = rnn_layers * rnn_hidden_size
		if bidirectional:
			self.hidden_size_total *= 2
		if rnn_type == 'LSTM':
			self.hidden_size_total *= 2

	def forward(self, packed_input):
		_, last_hidden = self.rnn(packed_input)
		if self.rnn.mode == 'LSTM':
			last_hidden = torch.cat(last_hidden, dim=-1)
		# Flatten the last_hidden into batch_size x rnn_layers * hidden_size
		last_hidden = last_hidden.transpose(0,1).contiguous().view(last_hidden.size(1), -1)
		return last_hidden



class RNN_Variational_Decoder(torch.nn.Module):
	"""
	Decoder module for RNN-VAE assuming probabilistic emission parameterized by two vectors (e.g., location and scale).
	References:
	Kingma and Willing 2014. Auto-Encoding Variational Bayes.
	Bowman et al. 2016. Generating Sentences from a Continuous Space.
	"""
	def __init__(self, output_size, rnn_hidden_size, mlp_hidden_size, feature_size, emission_sampler, rnn_type='LSTM', rnn_layers=1, input_dropout = 0.0, self_feedback=True, bidirectional=False, esn_leak=1.0):
		super(RNN_Variational_Decoder, self).__init__()
		assert rnn_layers==1, 'Only rnn_layers=1 is currently supported.'
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
		self.bidirectional = bidirectional
		if bidirectional:
			hidden_size_total *= 2
			self.rnn_cell_reverse = RNN_Cell(output_size, rnn_hidden_size, model_type=rnn_type, input_dropout=input_dropout, esn_leak=esn_leak)
			self.offset_predictor_reverse = MLP(rnn_hidden_size, mlp_hidden_size, 1)
			self.to_parameters_reverse = MLP_To_k_Vecs(rnn_hidden_size, mlp_hidden_size, output_size, 2)
		self.feature2hidden = torch.nn.Linear(feature_size, hidden_size_total)
		self.to_parameters = MLP_To_k_Vecs(rnn_hidden_size, mlp_hidden_size, output_size, 2)
		self.offset_predictor = MLP(rnn_hidden_size, mlp_hidden_size, 1)
		self.emission_sampler = emission_sampler
		self.rnn_cell = RNN_Cell(output_size, rnn_hidden_size, model_type=rnn_type, input_dropout=input_dropout, esn_leak=esn_leak)

	def forward(self, features, lengths=None, batch_sizes=None):
		"""
		The output is "flatten", meaning that it's PackedSequence.data.
		This should be sufficient for the training purpose etc. while we can avoid padding, which would affect the autograd and thus requires masking in the loss calculation.
		"""
		assert (not lengths is None) or (not batch_sizes is None), 'Either lengths or batch_sizes must be given.'
		if not lengths is None: # Mainly for the post training process.
			batch_sizes = self._length_to_batch_sizes(lengths)
		if self.bidirectional:
			return self._forward_bidirectional(features, batch_sizes)
		else:
			return self._forward_unidirectional(features, batch_sizes)
	

	# @torch.jit.script_method
	def _forward_unidirectional(self, features, batch_sizes):
		"""
		Manual implementation of RNNs based on RNNCell.
		"""
		hidden = self.feature2hidden(features)
		hidden = self.reshape_hidden(hidden)
		flatten_rnn_out = []
		flatten_emission_param1 = []
		flatten_emission_param2 = []
		flatten_out = []
		batched_input = torch.zeros(batch_sizes[0], self.rnn_cell.cell.input_size).to(features.device)
		for t in range(len(batch_sizes)):
			bs = batch_sizes[t]
			hidden = self.rnn_cell(batched_input[:bs], self.shrink_hidden(hidden,bs))
			rnn_out = self.get_output(hidden)
			emission_param1,emission_param2 = self.to_parameters(rnn_out)
			batched_input = self.emission_sampler(emission_param1, emission_param2)
			flatten_rnn_out += [rnn_out]
			flatten_emission_param1 += [emission_param1]
			flatten_emission_param2 += [emission_param2]
			flatten_out += [batched_input]
		flatten_rnn_out = torch.cat(flatten_rnn_out, dim=0)
		flatten_emission_param1 = torch.cat(flatten_emission_param1, dim=0)
		flatten_emission_param2 = torch.cat(flatten_emission_param2, dim=0)
		flatten_out = torch.cat(flatten_out, dim=0)
		flatten_offset_weights = self.offset_predictor(flatten_rnn_out).squeeze(-1) # num_batches x 1 -> num_batches
		return (((flatten_emission_param1,flatten_emission_param2), flatten_offset_weights, flatten_out),)

	def _forward_bidirectional(self, features, batch_sizes):
		"""
		The output is "flatten", meaning that it's PackedSequence.data.
		This should be sufficient for the training purpose etc. while we can avoid padding, which would affect the autograd and thus requires masking in the loss calculation.
		"""
		hidden = self.feature2hidden(features).view(features.size(0),-1,2)
		hidden_reverse_full = hidden[:,:,1]
		hidden = hidden[:,:,0]
		hidden = self.reshape_hidden(hidden)
		hidden_reverse_full = self.reshape_hidden(hidden_reverse_full)

		flatten_rnn_out = []
		flatten_emission_param1 = []
		flatten_emission_param2 = []
		flatten_out = []
		flatten_rnn_out_reverse = []
		flatten_emission_param1_reverse = []
		flatten_emission_param2_reverse = []
		flatten_out_reverse = []
		batched_input = torch.zeros(batch_sizes[0], self.rnn_cell.cell.input_size).to(features.device)
		zero_input_fullsize = torch.zeros_like(batched_input).to(features.device)
		previous_bs_reverse = batch_sizes[-1]
		batched_input_reverse = zero_input_fullsize[:previous_bs_reverse]
		hidden_reverse = hidden_reverse_full[:previous_bs_reverse]
		for t in range(len(batch_sizes)):
			bs = batch_sizes[t]
			bs_reverse = batch_sizes[-t-1]
			hidden = self.rnn_cell(batched_input[:bs], self.shrink_hidden(hidden,bs))
			hidden_reverse = self.rnn_cell_reverse(torch.cat([batched_input_reverse, zero_input_fullsize[previous_bs_reverse:bs_reverse]], dim=0), torch.cat([hidden_reverse, hidden_reverse_full[previous_bs_reverse:bs_reverse]], dim=0))
			previous_bs_reverse = bs_reverse
			rnn_out = self.get_output(hidden)
			rnn_out_reverse = self.get_output(hidden_reverse)
			emission_param1,emission_param2 = self.to_parameters(rnn_out)
			emission_param1_reverse, emission_param2_reverse = self.to_parameters_reverse(rnn_out_reverse)
			batched_input = self.emission_sampler(emission_param1, emission_param2)
			batched_input_reverse = self.emission_sampler(emission_param1_reverse, emission_param2_reverse)
			flatten_rnn_out += [rnn_out]
			flatten_emission_param1 += [emission_param1]
			flatten_emission_param2 += [emission_param2]
			flatten_out += [batched_input]
			flatten_rnn_out_reverse = [rnn_out_reverse] + flatten_rnn_out_reverse
			flatten_emission_param1_reverse = [emission_param1_reverse] + flatten_emission_param1_reverse
			flatten_emission_param2_reverse = [emission_param2_reverse] + flatten_emission_param2_reverse
			flatten_out_reverse = [batched_input_reverse] + flatten_out_reverse
		flatten_rnn_out = torch.cat(flatten_rnn_out, dim=0)
		flatten_emission_param1 = torch.cat(flatten_emission_param1, dim=0)
		flatten_emission_param2 = torch.cat(flatten_emission_param2, dim=0)
		flatten_out = torch.cat(flatten_out, dim=0)
		flatten_rnn_out_reverse = torch.cat(flatten_rnn_out_reverse, dim=0)
		flatten_emission_param1_reverse = torch.cat(flatten_emission_param1_reverse, dim=0)
		flatten_emission_param2_reverse = torch.cat(flatten_emission_param2_reverse, dim=0)
		flatten_out_reverse = torch.cat(flatten_out_reverse, dim=0)
		flatten_offset_weights = self.offset_predictor(flatten_rnn_out).squeeze(-1) # num_batches x 1 -> num_batches
		flatten_offset_weights_reverse = self.offset_predictor_reverse(flatten_rnn_out_reverse).squeeze(-1)
		return (
				((flatten_emission_param1,flatten_emission_param2), flatten_offset_weights, flatten_out),
				((flatten_emission_param1_reverse,flatten_emission_param2_reverse), flatten_offset_weights_reverse, flatten_out_reverse),
				)

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


class MLP(torch.jit.ScriptModule):
# class MLP(torch.nn.Module):
	"""
	Multi-Layer Perceptron.
	"""
	def __init__(self, input_size, hidden_size, output_size):
		super(MLP, self).__init__()
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.whole_network = torch.nn.Sequential(
			torch.nn.Linear(input_size, hidden_size),
			torch.nn.Tanh(),
			torch.nn.Linear(hidden_size, output_size)
			)

	@torch.jit.script_method
	def forward(self, batched_input):
		return self.whole_network(batched_input)

class ESN(torch.jit.ScriptModule):
	__constants__ = ['leak']
# class ESN(torch.nn.Module):
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
			flatten_hidden, last_hidden_l = self._forward_per_layer(flatten_input, packed_input.batch_sizes, getattr(self, 'weight_ih_l{l}'.format(l=l)), getattr(self, 'weight_hh_l{l}'.format(l=l)))
			last_hidden = torch.cat([last_hidden, last_hidden_l], dim=0)
			if self.bidirectional:
				flatten_hidden_back, last_hidden_l_back = self._forward_per_layer_backward(flatten_input, packed_input.batch_sizes, getattr(self, 'weight_ih_l{l}_reverse'.format(l=l)), getattr(self, 'weight_hh_l{l}_reverse'.format(l=l)))
				flatten_hidden = torch.cat([flatten_hidden, flatten_hidden_back], dim=-1)
				last_hidden = torch.cat([last_hidden, last_hidden_l_back], dim=0)
			flatten_input = self.drop(flatten_hidden)
		return flatten_hidden, last_hidden

	@torch.jit.script_method
	def _forward_per_layer(self, flatten_input, batch_sizes, weight_ih, weight_hh):
		# input2hidden
		input2hidden_transposed = weight_ih.mm(flatten_input.t())
		# hidden2hidden
		hidden_transposed = self.init_hidden(batch_sizes[0]).to(input2hidden_transposed.device).t()
		# flatten_hidden_transposed = torch.tensor([]).to(input2hidden_transposed.device)
		# last_hidden_transposed = torch.tensor([]).to(input2hidden_transposed.device)
		flatten_hidden_transposed = []
		last_hidden_transposed = []
		next_batch_sizes = torch.cat([batch_sizes[1:], torch.tensor([0])]).to(input2hidden_transposed.device)
		for t in range(len(batch_sizes)):
			bs,next_bs = batch_sizes[t], next_batch_sizes[t]
		# for bs,next_bs in zip(batch_sizes, next_batch_sizes):
			hidden_transposed = hidden_transposed[...,:bs]
			input2hidden_at_t_transposed = input2hidden_transposed[...,:bs]
			hidden2hidden_at_t_transposed = weight_hh.to_sparse().mm(hidden_transposed)
			hidden_transposed = (1.0 - self.leak) * hidden_transposed + self.leak * self.activation(input2hidden_at_t_transposed + hidden2hidden_at_t_transposed)
			# flatten_hidden_transposed = torch.cat([flatten_hidden_transposed,hidden_transposed], dim=-1)
			flatten_hidden_transposed += [hidden_transposed]
			input2hidden_transposed = input2hidden_transposed[...,bs:]
			# last_hidden_transposed = torch.cat([hidden_transposed[...,next_bs:], last_hidden_transposed], dim=-1)
			last_hidden_transposed = [hidden_transposed[...,next_bs:]] + last_hidden_transposed
		flatten_hidden_transposed = torch.cat(flatten_hidden_transposed, dim=-1)
		last_hidden_transposed = torch.cat(last_hidden_transposed, dim=-1)
		return flatten_hidden_transposed.t(), last_hidden_transposed.t().view(1,last_hidden_transposed.size(1),last_hidden_transposed.size(0))

	def _forward_per_layer_backward(self, flatten_input, batch_sizes, weight_ih, weight_hh):
		# input2hidden
		input2hidden_transposed = weight_ih.mm(flatten_input.t())
		# hidden2hidden
		init_hidden_fullsize_transposed = self.init_hidden(batch_sizes[0]).to(input2hidden_transposed.device).t()
		hidden_transposed = init_hidden_fullsize_transposed[:,:batch_sizes[-1]]
		# flatten_hidden_transposed = torch.tensor([]).to(input2hidden_transposed.device)
		flatten_hidden_transposed = []
		next_batch_sizes = torch.cat([batch_sizes[:-1].flip(0), torch.tensor([0])]).to(input2hidden_transposed.device)
		reversed_batch_sizes = batch_sizes.flip(0)
		for t in range(len(batch_sizes)):
			bs,next_bs = reversed_batch_sizes[t], next_batch_sizes[t]
		# for bs,next_bs in zip(, next_batch_sizes):
			input2hidden_at_t_transposed = input2hidden_transposed[...,-bs:]
			hidden2hidden_at_t_transposed = weight_hh.to_sparse().mm(hidden_transposed)
			hidden_transposed = (1.0 - self.leak) * hidden_transposed + self.leak * self.activation(input2hidden_at_t_transposed + hidden2hidden_at_t_transposed)
			# flatten_hidden_transposed = torch.cat([hidden_transposed,flatten_hidden_transposed], dim=-1)
			flatten_hidden_transposed = [hidden_transposed] + flatten_hidden_transposed
			input2hidden_transposed = input2hidden_transposed[...,:-bs]
			hidden_transposed = torch.cat([hidden_transposed,init_hidden_fullsize_transposed[:,bs:next_bs]], dim=-1)
		flatten_hidden_transposed = torch.cat(flatten_hidden_transposed, dim=-1)
		return flatten_hidden_transposed.t(), hidden_transposed.t().view(1,hidden_transposed.size(1),hidden_transposed.size(0))

	def init_hidden(self, batch_size):
		return torch.zeros((batch_size, self.hidden_size), requires_grad=False)

class ESNCell(torch.jit.ScriptModule):
	__constants__ = ['leak']
# class ESNCell(torch.nn.Module):
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
		self.weight_ih.data *= 6.0 / quantile
		self.weight_ih.data -= 3.0 / quantile

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
		eig_val,_ = torch.eig(self.weight_hh.data)
		self.weight_hh.data /= (eig_val.pow(2).sum(-1)).max().sqrt() / 0.99 # Adjust by the spectral radius.

		self.activation = torch.nn.Tanh()

	def forward(self, batched_input, hidden=None):
		if hidden is None:
			hidden = self.init_hidden(batched_input.size(0)).to(batched_input.device)
		return self._forward(batched_input, hidden)

	@torch.jit.script_method
	def _forward(self, batched_input, hidden):
		update = self.activation(self.weight_ih.mm(batched_input.t()) + self.weight_hh.to_sparse().mm(hidden.t())).t()
		hidden = (1.0 - self.leak) * hidden + self.leak * update
		return hidden

class SampleFromDirichlet(torch.nn.Module):
	def __init__(self, num_clusters, mlp_input_size, mlp_hidden_size, relax_scalar=0.05, base_counts = 1.0, max_weight=10.0, min_weight=0.1):
		super(SampleFromDirichlet, self).__init__()
		self.num_clusters = num_clusters
		self.relax_scalar = relax_scalar

		self.register_parameter(
				'q_pi_weights',
				torch.nn.Parameter(
					torch.randn(num_clusters),
					requires_grad=True)
				)

		if isinstance(base_counts, float):
			base_counts = torch.ones_like(self.q_pi_weights, requires_grad=False)
		self.base_counts = base_counts
		self.p_pi = torch.distributions.dirichlet.Dirichlet(base_counts)
		self.to_q_kappa_weights = MLP(mlp_input_size, mlp_hidden_size, num_clusters)
		self.max_weight = max_weight
		self.min_weight = min_weight
		self.to_non_negative = (lambda x: torch.nn.ReLU()(x) + min_weight)


	def forward(self, weights_seed):
		# Sample a relaxed category assignment kappa from q(kappa | x) = Dirichlet(q_kappa_weights).
		q_kappa_weights = self.to_non_negative(self.to_q_kappa_weights(weights_seed))
		q_kappa_given_x = torch.distributions.dirichlet.Dirichlet(q_kappa_weights)
		kappa = q_kappa_given_x.rsample()
		print(q_kappa_weights[0].min(), q_kappa_weights[0].median(), q_kappa_weights[0].max())
		print(kappa[0].min(), kappa[0].median(), kappa[0].max())

		# Sample a shape pi of the Dirichlet prior p(kappa | pi) from q(pi) = Dirichlet(self.q_pi_weights)
		q_pi = torch.distributions.dirichlet.Dirichlet(self.to_non_negative(self.q_pi_weights))
		pi = q_pi.rsample(sample_shape=(kappa.size(0),))

		# Compute the KL divergence between q(pi) and p(pi).
		kl_divergence = torch.distributions.kl_divergence(q_pi, self.p_pi) / kappa.size(0)

		# Compute the KL divergence between q(kappa | x) and p(kappa | pi).
		p_kappa_given_pi = torch.distributions.dirichlet.Dirichlet(self.relax_scalar * pi)
		kl_divergence += torch.distributions.kl_divergence(q_kappa_given_x, p_kappa_given_pi).sum()

		return kappa, kl_divergence, q_kappa_given_x


class SampleFromIsotropicGaussianMixture(torch.nn.Module):
	def __init__(self, prior_mean, prior_sd, num_clusters=None, ndim=None, post_mixture_noise=False, post_mixture_noise_prior_sd=None, mlp_input_size=None, mlp_hidden_size=None):
		super(SampleFromIsotropicGaussianMixture, self).__init__()
		if isinstance(prior_mean,float):
			assert not (ndim is None or num_clusters is None), 'num_clusters and ndim must be specified when prior_mean is a scalar.'
			prior_mean = torch.ones((num_clusters, ndim)) * prior_mean
		if isinstance(prior_sd,float):
			assert not (ndim is None or num_clusters is None), 'num_clusters and ndim must be specified when prior_mean is a scalar.'
			prior_sd = torch.ones((num_clusters, ndim)) * prior_mean
		self.prior_mean = prior_mean
		self.prior_sd = prior_sd
		self.post_mixture_noise = post_mixture_noise
		if post_mixture_noise_prior_sd is None:
			post_mixture_noise_prior_sd = 1.0
		if isinstance(post_mixture_noise_prior_sd, float):
			post_mixture_noise_prior_sd = torch.ones(self.prior_sd.size(-1)) * post_mixture_noise_prior_sd
		self.post_mixture_noise_prior_var =post_mixture_noise_prior_sd.pow(2)
		if post_mixture_noise:
			self.to_parameters = MLP_To_k_Vecs(mlp_input_size, mlp_hidden_size, self.prior_sd.size(-1), 2)
		else:
			self.prior_distr = torch.distributions.normal.Normal(self.prior_mean, self.prior_sd)
			self.register_parameter(
					'posterior_mean',
					torch.nn.Parameter(
						torch.randn_like(prior_mean)+self.prior_mean,
						requires_grad=True)
					)

			self.register_parameter(
					'posterior_log_var',
					torch.nn.Parameter(
						torch.randn_like(prior_sd.log()*2.0),
						requires_grad=True)
					)

	def forward(self, cluster_weights, parameter_seed=None):
		# broadcast cluster_weights
		cluster_weights = cluster_weights.view(cluster_weights.size()+(1,))
		if self.post_mixture_noise: # Two-level Gaussian noise, one for each cluster, the other for post mixture.
			# Sample values on mixture components from the posterior q(z | x) = N(posterior_mean, posterior_distr_sd).
			posterior_mean, posterior_log_var = self.to_parameters(parameter_seed)
			posterior_distr = torch.distributions.normal.Normal(posterior_mean, (0.5 * posterior_log_var).exp())
			samples = posterior_distr.rsample()

			# Get the prior distribution given the cluster_weights, p(z | cluster_weights).
			prior_distr = torch.distributions.normal.Normal(
				(self.prior_mean.view((1,)+self.prior_mean.size()) * cluster_weights).sum(1),
				((self.prior_sd.view((1,)+self.prior_sd.size()) * cluster_weights).pow(2).sum(1) + self.post_mixture_noise_prior_var.view((1,)+self.post_mixture_noise_prior_var.size())).sqrt()
			)

			# Measure the KL divergence between the posterior q(z | x) and the prior p(z | cluster_weights).
			kl_divergence = torch.distributions.kl_divergence(posterior_distr, prior_distr).sum()
		else: # The only noise is of each cluster's Gaussian.
			# Sample values on mixture components from q(z) = N(posterior_mean, posterior_distr_sd).
			posterior_distr = torch.distributions.normal.Normal(self.posterior_mean, (0.5 * self.posterior_log_var).exp())
			samples = posterior_distr.rsample((cluster_weights.size(0),))
			samples = (cluster_weights * samples).sum(1)

			# Measure the KL divergence between q(z) and p(z).
			kl_divergence = torch.distributions.kl_divergence(posterior_distr, self.prior_distr).sum() / cluster_weights.size(0)
			posterior_mean, posterior_log_var = self.posterior_mean, self.posterior_log_var
		return samples, kl_divergence, (posterior_mean, posterior_log_var)