# coding: utf-8

import torch
import math, collections

def choose_distribution(distribution_name):
	distributions = {
		"isotropic_gaussian":
			(
				sample_from_isotropic_gaussian,
				log_pdf_isotropic_gaussian,
				kl_isotropic_to_standard_gaussian,
				2
			)}
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

def pad_flatten_sequence(flatten, batch_sizes, padding_value=0.0, batch_first=False):
	return torch.nn._VF._pad_packed_sequence(
				flatten,
				batch_sizes.cpu(),
				batch_first,
				padding_value,
				batch_sizes.size(0)
				)

def get_mask_from_lengths(lengths):
	max_length = lengths.max()
	mask = torch.arange(max_length).view(1,-1).to(lengths.device)>=lengths.view(-1,1)
	return mask

def batch_sizes2lengths(batch_sizes):
	lengths = torch.zeros(batch_sizes[0]).long()
	for bs in batch_sizes:
		lengths[:bs] += 1
	return lengths

def lengths2batch_sizes(lengths):
	batch_sizes = torch.zeros(lengths.max()).long()
	for l in lengths:
		batch_sizes[:l] += 1
	return batch_sizes


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

	def pack_init_args(self):
		init_args = {
			"input_size": self.rnn.input_size,
			"rnn_hidden_size": self.rnn.hidden_size,
			"rnn_type": self.rnn.mode.split('_')[0],
			"rnn_layers": self.rnn.num_layers,
			"hidden_dropout": self.rnn.dropout,
			"bidirectional": self.rnn.bidirectional,
		}
		if init_args['rnn_type'] == 'ESN':
			init_args["esn_leak"] = self.rnn.leak
		return init_args

class AttentionEncoderToFixedLength(torch.nn.Module):
	def __init__(self, input_size, hidden_size, mlp_hidden_size, num_heads=8, num_layers=1, dropout=0.0):
		super(AttentionEncoderToFixedLength, self).__init__()
		self.self_attention = SelfAttentionEncoder(input_size, hidden_size, mlp_hidden_size, num_heads=num_heads, num_layers=num_layers, dropout=dropout)
		self.register_parameter(
			'cls_embedding',
			torch.nn.Parameter(torch.randn(input_size), requires_grad=True)
			)

	def forward(self, packed_input):
		"""
		Append a dummy frame to the beginning and use its output for the classification.
		cf. BERT
		"""
		padded_input, lengths = torch.nn.utils.rnn.pad_packed_sequence(packed_input, batch_first=False)
		padded_input = torch.cat([self.cls_embedding.expand(1,padded_input.size(1),-1), padded_input], dim=0)
		lengths += 1
		packed_input = torch.nn.utils.rnn.pack_padded_sequence(padded_input, lengths, batch_first=False)
		hidden = self.self_attention(packed_input)
		
		out = hidden[:packed_input.batch_sizes[0]]
		return out

	def pack_init_args(self):
		return self.self_attention.pack_init_args()


class SelfAttentionEncoder(torch.nn.Module):
	def __init__(self, input_size, hidden_size, mlp_hidden_size, num_heads=8, num_layers=1, dropout=0.0):
		super(SelfAttentionEncoder, self).__init__()
		self.to_hidden = MLP(input_size, mlp_hidden_size, hidden_size)
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.mlp_hidden_size = mlp_hidden_size
		self.num_heads = num_heads
		self.num_layers = num_layers
		self.dropout = dropout
		self.self_attention = torch.nn.Sequential(
									collections.OrderedDict([
									('layer{}'.format(l),
									SelfAttentionLayer(hidden_size, mlp_hidden_size, num_heads=num_heads, dropout=dropout)
									)
									for l in range(num_layers)
									]))

	def forward(self, input_seqs, lengths=None, batch_sizes=None):
		if isinstance(input_seqs, torch.nn.utils.rnn.PackedSequence):
			batch_sizes = input_seqs.batch_sizes
			input_seqs = input_seqs.data
		assert lengths is not None or batch_sizes is not None, 'lengths or batch_sizes must be specified if input_seqs is not a PackedSequence instance.'
		if lengths is None:
			lengths = batch_sizes2lengths(batch_sizes)
		if batch_sizes is None:
			batch_sizes = lengths2batch_sizes(lengths)
		lengths = lengths.to(input_seqs.device)
		hidden = self.to_hidden(input_seqs)
		pos_encodings = self.encode_position(lengths)
		hidden = hidden + pos_encodings

		input_as_dict = {'values':hidden, 'lengths':lengths, 'batch_sizes':batch_sizes}
		input_as_dict['subbatch_info'] = self.group_into_subbatches(lengths, batch_sizes)

		flatten_out = self.self_attention(input_as_dict)['values']
		return flatten_out

	def encode_position(self, lengths):
		max_length = lengths.max()
		half_hidden_size = self.hidden_size // 2
		encodings = (
			torch.arange(max_length).view(-1,1).to(lengths.device).float()
			/
			(torch.arange(0,1,2/self.hidden_size).view(1,-1).to(lengths.device)*math.log(10000.0)).exp()
		)
		encodings = torch.stack([encodings.sin(), encodings.cos()], dim=-1).view(encodings.size(0),-1)
		encodings = torch.nn.utils.rnn.pack_sequence(
						[encodings[:l] for l in lengths]
						).data
		return encodings

	def group_into_subbatches(self, lengths, batch_sizes):
		subbatch_size = 0
		subbatch_info = []
		subbatch_lengths = []
		subbatch_ixs = []
		max_length = lengths.max()
		for batch_ix, l in enumerate(lengths):
			if subbatch_lengths and self._is_full_subbatch_size(max_length, subbatch_size+1):
				subbatch_lengths = torch.tensor(subbatch_lengths).to(lengths.device)
				subbatch_sizes = lengths2batch_sizes(subbatch_lengths)
				subbatch_masks = self._get_masks(subbatch_lengths)
				subbatch_token_ixs = self._get_subbatch_token_ixs(batch_sizes, subbatch_ixs, subbatch_lengths)
				subbatch_info.append((subbatch_token_ixs, subbatch_lengths, subbatch_sizes, subbatch_masks))
				max_length = l # Assuming lengths descending.
				subbatch_size = 0
				subbatch_lengths = []
				subbatch_ixs = []
			subbatch_size += 1
			subbatch_ixs.append(batch_ix)
			subbatch_lengths.append(l)
		subbatch_lengths = torch.tensor(subbatch_lengths).to(lengths.device)
		subbatch_sizes = lengths2batch_sizes(subbatch_lengths)
		subbatch_masks = self._get_masks(subbatch_lengths)
		subbatch_token_ixs = self._get_subbatch_token_ixs(batch_sizes, subbatch_ixs, subbatch_lengths)
		subbatch_info.append((subbatch_token_ixs, subbatch_lengths, subbatch_sizes, subbatch_masks))
		return subbatch_info

	def _get_masks(self, lengths):
		max_length = lengths.max()
		masks = get_mask_from_lengths(lengths)[:,None,None,:]
		return masks

	def _get_subbatch_token_ixs(self, batch_sizes, subbatch_ixs, subbatch_lengths):
		return [bix+cum_bs
				for t,cum_bs in enumerate([0]+batch_sizes.cumsum(0).tolist()[:-1])
				for bix,l_ in zip(subbatch_ixs,subbatch_lengths)
				if t < l_
				]

	def _is_full_subbatch_size(self, max_length, subbatch_size):
		if 512**2*16 < max_length**2*subbatch_size:
			return True
		else:
			return False

	def pack_init_args(self):
		args = {
			'input_size':self.input_size,
			'hidden_size':self.hidden_size,
			'mlp_hidden_size':self.mlp_hidden_size,
			'num_heads':self.num_heads,
			'num_layers':self.num_layers,
			'dropout':self.dropout
		}
		return args


class SelfAttentionLayer(torch.nn.Module):
	def __init__(self, hidden_size, mlp_hidden_size, num_heads=8, dropout=0.0):
		super(SelfAttentionLayer, self).__init__()
		hidden_size_per_head = hidden_size // num_heads
		self.to_query = LinearSplit(hidden_size, hidden_size_per_head, num_heads)
		self.to_key = LinearSplit(hidden_size, hidden_size_per_head, num_heads)
		self.to_value = LinearSplit(hidden_size, hidden_size_per_head, num_heads)

		self.attention = DotProductAttention(dropout)
		self.linear_combine_heads = torch.nn.Linear(hidden_size_per_head*num_heads, hidden_size)
		self.top_feedfoward = MLP(hidden_size, mlp_hidden_size, hidden_size, nonlinearity='GELU')
		self.dropout = torch.nn.Dropout(dropout)
		self.layer_norm = torch.nn.LayerNorm(hidden_size)

	def forward(self, input_as_dict):
		flatten_input = input_as_dict['values']
		subbatch_info = input_as_dict['subbatch_info']
		query = torch.stack(self.to_query(flatten_input), dim=-2)
		key = torch.stack(self.to_key(flatten_input), dim=-2)
		value = torch.stack(self.to_value(flatten_input), dim=-2)
		attention = []
		for subbatch_token_ixs, subbatch_lengths, subbatch_sizes, subbatch_masks in subbatch_info:
			q,k,v = [pad_flatten_sequence(
						x[subbatch_token_ixs,...],
						subbatch_sizes,
						batch_first=True
					)[0].transpose(1,2).contiguous() # subbatch_size x num_heads x max_lengths x hidden_size_per_head
					for x in [query, key, value]
					]
			a = self.attention(q, k, v, mask=subbatch_masks)
			a = a.transpose(1,2).contiguous().view(a.size(0), a.size(2), -1)
			attention += [a_per_seq[:l,...] for a_per_seq,l in zip(a, subbatch_lengths)]
		attention = torch.nn.utils.rnn.pack_sequence(attention).data
		attention = self.linear_combine_heads(attention)
		attention = self.dropout(attention)
		attention = self.layer_norm(flatten_input + attention)

		out = self.top_feedfoward(attention)
		out = self.dropout(out)
		out = self.layer_norm(attention + out)
		input_as_dict['values'] = out
		return input_as_dict



class LinearSplit(torch.nn.Module):
	def __init__(self, input_size, output_size, num_splits):
		super(LinearSplit, self).__init__()
		self.linears = torch.nn.ModuleList([
								torch.nn.Linear(input_size, output_size)
								for ix in range(num_splits)
							])
		
	def forward(self, x):
		return [l(x) for l in self.linears]

class DotProductAttention(torch.nn.Module):
	def __init__(self, dropout=0.0):
		super(DotProductAttention, self).__init__()
		self.softmax = torch.nn.Softmax(dim=-1)
		self.dropout = torch.nn.Dropout(dropout)

	def forward(self, query, key, value, mask=None, memory_length=None):
		"""
		query: batch_size (x num_heads) x length_1 x hidden_size
		key, value: batch_size (x num_heads) x length_2 x hidden_size
		memory_length: None or 1-dim Tensor of batch_size. (length_2 = max(memory_length))
		"""
		weight = query.matmul(key.transpose(-2,-1))
		weight /= math.sqrt(key.size(-1)) # batch_size (x num_heads) x length_1 x length_2
		if not memory_length is None:
			mask = get_mask_from_lengths(memory_length).view((mask.size(0),)+(1,)*(weight.dim()-2)+(mask.size(-1),))
		if mask is None:
			weight = weight.masked_fill(mask, -float('inf'))
		weight = self.softmax(weight)
		weight = self.dropout(weight)
		out = weight.matmul(value)
		return out



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

class GELU(torch.jit.ScriptModule):
	"""
	Paper Section 3.4, last paragraph notice that BERT used the GELU instead of RELU
	Copied from BERT-pytorch:
	https://github.com/codertimo/BERT-pytorch/blob/master/bert_pytorch/model/utils/gelu.py
	"""
	__constants__ = ['sqrt_2_over_pi']
	def __init__(self):
		super(GELU, self).__init__()
		self.sqrt_2_over_pi = math.sqrt(2 / math.pi)

	@torch.jit.script_method
	def forward(self, x):
		return 0.5 * x * (1 + torch.tanh(self.sqrt_2_over_pi * (x + 0.044715 * torch.pow(x, 3))))

class MLP(torch.jit.ScriptModule):
# class MLP(torch.nn.Module):
	"""
	Multi-Layer Perceptron.
	"""
	def __init__(self, input_size, hidden_size, output_size, nonlinearity='Tanh'):
		super(MLP, self).__init__()
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.output_size = output_size
		if nonlinearity=='GELU':
			nonlinearity = GELU()
		else:
			nonlinearity = getattr(torch.nn, nonlinearity)()
		self.whole_network = torch.nn.Sequential(
			torch.nn.Linear(input_size, hidden_size),
			nonlinearity,
			torch.nn.Linear(hidden_size, output_size)
			)

	@torch.jit.script_method
	def forward(self, batched_input):
		return self.whole_network(batched_input)

	def pack_init_args(self):
		init_args = {
			'input_size':self.input_size,
			'hidden_size':self.hidden_size,
			'output_size':self.output_size
		}
		return init_args

class ESN(torch.jit.ScriptModule):
	__constants__ = ['leak','hidden_size']
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


	def forward(self, packed_input, h_0=None):
		flatten_input = packed_input.data
		last_hidden = []
		if h_0 is None:
			h_0 = self.init_hidden(packed_input.batch_sizes[0])
			if self.bidirectional:
				h_0 = torch.cat([
						h_0,
						self.init_hidden(packed_input.batch_sizes[0])
				], dim=0)
			h_0 = h_0.to(packed_input.data.device)
		h_0 = h_0.view(-1, self.num_layers, h_0.size(1), h_0.size(2))
		for l in range(self.num_layers):
			flatten_hidden, last_hidden_l = self._forward_per_layer(flatten_input, packed_input.batch_sizes, getattr(self, 'weight_ih_l{l}'.format(l=l)), getattr(self, 'weight_hh_l{l}'.format(l=l)), h_0[0,l])
			last_hidden += [last_hidden_l]
			if self.bidirectional:
				flatten_hidden_back, last_hidden_l_back = self._forward_per_layer_backward(flatten_input, packed_input.batch_sizes, getattr(self, 'weight_ih_l{l}_reverse'.format(l=l)), getattr(self, 'weight_hh_l{l}_reverse'.format(l=l)), h_0[1,l])
				flatten_hidden = torch.cat([flatten_hidden, flatten_hidden_back], dim=-1)
				last_hidden += [last_hidden_l_back]
			flatten_input = self.drop(flatten_hidden)
		last_hidden = torch.cat(last_hidden, dim=0)
		return flatten_hidden, last_hidden

	@torch.jit.script_method
	def _forward_per_layer(self, flatten_input, batch_sizes, weight_ih, weight_hh, h_0):
		# input2hidden
		input2hidden_transposed = weight_ih.mm(flatten_input.t())
		# hidden2hidden
		hidden_transposed = h_0.t()
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

	def _forward_per_layer_backward(self, flatten_input, batch_sizes, weight_ih, weight_hh, h_0):
		# input2hidden
		input2hidden_transposed = weight_ih.mm(flatten_input.t())
		# hidden2hidden
		init_hidden_fullsize_transposed = h_0.t()
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
		return torch.zeros((self.num_layers, batch_size, self.hidden_size), requires_grad=False)

class ESNCell(torch.jit.ScriptModule):
	__constants__ = ['leak','hidden_size']
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


	def init_hidden(self, batch_size):
		return torch.zeros((batch_size, self.hidden_size), requires_grad=False)


class TakeMean(torch.nn.Module):
	def forward(self, packed_input):
		padded_input, lengths = torch.nn.utils.rnn.pad_packed_sequence(packed_input, batch_first=True)
		lengths = lengths.to(padded_input.device)
		out = torch.stack([seq[:l].mean(dim=0) for seq,l in zip(padded_input,lengths)])
		out = torch.cat([out, lengths.float().view(-1,1)], dim=-1)
		return out

	def pack_init_args(self):
		return {}

class TakeMedian(torch.nn.Module):
	def forward(self, packed_input):
		padded_input, lengths = torch.nn.utils.rnn.pad_packed_sequence(packed_input, batch_first=True)
		lengths = lengths.to(padded_input.device)
		out = torch.stack([seq[:l].median(dim=0)[0] for seq,l in zip(padded_input,lengths)])
		out = torch.cat([out, lengths.float().view(-1,1)], dim=-1)
		return out

	def pack_init_args(self):
		return {}

class Resample(torch.nn.Module):
	def __init__(self, num_samples, no_length=False):
		super(Resample, self).__init__()
		self.num_samples = num_samples
		self.no_length = no_length
		
	def forward(self, packed_input):
		padded_input, lengths = torch.nn.utils.rnn.pad_packed_sequence(packed_input, batch_first=True)
		lengths = lengths.to(padded_input.device)
		out = torch.stack([self.resample(seq[:l]) for seq,l in zip(padded_input,lengths)])
		if not self.no_length:
			out = torch.cat([out, lengths.float().view(-1,1)], dim=-1)
		return out

	def resample(self, seq):
		gcd = math.gcd(seq.size(0), self.num_samples)
		lcm = seq.size(0) * self.num_samples // gcd

		upsample = torch.nn.Upsample(size=lcm, mode='linear', align_corners=False)
		seq = upsample(
				seq.t().view(1,seq.size(1),seq.size(0))
				).view(seq.size(1),-1).t()

		step_size = seq.size(0) // self.num_samples
		return seq[::step_size].contiguous().view(-1)

	def pack_init_args(self):
		return {'num_samples':self.num_samples, 'no_length':self.no_length}