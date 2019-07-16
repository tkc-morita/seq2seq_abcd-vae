# coding: utf-8

import torch
from modules.data_utils import Compose
import numpy as np
from modules import model, data_utils
from logging import getLogger,FileHandler,DEBUG,Formatter
import os, argparse, itertools

logger = getLogger(__name__)

def update_log_handler(file_dir):
	current_handlers=logger.handlers[:]
	for h in current_handlers:
		logger.removeHandler(h)
	log_file_path = os.path.join(file_dir,'history.log')
	if os.path.isfile(log_file_path):
		retrieval = True
	else:
		retrieval = False
	handler = FileHandler(filename=log_file_path)	#Define the handler.
	handler.setLevel(DEBUG)
	formatter = Formatter('{asctime} - {levelname} - {message}', style='{')	#Define the log format.
	handler.setFormatter(formatter)
	logger.setLevel(DEBUG)
	logger.addHandler(handler)	#Register the handler for the logger.
	if retrieval:
		logger.info("LEARNING RETRIEVED.")
	else:
		logger.info("Logger set up.")
		logger.info("PyTorch ver.: {ver}".format(ver=torch.__version__))
	return retrieval,log_file_path



class Learner(object):
	def __init__(self,
			input_size,
			encoder_rnn_hidden_size,
			decoder_rnn_hidden_size,
			mlp_hidden_size,
			feature_size,
			num_clusters,
			save_dir,
			encoder_rnn_type='LSTM',
			decoder_rnn_type='LSTM',
			encoder_rnn_layers=1,
			bidirectional_encoder=True,
			bidirectional_decoder=False,
			right2left_decoder_weight=0.5,
			encoder_hidden_dropout = 0.0,
			decoder_input_dropout = 0.0,
			device=False,
			seed=1111,
			emission_distribution='isotropic_gaussian',
			decoder_self_feedback=True,
			esn_leak=1.0,
			relax_scalar=0.05,
			prior_base_counts = 1.0,
			p_z_mean = None,
			p_z_sd = 1.0,
			hierarchical_noise=False,
			post_mixture_noise_prior_sd=1.0,
			posterior_base_counts=None,
			num_speakers=None,
			speaker_embed_dim=None
			):
		self.retrieval,self.log_file_path = update_log_handler(save_dir)
		if torch.cuda.is_available():
			if device.startswith('cuda'):
				logger.info('CUDA Version: {version}'.format(version=torch.version.cuda))
				if torch.backends.cudnn.enabled:
					logger.info('cuDNN Version: {version}'.format(version=torch.backends.cudnn.version()))
			else:
				print('CUDA is available. Restart with option -C or --cuda to activate it.')

		self.bce_with_logits_loss = torch.nn.BCEWithLogitsLoss(reduction='sum')

		self.save_dir = save_dir

		self.device = torch.device(device)
		logger.info('Device: {device}'.format(device=device))

		torch.backends.cudnn.deterministic = True
		torch.backends.cudnn.benchmark = False

		if self.retrieval:
			self.last_epoch = self.retrieve_model(device=device)
			logger.info('Model retrieved.')
		else:
			torch.manual_seed(seed)
			torch.cuda.manual_seed_all(seed) # According to the docs, "It’s safe to call this function if CUDA is not available; in that case, it is silently ignored."
			self.emission_distribution =  emission_distribution
			emission_sampler,self.log_pdf_emission,_ = model.choose_distribution(self.emission_distribution)
			if encoder_hidden_dropout > 0.0 and encoder_rnn_layers==1:
				logger.warning('Non-zero dropout cannot be used for the single-layer encoder RNN (because there is no non-top hidden layers).')
				logger.info('encoder_hidden_dropout reset from {do} to 0.0.'.format(do=encoder_hidden_dropout))
				encoder_hidden_dropout = 0.0
			if p_z_mean is None:
				num_cat_bit_length = num_clusters.bit_length() - 1
				logger.info('# of categories is changed from {num_clusters} to 2**{num_cat_bit_length}.'.format(num_clusters=num_clusters, num_cat_bit_length=num_cat_bit_length))
				logger.info('feature_size is changed from {feature_size} to {num_cat_bit_length}'.format(feature_size=feature_size, num_cat_bit_length=num_cat_bit_length))
				num_clusters = 2**num_cat_bit_length
				feature_size = num_cat_bit_length
				p_z_mean = torch.tensor(list(itertools.product((-2.0,2.0), repeat=feature_size)))
				logger.info('The means of the mixture components are in {{-2.0, 2.0}}^{feature_size}'.format(feature_size=feature_size))
				p_z_sd = torch.ones_like(p_z_mean)
				logger.info('The standard deviation of the mixture components is 1.0 for all the dimensions.'.format(feature_size=feature_size))
			self.encoder = model.RNN_Variational_Encoder(input_size, encoder_rnn_hidden_size, rnn_type=encoder_rnn_type, rnn_layers=encoder_rnn_layers, hidden_dropout=encoder_hidden_dropout, bidirectional=bidirectional_encoder, esn_leak=esn_leak)
			self.mixture_ratio_sampler = model.SampleFromDirichlet(num_clusters, self.encoder.hidden_size_total, mlp_hidden_size, relax_scalar=relax_scalar, prior_base_counts = prior_base_counts, posterior_base_counts=posterior_base_counts)
			self.mixture_components = model.SampleFromIsotropicGaussianMixture(p_z_mean, p_z_sd, num_clusters=num_clusters, ndim=feature_size, post_mixture_noise=hierarchical_noise, post_mixture_noise_prior_sd=post_mixture_noise_prior_sd, mlp_input_size=self.encoder.hidden_size_total, mlp_hidden_size=mlp_hidden_size)
			self.decoder = model.RNN_Variational_Decoder(input_size, decoder_rnn_hidden_size, mlp_hidden_size, feature_size, emission_sampler, rnn_type=decoder_rnn_type, input_dropout=decoder_input_dropout, self_feedback=decoder_self_feedback, esn_leak=esn_leak, bidirectional=bidirectional_decoder, num_speakers=num_speakers, speaker_embed_dim=speaker_embed_dim)
			self.parameters = lambda:itertools.chain(self.encoder.parameters(), self.mixture_ratio_sampler.parameters(), self.mixture_components.parameters(), self.decoder.parameters())
			logger.info('Data to be encoded into {feature_size}-dim features.'.format(feature_size=feature_size))
			logger.info('Features are assumed to be distributed according to mixture of isotropic Gaussians.')
			logger.info("# of mixture components: {num_clusters}".format(num_clusters=num_clusters))
			logger.info("Prior distribution of the mixture ratio, pi, is Dirichlet({prior_base_counts}).".format(prior_base_counts=prior_base_counts))
			logger.info("Assignments to the mixture components are continuously relaxed by samples from Drichlet({relax_scalar} * pi)".format(relax_scalar=relax_scalar))
			if posterior_base_counts is None:
				posterior_base_counts = '0.1 / {num_clusters}'.format(num_clusters=num_clusters)
			logger.info("To avoid underflow, the approximated Dirichlet posterior of the relaxed assignments has the base count: {posterior_base_counts}".format(posterior_base_counts=posterior_base_counts))
			if hierarchical_noise:
				logger.info('Another level of noise is added to the mixed features in prior according to the standard multivariate Gaussian N(0, {post_mixture_noise_prior_sd}*I).'.format(post_mixture_noise_prior_sd=post_mixture_noise_prior_sd))
				logger.info('Posterior on the individual feature distribution is approximated by an isotropic Gaussian.')
			logger.info('Conditioned on the features, data are assumed to be distributed according to {emission_distribution}'.format(emission_distribution=emission_distribution))
			logger.info('Random seed: {seed}'.format(seed = seed))
			logger.info('Type of RNN used for the encoder: {rnn_type}'.format(rnn_type=encoder_rnn_type))
			logger.info('Type of RNN used for the decoder: {rnn_type}'.format(rnn_type=decoder_rnn_type))
			logger.info("# of RNN hidden layers in the encoder RNN: {hl}".format(hl=encoder_rnn_layers))
			logger.info("# of hidden units in the encoder RNNs: {hs}".format(hs=encoder_rnn_hidden_size))
			logger.info("# of hidden units in the decoder RNNs: {hs}".format(hs=decoder_rnn_hidden_size))
			logger.info("# of hidden units in the MLPs: {hs}".format(hs=mlp_hidden_size))
			logger.info("Encoder is bidirectional: {bidirectional_encoder}".format(bidirectional_encoder=bidirectional_encoder))
			logger.info("Decoder is bidirectional: {bidirectional_decoder}".format(bidirectional_decoder=bidirectional_decoder))
			if bidirectional_decoder:
				logger.info("Probability of emission by the right-to-left decoder: {p}".format(p=right2left_decoder_weight))
				self.log_left2right_decoder_weight = torch.tensor(1 - right2left_decoder_weight).log().item()
				self.log_right2left_decoder_weight = torch.tensor(right2left_decoder_weight).log().item()
			else:
				self.log_left2right_decoder_weight = 0.0
				self.log_right2left_decoder_weight = None
			logger.info("Dropout rate in the non-top layers of the encoder RNN: {do}".format(do=encoder_hidden_dropout))
			logger.info("Self-feedback to the decoder: {decoder_self_feedback}".format(decoder_self_feedback=decoder_self_feedback))
			if decoder_self_feedback:
				logger.info("Dropout rate in the input to the decoder RNN: {do}".format(do=decoder_input_dropout))
			if encoder_rnn_type == 'ESN' or decoder_rnn_type == 'ESN':
				logger.info('ESN leak: {leak}'.format(leak=esn_leak))
			if not speaker_embed_dim is None:
				logger.info("Speaker ID # is embedded and fed to the decoder.")
				logger.info("# of speakers: {num_speakers}".format(num_speakers=num_speakers))
				logger.info("Embedding dimension: {speaker_embed_dim}".format(speaker_embed_dim=speaker_embed_dim))


		self.encoder.to(self.device)
		self.mixture_ratio_sampler.to(self.device)
		self.mixture_components.to(self.device)
		self.decoder.to(self.device)
		




	def train(self, dataloader):
		"""
		Training phase. Updates weights.
		"""
		self.encoder.train() # Turn on training mode which enables dropout.
		self.mixture_ratio_sampler.train()
		self.mixture_components.train()
		self.decoder.train()
		self.bce_with_logits_loss.train()

		emission_loss = 0
		end_prediction_loss = 0
		kl_loss = 0

		num_batches = dataloader.get_num_batches()
		num_strings = len(dataloader.dataset)

		for batch_ix,(packed_input, is_offset, speaker, _) in enumerate(dataloader, 1):
			packed_input = packed_input.to(self.device)
			is_offset = is_offset.to(self.device)
			speaker = speaker.to(self.device)

			self.optimizer.zero_grad()

			last_hidden = self.encoder(packed_input)
			cluster_weights,kl_weight,_ = self.mixture_ratio_sampler(last_hidden, num_strings)
			features,kl_value,_ = self.mixture_components(cluster_weights, num_strings, parameter_seed=last_hidden)
			kl_loss_per_batch = kl_weight + kl_value
			emission_loss_per_batch = []
			end_prediction_loss_per_batch = []
			for (emission_params,flatten_offset_prediction,_), log_loss_weight in zip(self.decoder(features, batch_sizes=packed_input.batch_sizes, speaker=speaker), (self.log_left2right_decoder_weight, self.log_right2left_decoder_weight)):
				emission_loss_per_batch += [-self.log_pdf_emission(packed_input.data, *emission_params) + log_loss_weight]
				end_prediction_loss_per_batch += [self.bce_with_logits_loss(flatten_offset_prediction, is_offset.data) + log_loss_weight]
			emission_loss_per_batch = torch.stack(emission_loss_per_batch).logsumexp(dim=0)
			end_prediction_loss_per_batch = torch.stack(end_prediction_loss_per_batch).logsumexp(dim=0)

			loss = emission_loss_per_batch + end_prediction_loss_per_batch + kl_loss_per_batch
			(loss / packed_input.batch_sizes[0]).backward()

			# `clip_grad_norm` helps prevent the exploding gradient problem in RNNs.
			torch.nn.utils.clip_grad_norm_(self.parameters(), self.gradient_clip)

			self.optimizer.step()

			emission_loss += emission_loss_per_batch.item()
			end_prediction_loss += end_prediction_loss_per_batch.item()
			kl_loss += kl_loss_per_batch.item()

			logger.info('{batch_ix}/{num_batches} training batches complete.'.format(batch_ix=batch_ix, num_batches=num_batches))

		emission_loss /= num_strings
		end_prediction_loss /= num_strings
		kl_loss /= num_strings
		mean_loss = emission_loss + end_prediction_loss + kl_loss
		logger.info('mean training emission negative pdf loss (per string): {:5.4f}'.format(emission_loss))
		logger.info('mean training end-prediction loss (per string): {:5.4f}'.format(end_prediction_loss))
		logger.info('mean training KL (per string): {:5.4f}'.format(kl_loss))
		logger.info('mean training total loss (per string): {:5.4f}'.format(mean_loss))


	def test_or_validate(self, dataloader, train_data_size):
		"""
		Test/validation phase. No update of weights.
		"""
		self.encoder.eval() # Turn on evaluation mode which disables dropout.
		self.mixture_ratio_sampler.eval()
		self.mixture_components.eval()
		self.decoder.eval()
		self.bce_with_logits_loss.eval()

		emission_loss = 0
		end_prediction_loss = 0
		kl_loss = 0
		sign_loss = 0

		num_batches = dataloader.get_num_batches()

		with torch.no_grad():
			for batch_ix, (packed_input, is_offset, speaker, _) in enumerate(dataloader, 1):
				packed_input = packed_input.to(self.device)
				is_offset = is_offset.to(self.device)
				speaker = speaker.to(self.device)

				last_hidden = self.encoder(packed_input)
				cluster_weights,kl_weight,_ = self.mixture_ratio_sampler(last_hidden, train_data_size)
				features,kl_value,_ = self.mixture_components(cluster_weights, train_data_size, parameter_seed=last_hidden)
				kl_loss += kl_weight + kl_value
				emission_loss_per_batch = []
				end_prediction_loss_per_batch = []
				for (emission_params,flatten_offset_prediction,_), log_loss_weight in zip(self.decoder(features, batch_sizes=packed_input.batch_sizes, speaker=speaker), (self.log_left2right_decoder_weight, self.log_right2left_decoder_weight)):
					emission_loss_per_batch += [-self.log_pdf_emission(packed_input.data, *emission_params) + log_loss_weight]
					end_prediction_loss_per_batch += [self.bce_with_logits_loss(flatten_offset_prediction, is_offset.data) + log_loss_weight]
				emission_loss += torch.stack(emission_loss_per_batch).logsumexp(dim=0)
				end_prediction_loss += torch.stack(end_prediction_loss_per_batch).logsumexp(dim=0)

				logger.info('{batch_ix}/{num_batches} validation batches complete.'.format(batch_ix=batch_ix, num_batches=num_batches))

		num_strings = len(dataloader.dataset)
		emission_loss /= num_strings
		end_prediction_loss /= num_strings
		kl_loss /= num_strings
		mean_loss = emission_loss + end_prediction_loss + kl_loss
		logger.info('mean validation emission negative pdf loss (per string): {:5.4f}'.format(emission_loss))
		logger.info('mean validation end-prediction loss (per string): {:5.4f}'.format(end_prediction_loss))
		logger.info('mean validation KL (per string): {:5.4f}'.format(kl_loss))
		logger.info('mean validation total loss (per string): {:5.4f}'.format(mean_loss))
		return mean_loss




	def learn(self, train_dataset, valid_dataset, num_epochs, batch_size_train, batch_size_valid, learning_rate=0.1, momentum= 0.9, gradient_clip = 0.25, patience=0):
		train_dataloader = data_utils.DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True)
		valid_dataloader = data_utils.DataLoader(valid_dataset, batch_size=batch_size_valid)
		if self.retrieval:
			initial_epoch = self.last_epoch + 1
			logger.info('To be restarted from the beginning of epoch #: {epoch}'.format(epoch=initial_epoch))
		else:
			self.optimizer = torch.optim.SGD(self.parameters(), lr=learning_rate, momentum=momentum)
			self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=patience)
			logger.info("START LEARNING.")
			logger.info("max # of epochs: {ep}".format(ep=num_epochs))
			logger.info("batch size for training data: {size}".format(size=batch_size_train))
			logger.info("batch size for validation data: {size}".format(size=batch_size_valid))
			logger.info("initial learning rate: {lr}".format(lr=learning_rate))
			logger.info("momentum for SGD: {momentum}".format(momentum=momentum))
			self.gradient_clip = gradient_clip
			logger.info("gradient clipping: {gc}".format(gc=self.gradient_clip))
			initial_epoch = 1
			
		for epoch in range(initial_epoch, num_epochs+1):
			logger.info('START OF EPOCH: {:3d}'.format(epoch))
			logger.info('current learning rate: {lr}'.format(lr=self.optimizer.param_groups[0]['lr']))

			logger.info('start of TRAINING phase.')
			self.train(train_dataloader)
			logger.info('end of TRAINING phase.')

			logger.info('start of VALIDATION phase.')
			mean_valid_loss = self.test_or_validate(valid_dataloader, len(train_dataloader.dataset))
			logger.info('end of VALIDATION phase.')

			self.lr_scheduler.step(mean_valid_loss)
			self.save_model(epoch)
			logger.info('END OF EPOCH: {:3d}'.format(epoch))
			
		logger.info('END OF TRAINING')


	def save_model(self, epoch):
		"""
		Save model config.
		Allow multiple tries to prevent immediate I/O errors.
		"""
		checkpoint = {
			'epoch':epoch,
			'encoder':self.encoder.state_dict(),
			'decoder':self.decoder.state_dict(),
			'mixture_ratio_sampler':self.mixture_ratio_sampler.state_dict(),
			'mixture_components':self.mixture_components.state_dict(),
			'optimizer':self.optimizer.state_dict(),
			'lr_scheduler':self.lr_scheduler.state_dict(),
			'gradient_clip':self.gradient_clip,
			'input_size':self.encoder.rnn.input_size,
			'encoder_rnn_type':self.encoder.rnn.mode.split('_')[0],
			'decoder_rnn_type':self.decoder.rnn_cell.mode,
			'encoder_rnn_hidden_size':self.encoder.rnn.hidden_size,
			'decoder_rnn_hidden_size':self.decoder.rnn_cell.cell.hidden_size,
			'encoder_rnn_layers':self.encoder.rnn.num_layers,
			'bidirectional_encoder':self.encoder.rnn.bidirectional,
			'bidirectional_decoder':self.decoder.bidirectional,
			'log_left2right_decoder_weight':self.log_left2right_decoder_weight,
			'log_right2left_decoder_weight':self.log_right2left_decoder_weight,
			'encoder_hidden_dropout':self.encoder.rnn.dropout,
			'decoder_input_dropout':self.decoder.rnn_cell.drop.p,
			'mlp_hidden_size':self.decoder.offset_predictor.hidden_size,
			'feature_size':self.decoder.feature_size,
			'posterior_base_counts':self.mixture_ratio_sampler.posterior_base_counts,
			'num_clusters':self.mixture_ratio_sampler.num_clusters,
			'relax_scalar':self.mixture_ratio_sampler.relax_scalar,
			'hierarchical_noise':self.mixture_components.post_mixture_noise,
			'emission_distribution':self.emission_distribution,
			'random_state':torch.get_rng_state(),
		}
		if torch.cuda.is_available():
			checkpoint['random_state_cuda'] = torch.cuda.get_rng_state_all()
		if checkpoint['encoder_rnn_type'] == 'ESN':
			checkpoint['esn_leak'] = self.encoder.rnn.leak
		elif checkpoint['decoder_rnn_type'] == 'ESN':
			checkpoint['esn_leak'] = self.decoder.rnn_cell.cell.leak
		if not self.decoder.embed_speaker is None:
			checkpoint['num_speakers'] = self.decoder.embed_speaker.num_embeddings
			checkpoint['speaker_embed_dim'] = self.decoder.embed_speaker.embedding_dim
		torch.save(checkpoint, os.path.join(self.save_dir, 'checkpoint.pt'))
		logger.info('Config successfully saved.')


	def retrieve_model(self, checkpoint_path = None, device='cpu'):
		if checkpoint_path is None:
			checkpoint_path = os.path.join(self.save_dir, 'checkpoint.pt')
		checkpoint = torch.load(checkpoint_path, map_location='cpu') # Random state needs to be loaded to CPU first even when cuda is available.

		input_size = checkpoint['input_size']
		encoder_rnn_type = checkpoint['encoder_rnn_type']
		decoder_rnn_type = checkpoint['decoder_rnn_type']
		encoder_rnn_hidden_size = checkpoint['encoder_rnn_hidden_size']
		decoder_rnn_hidden_size = checkpoint['decoder_rnn_hidden_size']
		encoder_rnn_layers = checkpoint['encoder_rnn_layers']
		bidirectional_encoder = checkpoint['bidirectional_encoder']
		bidirectional_decoder = checkpoint['bidirectional_decoder']
		self.log_left2right_decoder_weight = checkpoint['log_left2right_decoder_weight']
		self.log_right2left_decoder_weight = checkpoint['log_right2left_decoder_weight']
		encoder_hidden_dropout = checkpoint['encoder_hidden_dropout']
		decoder_input_dropout = checkpoint['decoder_input_dropout']
		feature_size = checkpoint['feature_size']
		mlp_hidden_size = checkpoint['mlp_hidden_size']
		num_clusters = checkpoint['num_clusters']
		relax_scalar = checkpoint['relax_scalar']
		posterior_base_counts = checkpoint['posterior_base_counts']
		hierarchical_noise = checkpoint['hierarchical_noise']

		if encoder_rnn_type == 'ESN' or decoder_rnn_type == 'ESN':
			esn_leak = checkpoint['esn_leak']
		else:
			esn_leak = 1.0
		if 'num_speakers' in checkpoint:
			num_speakers = checkpoint['num_speakers']
			speaker_embed_dim = checkpoint['speaker_embed_dim']
		else:
			num_speakers = None
			speaker_embed_dim = None

		self.emission_distribution = checkpoint['emission_distribution']
		emission_sampler,self.log_pdf_emission,_ = model.choose_distribution(self.emission_distribution)

		self.encoder = model.RNN_Variational_Encoder(input_size, encoder_rnn_hidden_size, rnn_type=encoder_rnn_type, rnn_layers=encoder_rnn_layers, bidirectional=bidirectional_encoder, hidden_dropout=encoder_hidden_dropout, esn_leak=esn_leak)
		self.mixture_ratio_sampler = model.SampleFromDirichlet(num_clusters, self.encoder.hidden_size_total, mlp_hidden_size, relax_scalar=relax_scalar, posterior_base_counts=posterior_base_counts)
		self.mixture_components = model.SampleFromIsotropicGaussianMixture(num_clusters=num_clusters, ndim=feature_size, post_mixture_noise=hierarchical_noise, mlp_input_size=self.encoder.hidden_size_total, mlp_hidden_size=mlp_hidden_size)
		self.decoder = model.RNN_Variational_Decoder(input_size, decoder_rnn_hidden_size, mlp_hidden_size, feature_size, emission_sampler, rnn_type=decoder_rnn_type, input_dropout=decoder_input_dropout, esn_leak=esn_leak, bidirectional=bidirectional_decoder, num_speakers=num_speakers, speaker_embed_dim=speaker_embed_dim)
		self.encoder.load_state_dict(checkpoint['encoder'])
		self.mixture_ratio_sampler.load_state_dict(checkpoint['mixture_ratio_sampler'])
		self.mixture_components.load_state_dict(checkpoint['mixture_components'])
		self.decoder.load_state_dict(checkpoint['decoder'])
		self.parameters = lambda:itertools.chain(self.encoder.parameters(), self.mixture_ratio_sampler.parameters(), self.mixture_components.parameters(), self.decoder.parameters())

		self.optimizer = torch.optim.SGD(self.parameters(), lr=0.1)
		self.optimizer.load_state_dict(checkpoint['optimizer'])

		self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer)
		self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

		self.gradient_clip = checkpoint['gradient_clip']
		
		torch.set_rng_state(checkpoint['random_state'])
		if device=='cuda':
			torch.cuda.set_rng_state_all(checkpoint['random_state_cuda'])
		return checkpoint['epoch']





def get_parameters():
	par_parser = argparse.ArgumentParser()

	par_parser.add_argument('input_root', type=str, help='Path to the root directory under which inputs are located.')
	par_parser.add_argument('annotation_file', type=str, help='Path to the annotation csv file.')
	par_parser.add_argument('--annotation_sep', type=str, default=',', help='Separator symbol of the annotation file. Comma "," by default (i.e., csv).')
	par_parser.add_argument('-S', '--save_root', type=str, default=None, help='Path to the directory where results are saved.')
	par_parser.add_argument('-j', '--job_id', type=str, default='NO_JOB_ID', help='Job ID. For users of computing clusters.')
	par_parser.add_argument('-s', '--seed', type=int, default=1111, help='random seed')
	par_parser.add_argument('-d', '--device', type=str, default='cpu', help='Computing device.')
	par_parser.add_argument('-e', '--epochs', type=int, default=40, help='# of epochs to train the model.')
	par_parser.add_argument('-b', '--batch_size', type=int, default=16, help='Batch size for training.')
	par_parser.add_argument('--validation_batch_size', type=int, default=None, help='Batch size for validation. Same as for training by default.')
	par_parser.add_argument('-l', '--learning_rate', type=float, default=1.0, help='Initial learning rate.')
	par_parser.add_argument('-M', '--momentum', type=float, default=0.0, help='Momentum for the storchastic gradient descent.')
	par_parser.add_argument('-c', '--clip', type=float, default=1.0, help='Gradient clipping.')
	par_parser.add_argument('-p', '--patience', type=int, default=0, help='# of epochs before updating the learning rate.')
	par_parser.add_argument('-R', '--encoder_rnn_type', type=str, default='LSTM', help='Name of RNN to be used for the encoder.')
	par_parser.add_argument('--decoder_rnn_type', type=str, default=None, help='Name of RNN to be used for the decoder. Same as the encoder by default.')
	par_parser.add_argument('-f', '--feature_size', type=int, default=13, help='# of dimensions of features into which data are encoded.')
	par_parser.add_argument('--encoder_rnn_layers', type=int, default=1, help='# of hidden layers in the encoder RNN.')
	par_parser.add_argument('--encoder_rnn_hidden_size', type=int, default=100, help='# of the RNN units in the encoder RNN.')
	par_parser.add_argument('--decoder_rnn_hidden_size', type=int, default=100, help='# of the RNN units in the decoder RNN.')
	par_parser.add_argument('--mlp_hidden_size', type=int, default=200, help='# of neurons in the hidden layer of the MLP transforms.')
	par_parser.add_argument('--speaker_embed_dim', type=int, default=None, help='If specified, the decoder receives an embedding of the speaker ID with the specified dim. No embedding by default.')
	par_parser.add_argument('--encoder_hidden_dropout', type=float, default=0.0, help='Dropout rate in the non-top layers of the encoder RNN.')
	par_parser.add_argument('--decoder_input_dropout', type=float, default=0.0, help='Dropout rate in the input to the decoder RNN.')
	par_parser.add_argument('--greedy_decoder', action='store_true', help='If selected, decoder becomes greedy and will not receive self-feedback.')
	par_parser.add_argument('--esn_leak', type=float, default=1.0, help='Leak for the echo-state network. Ignored if the RNN type is not ESN.')
	par_parser.add_argument('--bidirectional_decoder', action='store_true', help='If selected, use the weighted sum of losses from left-to-right and right-to-left decoders (to avoid the uninformative latent variable problem).')
	par_parser.add_argument('--right2left_decoder_weight', type=float, default=0.5, help='The weight of the right-to-left decoder when bidirectional_decoder==True.')
	par_parser.add_argument('-C', '--num_clusters', type=int, default=64, help='Max # of clusters. Currently floored to a power of 2.')
	par_parser.add_argument('--relax_scalar', type=float, default=0.01, help='Concentration of the Dirichlet distribution that relaxes the categorical assignments to the clusters.')
	par_parser.add_argument('--prior_base_counts', type=float, default=[1.0], nargs='+', help='Base counts of the clusters for the prior (i.e., the parameters of the Dirichlet prior of the clusters).')
	par_parser.add_argument('--posterior_base_counts', type=float, default=None, help='Base counts of the clusters for the posterior, which avoids underflow in the computation of Dirichlet gradients. Use 0.1 / num_clusters by default.')
	par_parser.add_argument('-H', '--hierarchical_noise', action='store_true', help='If selected, assume an additional noise to the mixed features of each data point.')
	par_parser.add_argument('--fft_frame_length', type=float, default=0.008, help='FFT frame length in sec.')
	par_parser.add_argument('--fft_step_size', type=float, default=0.004, help='FFT step size in sec.')
	par_parser.add_argument('--fft_window_type', type=str, default='hann_window', help='Window type for FFT. "hann_window" by default.')
	par_parser.add_argument('--fft_no_centering', action='store_true', help='If selected, no centering in FFT.')
	par_parser.add_argument('--channel', type=int, default=0, help='Channel ID # (starting from 0) of multichannel recordings to use.')
	par_parser.add_argument('-N','--data_normalizer', type=float, default=1.0, help='Normalizing constant to devide the data.')
	par_parser.add_argument('-E','--epsilon', type=float, default=2**(-15), help='Small positive real number to add to avoid log(0).')

	return par_parser.parse_args()


def get_save_dir(save_root, job_id_str):
	save_dir = os.path.join(
					save_root,
					job_id_str # + '_START-AT-' + datetime.datetime.now().strftime('%y-%m-%d-%H-%M-%S-%f')
				)
	if not os.path.isdir(save_dir):
		os.makedirs(save_dir)
	return save_dir

if __name__ == '__main__':
	parameters = get_parameters()

	save_root = parameters.save_root
	if save_root is None:
		save_root = parameters.input_root
	save_dir = get_save_dir(save_root, parameters.job_id)

	data_parser = data_utils.Data_Parser(parameters.input_root, parameters.annotation_file, annotation_sep=parameters.annotation_sep)
	fs = data_parser.get_sample_freq() # Assuming all the wav files have the same fs, get the 1st file's.
	num_speakers = data_parser.get_num_speakers()

	fft_frame_length = int(np.floor(parameters.fft_frame_length * fs))
	fft_step_size = int(np.floor(parameters.fft_step_size * fs))

	if parameters.decoder_rnn_type is None:
		parameters.decoder_rnn_type = parameters.encoder_rnn_type

	if len(parameters.prior_base_counts)==1:
		prior_base_counts = parameters.prior_base_counts[0]
	else:
		prior_base_counts = torch.tensor(parameters.prior_base_counts)

	# Get a model.
	learner = Learner(
				int(fft_frame_length / 2 + 1),
				parameters.encoder_rnn_hidden_size,
				parameters.decoder_rnn_hidden_size,
				parameters.mlp_hidden_size,
				parameters.feature_size,
				parameters.num_clusters,
				save_dir,
				encoder_rnn_type=parameters.encoder_rnn_type,
				decoder_rnn_type=parameters.decoder_rnn_type,
				encoder_rnn_layers=parameters.encoder_rnn_layers,
				encoder_hidden_dropout=parameters.encoder_hidden_dropout,
				decoder_input_dropout=parameters.decoder_input_dropout,
				device = parameters.device,
				seed = parameters.seed,
				decoder_self_feedback=not parameters.greedy_decoder,
				bidirectional_decoder=parameters.bidirectional_decoder,
				right2left_decoder_weight=parameters.right2left_decoder_weight,
				prior_base_counts=prior_base_counts,
				relax_scalar=parameters.relax_scalar,
				hierarchical_noise=parameters.hierarchical_noise,
				posterior_base_counts=parameters.posterior_base_counts,
				num_speakers=num_speakers,
				speaker_embed_dim=parameters.speaker_embed_dim
				)

	to_tensor = data_utils.ToTensor()
	stft = data_utils.STFT(fft_frame_length, fft_step_size, window=parameters.fft_window_type, centering=not parameters.fft_no_centering)
	log_and_normalize = data_utils.Transform(lambda x: (x + parameters.epsilon).log() / parameters.data_normalizer)
	logger.info("log(abs(STFT(wav))) + {eps}) / {normalizer} will be the input.".format(eps=parameters.epsilon, normalizer=parameters.data_normalizer))
	logger.info("Sampling frequency of data: {fs}".format(fs=fs))
	logger.info("STFT window type: {fft_window}".format(fft_window=parameters.fft_window_type))
	logger.info("STFT frame lengths: {fft_frame_length_in_sec} sec".format(fft_frame_length_in_sec=parameters.fft_frame_length))
	logger.info("STFT step size: {fft_step_size_in_sec} sec".format(fft_step_size_in_sec=parameters.fft_step_size))


	train_dataset = data_parser.get_data(data_type='train', transform=Compose([to_tensor,stft,log_and_normalize]), channel=parameters.channel)
	valid_dataset = data_parser.get_data(data_type='valid', transform=Compose([to_tensor,stft,log_and_normalize]), channel=parameters.channel)
	

	if parameters.validation_batch_size is None:
		parameters.validation_batch_size = parameters.batch_size

	# Train the model.
	learner.learn(
			train_dataset,
			valid_dataset,
			parameters.epochs,
			parameters.batch_size,
			parameters.validation_batch_size,
			learning_rate=parameters.learning_rate,
			momentum=parameters.momentum,
			gradient_clip = parameters.clip,
			patience = parameters.patience
			)