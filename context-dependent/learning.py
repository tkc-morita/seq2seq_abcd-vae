# coding: utf-8

import torch
from modules.data_utils import Compose
import numpy as np
from modules import model, data_utils
from logging import getLogger,FileHandler,DEBUG,Formatter
import os, argparse, itertools, collections

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
			feature_distribution='isotropic_gaussian',
			emission_distribution='isotropic_gaussian',
			decoder_self_feedback=True,
			esn_leak=1.0,
			num_speakers=None,
			speaker_embed_dim=None,
			context_feature_size=None,
			frame_embedding=False,
			frame_feature_size=8,
			frame_conv_stride=2,
			frame_conv_kernel_size=3,
			):
		self.retrieval,self.log_file_path = update_log_handler(save_dir)
		if torch.cuda.is_available():
			if device.startswith('cuda'):
				logger.info('CUDA Version: {version}'.format(version=torch.version.cuda))
				if torch.backends.cudnn.enabled:
					logger.info('cuDNN Version: {version}'.format(version=torch.backends.cudnn.version()))
			else:
				print('CUDA is available. Restart with option -C or --cuda to activate it.')


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
			if encoder_hidden_dropout > 0.0 and encoder_rnn_layers==1:
				logger.warning('Non-zero dropout cannot be used for the single-layer encoder RNN (because there is no non-top hidden layers).')
				logger.info('encoder_hidden_dropout reset from {do} to 0.0.'.format(do=encoder_hidden_dropout))
				encoder_hidden_dropout = 0.0
			self.modules = []
			if frame_embedding:
				num_cnn_layers = (input_size // frame_feature_size).bit_length() - 1
				cnn_last_out_size = input_size
				padding_size = frame_conv_kernel_size // 2
				for l in range(num_cnn_layers):
					cnn_last_out_size = int(np.floor((cnn_last_out_size + 2*padding_size - frame_conv_kernel_size)/frame_conv_stride + 1))
				self.frame_encoder = model.CNN_Variational_Encoder(num_cnn_layers, stride=frame_conv_stride, kernel_size=frame_conv_kernel_size)
				self.frame_feature_sampler = model.Sampler(cnn_last_out_size, mlp_hidden_size, frame_feature_size, distribution_name=feature_distribution)
				self.frame_decoder = model.MLP_Variational_Decoder(input_size, mlp_hidden_size, frame_feature_size, emission_distr_name=emission_distribution, num_speakers=num_speakers, speaker_embed_dim=speaker_embed_dim)
				self.frame_encoder.to(self.device)
				self.frame_feature_sampler.to(self.device)
				self.frame_decoder.to(self.device)
				logger.info('Input frames are embedded into {frame_feature_size}-dim feature space by {num_cnn_layers}-layer CNN and MLP VAE.'.format(frame_feature_size=frame_feature_size, num_cnn_layers=num_cnn_layers))
				logger.info('CNN stride: {frame_conv_stride}'.format(frame_conv_stride=frame_conv_stride))
				logger.info('CNN kernel_size: {frame_conv_kernel_size}'.format(frame_conv_kernel_size=frame_conv_kernel_size))
				self.modules += [self.frame_encoder, self.frame_feature_sampler, self.frame_decoder]
				rnn_input_size = frame_feature_size
			else:
				self.frame_encoder = None
				self.frame_feature_sampler = None
				self.frame_decoder = None
				rnn_input_size = input_size
			self.encoder = model.RNN_Variational_Encoder(rnn_input_size, encoder_rnn_hidden_size, rnn_type=encoder_rnn_type, rnn_layers=encoder_rnn_layers, hidden_dropout=encoder_hidden_dropout, bidirectional=bidirectional_encoder, esn_leak=esn_leak)
			logger.info('Data to be encoded into {feature_size}-dim features.'.format(feature_size=feature_size))
			logger.info('Features are assumed to be distributed according to {feature_distribution}.'.format(feature_distribution=feature_distribution))
			if context_feature_size is None:
				context_feature_size = feature_size
			self.feature_sampler = model.Sampler(self.encoder.hidden_size_total, mlp_hidden_size, feature_size, distribution_name=feature_distribution)
			self.decoder = model.RNN_Variational_Decoder(rnn_input_size, decoder_rnn_hidden_size, mlp_hidden_size, feature_size+context_feature_size*2, emission_distr_name=emission_distribution, rnn_type=decoder_rnn_type, input_dropout=decoder_input_dropout, self_feedback=decoder_self_feedback, esn_leak=esn_leak, bidirectional=bidirectional_decoder, right2left_weight=right2left_decoder_weight, num_speakers=num_speakers, speaker_embed_dim=speaker_embed_dim)
			self.encoder.to(self.device)
			self.feature_sampler.to(self.device)
			self.decoder.to(self.device)
			self.modules += [self.encoder, self.feature_sampler, self.decoder]
			self.prefix_encoder = None
			self.suffix_encoder = None
			self.prefix_frame_decoder = None
			self.prefix_frame_feature_sampler = None
			self.prefix_frame_decoder = None
			self.suffix_frame_decoder = None
			self.suffix_frame_feature_sampler = None
			self.suffix_frame_decoder = None
			if context_feature_size>0:
				if frame_embedding:
					self.prefix_frame_encoder = model.CNN_Variational_Encoder(num_cnn_layers, stride=frame_conv_stride, kernel_size=frame_conv_kernel_size)
					self.prefix_frame_feature_sampler = model.Sampler(cnn_last_out_size, mlp_hidden_size, frame_feature_size, distribution_name=feature_distribution)
					self.prefix_frame_decoder = model.MLP_Variational_Decoder(input_size, mlp_hidden_size, frame_feature_size, emission_distr_name=emission_distribution, num_speakers=num_speakers, speaker_embed_dim=speaker_embed_dim)
					self.prefix_frame_encoder.to(self.device)
					self.prefix_frame_feature_sampler.to(self.device)
					self.prefix_frame_decoder.to(self.device)
					self.modules += [self.prefix_frame_encoder, self.prefix_frame_decoder, self.prefix_frame_feature_sampler]

					self.suffix_frame_encoder = model.CNN_Variational_Encoder(num_cnn_layers, stride=frame_conv_stride, kernel_size=frame_conv_kernel_size)
					self.suffix_frame_feature_sampler = model.Sampler(cnn_last_out_size, mlp_hidden_size, frame_feature_size, distribution_name=feature_distribution)
					self.suffix_frame_decoder = model.MLP_Variational_Decoder(input_size, mlp_hidden_size, frame_feature_size, emission_distr_name=emission_distribution, num_speakers=num_speakers, speaker_embed_dim=speaker_embed_dim)
					self.suffix_frame_encoder.to(self.device)
					self.suffix_frame_feature_sampler.to(self.device)
					self.suffix_frame_decoder.to(self.device)
					self.modules += [self.suffix_frame_encoder, self.suffix_frame_decoder, self.suffix_frame_feature_sampler]
				prefix_encoder = model.RNN_Variational_Encoder(rnn_input_size, encoder_rnn_hidden_size, rnn_type=encoder_rnn_type, rnn_layers=encoder_rnn_layers, hidden_dropout=encoder_hidden_dropout, bidirectional=bidirectional_encoder, esn_leak=esn_leak)
				self.prefix_encoder = torch.nn.Sequential(collections.OrderedDict([
					('rnn', prefix_encoder),
					('mlp', model.MLP(prefix_encoder.hidden_size_total, mlp_hidden_size, context_feature_size))
				]))
				suffix_encoder = model.RNN_Variational_Encoder(rnn_input_size, encoder_rnn_hidden_size, rnn_type=encoder_rnn_type, rnn_layers=encoder_rnn_layers, hidden_dropout=encoder_hidden_dropout, bidirectional=bidirectional_encoder, esn_leak=esn_leak)
				self.suffix_encoder = torch.nn.Sequential(collections.OrderedDict([
					('rnn', suffix_encoder),
					('mlp', model.MLP(suffix_encoder.hidden_size_total, mlp_hidden_size, context_feature_size))
				]))
				self.prefix_encoder.to(self.device)
				self.suffix_encoder.to(self.device)
				logger.info('Prefix and suffix to the target interval are (deterministically) embedded into {context_feature_size}-dim features.'.format(context_feature_size=context_feature_size))
				self.modules += [self.prefix_encoder, self.suffix_encoder]
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
			logger.info("Dropout rate in the non-top layers of the encoder RNN: {do}".format(do=encoder_hidden_dropout))
			logger.info("Self-feedback to the decoder: {decoder_self_feedback}".format(decoder_self_feedback=decoder_self_feedback))
			if decoder_self_feedback:
				logger.info("Dropout rate in the input to the decoder RNN: {do}".format(do=decoder_input_dropout))
			if encoder_rnn_type == 'ESN' or decoder_rnn_type == 'ESN':
				logger.info('ESN leak: {leak}'.format(leak=esn_leak))
			if not speaker_embed_dim is None:
				logger.info("Speaker ID # is embedded and fed to the decoder(s).")
				logger.info("# of speakers: {num_speakers}".format(num_speakers=num_speakers))
				logger.info("Embedding dimension: {speaker_embed_dim}".format(speaker_embed_dim=speaker_embed_dim))
			self.parameters = lambda:itertools.chain(*[m.parameters() for m in self.modules])





	def train(self, dataloader):
		"""
		Training phase. Updates weights.
		"""
		[m.train() for m in self.modules] # Turn on training mode which enables dropout.

		emission_loss = 0
		end_prediction_loss = 0
		kl_loss = 0

		num_batches = dataloader.get_num_batches()

		for batch_ix,(packed_input, is_offset, packed_prefix, packed_suffix, speaker, _) in enumerate(dataloader, 1):
			packed_input = packed_input.to(self.device)
			is_offset = is_offset.to(self.device)
			packed_prefix = packed_prefix.to(self.device)
			packed_suffix = packed_suffix.to(self.device)
			speaker = speaker.to(self.device)

			self.optimizer.zero_grad()

			if not self.frame_encoder is None:
				cnn_out = self.frame_encoder(packed_input.data)
				frame_feature_params = self.frame_feature_sampler(cnn_out)
				frame_features = self.frame_feature_sampler.sample(frame_feature_params)
				kl_loss_per_batch = self.frame_feature_sampler.kl_divergence(frame_feature_params)
				speaker_per_frame = torch.cat([speaker[:bs] for bs in packed_input.batch_sizes])
				frame_emission_loss,_ = self.frame_decoder(frame_features, speaker=speaker_per_frame, ground_truth_out=packed_input.data)
				packed_input = torch.nn.utils.rnn.PackedSequence(frame_features, packed_input.batch_sizes) # Unrecomendded instantiation
				# packed_input = torch.nn.utils.rnn.pack_sequence([[frame_features[packed_input.batch_sizes[:t].sum()+b] for t,bs in enumerate(packed_input.batch_sizes) if bs>b] for b in range(packed_input.batch_sizes[0])]) # Alternative way of building the packed sequence.
			else:
				kl_loss_per_batch = 0.0
				frame_emission_loss = 0.0

			last_hidden = self.encoder(packed_input)
			feature_params = self.feature_sampler(last_hidden)
			features = self.feature_sampler.sample(feature_params)
			if not self.prefix_encoder is None:
				if not self.prefix_frame_encoder is None:
					# Embed prefix frames
					cnn_out = self.prefix_frame_encoder(packed_prefix.data)
					frame_feature_params = self.prefix_frame_feature_sampler(cnn_out)
					frame_features = self.prefix_frame_feature_sampler.sample(frame_feature_params)
					kl_loss_per_batch += self.prefix_frame_feature_sampler.kl_divergence(frame_feature_params)
					speaker_per_frame = torch.cat([speaker[:bs] for bs in packed_prefix.batch_sizes])
					frame_emission_loss_,_ = self.prefix_frame_decoder(frame_features, speaker=speaker_per_frame, ground_truth_out=packed_prefix.data)
					frame_emission_loss += frame_emission_loss_
					packed_prefix = torch.nn.utils.rnn.PackedSequence(frame_features, packed_prefix.batch_sizes) # Unrecomendded instantiation

					# Embed suffix frames
					cnn_out = self.suffix_frame_encoder(packed_suffix.data)
					frame_feature_params = self.suffix_frame_feature_sampler(cnn_out)
					frame_features = self.suffix_frame_feature_sampler.sample(frame_feature_params)
					kl_loss_per_batch += self.suffix_frame_feature_sampler.kl_divergence(frame_feature_params)
					speaker_per_frame = torch.cat([speaker[:bs] for bs in packed_suffix.batch_sizes])
					frame_emission_loss_,_ = self.suffix_frame_decoder(frame_features, speaker=speaker_per_frame, ground_truth_out=packed_suffix.data)
					frame_emission_loss += frame_emission_loss_
					packed_suffix = torch.nn.utils.rnn.PackedSequence(frame_features, packed_suffix.batch_sizes) # Unrecomendded instantiation
				prefix_features = self.prefix_encoder(packed_prefix)
				suffix_features = self.suffix_encoder(packed_suffix)
				features = torch.cat([features, prefix_features, suffix_features], dim=-1) # This concat order is arbitrary.
			kl_loss_per_batch += self.feature_sampler.kl_divergence(feature_params)
			emission_loss_per_batch, end_prediction_loss_per_batch, _, _, _ = self.decoder(features, batch_sizes=packed_input.batch_sizes, speaker=speaker, ground_truth_out=packed_input.data, ground_truth_offset=is_offset.data)
			emission_loss_per_batch += frame_emission_loss

			loss = emission_loss_per_batch + end_prediction_loss_per_batch + kl_loss_per_batch
			loss /= packed_input.batch_sizes[0]
			loss.backward()

			# `clip_grad_norm` helps prevent the exploding gradient problem in RNNs.
			torch.nn.utils.clip_grad_norm_(self.parameters(), self.gradient_clip)

			self.optimizer.step()

			emission_loss += emission_loss_per_batch.item()
			end_prediction_loss += end_prediction_loss_per_batch.item()
			kl_loss += kl_loss_per_batch.item()

			logger.info('{batch_ix}/{num_batches} training batches complete. mean loss: {loss:5.4f}'.format(batch_ix=batch_ix, num_batches=num_batches, loss=loss.item()))

		num_strings = len(dataloader.dataset)
		emission_loss /= num_strings
		end_prediction_loss /= num_strings
		kl_loss /= num_strings
		mean_loss = emission_loss + end_prediction_loss + kl_loss
		logger.info('mean training emission negative pdf loss (per string): {:5.4f}'.format(emission_loss))
		logger.info('mean training end-prediction loss (per string): {:5.4f}'.format(end_prediction_loss))
		logger.info('mean training KL (per string): {:5.4f}'.format(kl_loss))
		logger.info('mean training total loss (per string): {:5.4f}'.format(mean_loss))


	def test_or_validate(self, dataloader):
		"""
		Test/validation phase. No update of weights.
		"""
		[m.eval() for m in self.modules] # Turn on evaluation mode which disables dropout.

		emission_loss = 0
		end_prediction_loss = 0
		kl_loss = 0
		sign_loss = 0

		num_batches = dataloader.get_num_batches()

		with torch.no_grad():
			for batch_ix, (packed_input, is_offset, packed_prefix, packed_suffix, speaker, _) in enumerate(dataloader, 1):
				packed_input = packed_input.to(self.device)
				is_offset = is_offset.to(self.device)
				packed_prefix = packed_prefix.to(self.device)
				packed_suffix = packed_suffix.to(self.device)
				speaker = speaker.to(self.device)

				if not self.frame_encoder is None:
					cnn_out = self.frame_encoder(packed_input.data)
					frame_feature_params = self.frame_feature_sampler(cnn_out)
					frame_features = self.frame_feature_sampler.sample(frame_feature_params)
					kl_loss += self.frame_feature_sampler.kl_divergence(frame_feature_params).item()
					speaker_per_frame = torch.cat([speaker[:bs] for bs in packed_input.batch_sizes])
					frame_emission_loss,_ = self.frame_decoder(frame_features, speaker=speaker_per_frame, ground_truth_out=packed_input.data)
					emission_loss += frame_emission_loss.item()
					packed_input = torch.nn.utils.rnn.PackedSequence(frame_features, packed_input.batch_sizes) # Unrecomendded instantiation
					# packed_input = torch.nn.utils.rnn.pack_sequence([[frame_features[packed_input.batch_sizes[:t].sum()+b] for t,bs in enumerate(packed_input.batch_sizes) if bs>b] for b in range(packed_input.batch_sizes[0])]) # Alternative way of building the packed sequence.

				last_hidden = self.encoder(packed_input)
				feature_params = self.feature_sampler(last_hidden)
				features = self.feature_sampler.sample(feature_params)
				if not self.prefix_encoder is None:
					if not self.prefix_frame_encoder is None:
						# Embed prefix frames
						cnn_out = self.prefix_frame_encoder(packed_prefix.data)
						frame_feature_params = self.prefix_frame_feature_sampler(cnn_out)
						frame_features = self.prefix_frame_feature_sampler.sample(frame_feature_params)
						kl_loss += self.prefix_frame_feature_sampler.kl_divergence(frame_feature_params).item()
						speaker_per_frame = torch.cat([speaker[:bs] for bs in packed_prefix.batch_sizes])
						frame_emission_loss_,_ = self.prefix_frame_decoder(frame_features, speaker=speaker_per_frame, ground_truth_out=packed_prefix.data)
						emission_loss += frame_emission_loss_.item()
						packed_prefix = torch.nn.utils.rnn.PackedSequence(frame_features, packed_prefix.batch_sizes) # Unrecomendded instantiation

						# Embed suffix frames
						cnn_out = self.suffix_frame_encoder(packed_suffix.data)
						frame_feature_params = self.suffix_frame_feature_sampler(cnn_out)
						frame_features = self.suffix_frame_feature_sampler.sample(frame_feature_params)
						kl_loss += self.suffix_frame_feature_sampler.kl_divergence(frame_feature_params).item()
						speaker_per_frame = torch.cat([speaker[:bs] for bs in packed_suffix.batch_sizes])
						frame_emission_loss_,_ = self.suffix_frame_decoder(frame_features, speaker=speaker_per_frame, ground_truth_out=packed_suffix.data)
						emission_loss += frame_emission_loss_.item()
						packed_suffix = torch.nn.utils.rnn.PackedSequence(frame_features, packed_suffix.batch_sizes) # Unrecomendded instantiation
					prefix_features = self.prefix_encoder(packed_prefix)
					suffix_features = self.suffix_encoder(packed_suffix)
					features = torch.cat([features, prefix_features, suffix_features], dim=-1) # This concat order is arbitrary.
				kl_loss += self.feature_sampler.kl_divergence(feature_params).item()
				emission_loss_per_batch, end_prediction_loss_per_batch, _, _, _ = self.decoder(features, batch_sizes=packed_input.batch_sizes, speaker=speaker, ground_truth_out=packed_input.data, ground_truth_offset=is_offset.data)
				emission_loss += emission_loss_per_batch.item()
				end_prediction_loss += end_prediction_loss_per_batch.item()

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
			mean_valid_loss = self.test_or_validate(valid_dataloader)
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
			'encoder_init_args':self.encoder.pack_init_args(),
			'feature_sampler':self.feature_sampler.state_dict(),
			'feature_sampler_init_args':self.feature_sampler.pack_init_args(),
			'decoder':self.decoder.state_dict(),
			'decoder_init_args':self.decoder.pack_init_args(),
			'optimizer':self.optimizer.state_dict(),
			'lr_scheduler':self.lr_scheduler.state_dict(),
			'gradient_clip':self.gradient_clip,
			'random_state':torch.get_rng_state(),
		}
		if not self.prefix_encoder is None:
			checkpoint['prefix_encoder'] = self.prefix_encoder.state_dict()
			checkpoint['prefix_encoder_init_args'] = self.prefix_encoder.rnn.pack_init_args()
			checkpoint['suffix_encoder'] = self.suffix_encoder.state_dict()
			checkpoint['suffix_encoder_init_args'] = self.suffix_encoder.rnn.pack_init_args()
			if not self.prefix_frame_encoder is None:
				checkpoint['prefix_frame_encoder'] = self.prefix_frame_encoder.state_dict()
				checkpoint['prefix_frame_encoder_init_args'] = self.prefix_frame_encoder.pack_init_args()
				checkpoint['prefix_frame_feature_sampler'] = self.prefix_frame_feature_sampler.state_dict()
				checkpoint['prefix_frame_feature_sampler_init_args'] = self.prefix_frame_feature_sampler.pack_init_args()
				checkpoint['prefix_frame_decoder'] = self.prefix_frame_decoder.state_dict()
				checkpoint['prefix_frame_decoder_init_args'] = self.prefix_frame_decoder.pack_init_args()

				checkpoint['suffix_frame_encoder'] = self.suffix_frame_encoder.state_dict()
				checkpoint['suffix_frame_encoder_init_args'] = self.suffix_frame_encoder.pack_init_args()
				checkpoint['suffix_frame_feature_sampler'] = self.suffix_frame_feature_sampler.state_dict()
				checkpoint['suffix_frame_feature_sampler_init_args'] = self.suffix_frame_feature_sampler.pack_init_args()
				checkpoint['suffix_frame_decoder'] = self.suffix_frame_decoder.state_dict()
				checkpoint['suffix_frame_decoder_init_args'] = self.suffix_frame_decoder.pack_init_args()
		if not self.frame_encoder is None:
			checkpoint['frame_encoder'] = self.frame_encoder.state_dict()
			checkpoint['frame_encoder_init_args'] = self.frame_encoder.pack_init_args()
			checkpoint['frame_feature_sampler'] = self.frame_feature_sampler.state_dict()
			checkpoint['frame_feature_sampler_init_args'] = self.frame_feature_sampler.pack_init_args()
			checkpoint['frame_decoder'] = self.frame_decoder.state_dict()
			checkpoint['frame_decoder_init_args'] = self.frame_decoder.pack_init_args()
		if torch.cuda.is_available():
			checkpoint['random_state_cuda'] = torch.cuda.get_rng_state_all()
		torch.save(checkpoint, os.path.join(self.save_dir, 'checkpoint.pt'))
		logger.info('Config successfully saved.')


	def retrieve_model(self, checkpoint_path = None, device='cpu'):
		if checkpoint_path is None:
			checkpoint_path = os.path.join(self.save_dir, 'checkpoint.pt')
		checkpoint = torch.load(checkpoint_path, map_location='cpu') # Random state needs to be loaded to CPU first even when cuda is available.

		for key,value in checkpoint.items():
			if 'init_parameters' in key:
				new_key = key.replace('init_parameters', 'init_args')
				checkpoint[new_key] = value

		self.encoder = model.RNN_Variational_Encoder(**checkpoint['encoder_init_args'])
		self.feature_sampler = model.Sampler(**checkpoint['feature_sampler_init_args'])
		self.decoder = model.RNN_Variational_Decoder(**checkpoint['decoder_init_args'])
		self.encoder.load_state_dict(checkpoint['encoder'])
		self.feature_sampler.load_state_dict(checkpoint['feature_sampler'])
		self.decoder.load_state_dict(checkpoint['decoder'])
		self.encoder.to(self.device)
		self.feature_sampler.to(self.device)
		self.decoder.to(self.device)
		self.modules = [self.encoder, self.feature_sampler, self.decoder]

		self.frame_encoder = None
		self.frame_feature_sampler = None
		self.frame_decoder = None
		if 'frame_encoder' in checkpoint:
			self.frame_encoder = model.CNN_Variational_Encoder(**checkpoint['frame_encoder_init_args'])
			self.frame_feature_sampler = model.Sampler(**checkpoint['frame_feature_sampler_init_args'])
			self.frame_decoder = model.MLP_Variational_Decoder(**checkpoint['frame_decoder_init_args'])
			self.frame_encoder.load_state_dict(checkpoint['frame_encoder'])
			self.frame_feature_sampler.load_state_dict(checkpoint['frame_feature_sampler'])
			self.frame_decoder.load_state_dict(checkpoint['frame_decoder'])
			self.frame_encoder.to(self.device)
			self.frame_feature_sampler.to(self.device)
			self.frame_decoder.to(self.device)
			self.modules += [self.frame_encoder, self.frame_feature_sampler, self.frame_decoder]
			

		self.prefix_encoder = None
		self.suffix_encoder = None
		self.prefix_frame_encoder = None
		self.prefix_frame_feature_sampler = None
		self.prefix_frame_decoder = None
		self.suffix_frame_encoder = None
		self.suffix_frame_feature_sampler = None
		self.suffix_frame_decoder = None
		if 'prefix_encoder' in checkpoint:
			mlp_hidden_size = checkpoint['decoder_init_args']['mlp_hidden_size']
			context_feature_size = (checkpoint['decoder_init_args']['feature_size'] - checkpoint['feature_sampler_init_args']['output_size']) // 2
			prefix_encoder = model.RNN_Variational_Encoder(**checkpoint['prefix_encoder_init_args'])
			self.prefix_encoder = torch.nn.Sequential(collections.OrderedDict([
				('rnn', prefix_encoder),
				('mlp', model.MLP(prefix_encoder.hidden_size_total, mlp_hidden_size, context_feature_size))
			]))
			suffix_encoder = model.RNN_Variational_Encoder(**checkpoint['suffix_encoder_init_args'])
			self.suffix_encoder = torch.nn.Sequential(collections.OrderedDict([
				('rnn', suffix_encoder),
				('mlp', model.MLP(suffix_encoder.hidden_size_total, mlp_hidden_size, context_feature_size))
			]))
			self.prefix_encoder.load_state_dict(checkpoint['prefix_encoder'])
			self.suffix_encoder.load_state_dict(checkpoint['suffix_encoder'])
			self.prefix_encoder.to(self.device)
			self.suffix_encoder.to(self.device)
			self.modules += [self.prefix_encoder, self.suffix_encoder]
			if 'prefix_frame_encoder' in checkpoint:
				self.prefix_frame_encoder = model.CNN_Variational_Encoder(**checkpoint['prefix_frame_encoder_init_args'])
				self.prefix_frame_feature_sampler = model.Sampler(**checkpoint['prefix_frame_feature_sampler_init_args'])
				self.prefix_frame_decoder = model.MLP_Variational_Decoder(**checkpoint['prefix_frame_decoder_init_args'])
				self.prefix_frame_encoder.load_state_dict(checkpoint['prefix_frame_encoder'])
				self.prefix_frame_feature_sampler.load_state_dict(checkpoint['prefix_frame_feature_sampler'])
				self.prefix_frame_decoder.load_state_dict(checkpoint['prefix_frame_decoder'])
				self.prefix_frame_encoder.to(self.device)
				self.prefix_frame_feature_sampler.to(self.device)
				self.prefix_frame_decoder.to(self.device)
				self.modules += [self.prefix_frame_encoder, self.prefix_frame_feature_sampler, self.prefix_frame_decoder]

				self.suffix_frame_encoder = model.CNN_Variational_Encoder(**checkpoint['suffix_frame_encoder_init_args'])
				self.suffix_frame_feature_sampler = model.Sampler(**checkpoint['suffix_frame_feature_sampler_init_args'])
				self.suffix_frame_decoder = model.MLP_Variational_Decoder(**checkpoint['suffix_frame_decoder_init_args'])
				self.suffix_frame_encoder.load_state_dict(checkpoint['suffix_frame_encoder'])
				self.suffix_frame_feature_sampler.load_state_dict(checkpoint['suffix_frame_feature_sampler'])
				self.suffix_frame_decoder.load_state_dict(checkpoint['suffix_frame_decoder'])
				self.suffix_frame_encoder.to(self.device)
				self.suffix_frame_feature_sampler.to(self.device)
				self.suffix_frame_decoder.to(self.device)
				self.modules += [self.suffix_frame_encoder, self.suffix_frame_feature_sampler, self.suffix_frame_decoder]

		self.parameters = lambda:itertools.chain(*[m.parameters() for m in self.modules])

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
	par_parser.add_argument('--context_feature_size', type=int, default=None, help='# of dimensions of features info which prefix and suffix are encoded. Equal to feature_size by default.')
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
	par_parser.add_argument('--fft_frame_length', type=float, default=0.008, help='FFT frame length in sec.')
	par_parser.add_argument('--fft_step_size', type=float, default=0.004, help='FFT step size in sec.')
	par_parser.add_argument('--fft_window_type', type=str, default='hann_window', help='Window type for FFT. "hann_window" by default.')
	par_parser.add_argument('--fft_no_centering', action='store_true', help='If selected, no centering in FFT.')
	par_parser.add_argument('--channel', type=int, default=0, help='Channel ID # (starting from 0) of multichannel recordings to use.')
	par_parser.add_argument('--mfcc', action='store_true', help='Use the MFCCs for the input.')
	par_parser.add_argument('--num_mfcc', type=int, default=20, help='# of MFCCs to use as the input.')
	par_parser.add_argument('--frame_embedding', action='store_true', help='If selected, embed each FFT frame into an arbitrary size by CNN.')
	par_parser.add_argument('--frame_feature_size', type=int, default=8, help='The size of the embedded FFT frame.')
	par_parser.add_argument('--frame_conv_stride', type=int, default=2, help='Stride of the frame convolution per layer.')
	par_parser.add_argument('--frame_conv_kernel_size', type=int, default=3, help='Kernel size of the frame convolution per layer.')
	par_parser.add_argument('--context_length', type=float, default=0.0, help='Length of the prefix and suffix sound wave in sec.')
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

	to_tensor = data_utils.ToTensor()
	if parameters.mfcc:
		from torchaudio.transforms import MFCC
		broadcast = data_utils.Transform(lambda x: x.view(1,-1)) # 1st dim for channel
		mfcc = MFCC(sample_rate=fs, n_mfcc=parameters.num_mfcc, log_mels=True, melkwargs={
				'n_fft':fft_frame_length,
				'win_length':fft_frame_length,
				'hop_length':fft_step_size,
				'window_fn':getattr(torch, parameters.fft_window_type)
				})
		squeeze_transpose_and_normalize = data_utils.Transform(lambda x: x.squeeze(dim=0).t() / parameters.data_normalizer)
		transform = Compose([to_tensor,broadcast,mfcc,squeeze_transpose_and_normalize])
		input_size = parameters.num_mfcc
	else:
		stft = data_utils.STFT(fft_frame_length, fft_step_size, window=parameters.fft_window_type, centering=not parameters.fft_no_centering)
		log_and_normalize = data_utils.Transform(lambda x: (x + parameters.epsilon).log() / parameters.data_normalizer)
		transform = Compose([to_tensor,stft,log_and_normalize])
		input_size = int(fft_frame_length / 2 + 1)

	if parameters.decoder_rnn_type is None:
		parameters.decoder_rnn_type = parameters.encoder_rnn_type

	if parameters.context_length <= 0.0:
		parameters.context_feature_size = 0

	# Get a model.
	learner = Learner(
				input_size,
				parameters.encoder_rnn_hidden_size,
				parameters.decoder_rnn_hidden_size,
				parameters.mlp_hidden_size,
				parameters.feature_size,
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
				num_speakers=num_speakers,
				speaker_embed_dim=parameters.speaker_embed_dim,
				context_feature_size=parameters.context_feature_size,
				frame_embedding=parameters.frame_embedding,
				frame_feature_size=parameters.frame_feature_size,
				frame_conv_stride=parameters.frame_conv_stride,
				frame_conv_kernel_size=parameters.frame_conv_kernel_size
				)

	logger.info("Sampling frequency of data: {fs}".format(fs=fs))
	logger.info("STFT window type: {fft_window}".format(fft_window=parameters.fft_window_type))
	logger.info("STFT frame lengths: {fft_frame_length_in_sec} sec".format(fft_frame_length_in_sec=parameters.fft_frame_length))
	logger.info("STFT step size: {fft_step_size_in_sec} sec".format(fft_step_size_in_sec=parameters.fft_step_size))
	if parameters.mfcc:
		logger.info("{num_mfcc}-dim MFCCs will be the input.".format(num_mfcc=parameters.num_mfcc))
	else:
		logger.info("log(abs(STFT(wav))) + {eps}) / {normalizer} will be the input.".format(eps=parameters.epsilon, normalizer=parameters.data_normalizer))
	if parameters.context_length>0:
		logger.info("{context_length} sec before and after the target region are fed to the decoder.".format(context_length=parameters.context_length))
	else:
		logger.info('Decoder does not receive context info around the target interval.')

	train_dataset = data_parser.get_data(data_type='train', transform=transform, channel=parameters.channel, context_length_in_sec=parameters.context_length)
	valid_dataset = data_parser.get_data(data_type='valid', transform=transform, channel=parameters.channel, context_length_in_sec=parameters.context_length)
	

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