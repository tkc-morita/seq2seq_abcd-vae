# coding: utf-8

import torch
from torchvision.transforms import Compose
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
	return retrieval,log_file_path



class Learner(object):
	def __init__(self, input_size, rnn_hidden_size, mlp_hidden_size, feature_size, save_dir, rnn_type='GRU', rnn_layers=1, bidirectional_encoder=True, dropout = 0.5, device=False, seed=1111, feature_distribution='isotropic_gaussian', emission_distribution='isotropic_gaussian', decoder_self_feedback=True):
		self.retrieval,self.log_file_path = update_log_handler(save_dir)
		if not self.retrieval:
			torch.manual_seed(seed)
			if torch.cuda.is_available():
				if device.startswith('cuda'):
					torch.cuda.manual_seed_all(seed)
				else:
					print('CUDA is available. Restart with option -C or --cuda to activate it.')

		self.bce_with_logits_loss = torch.nn.BCEWithLogitsLoss(reduction='sum')

		self.save_dir = save_dir

		self.device = torch.device(device)
		logger.info('Device: {device}'.format(device=device))

		

		if self.retrieval:
			self.last_epoch = self.retrieve_model(device=device)
			logger.info('Model retrieved.')
		else:
			self.feature_distribution = feature_distribution
			self.emission_distribution =  emission_distribution
			self.feature_sampler, _, self.kl_func = model.choose_distribution(feature_distribution)
			emission_sampler,self.log_pdf_emission,_ = model.choose_distribution(self.emission_distribution)
			self.encoder = model.RNN_Variational_Encoder(input_size, rnn_hidden_size, mlp_hidden_size, feature_size, rnn_type=rnn_type, rnn_layers=rnn_layers, dropout=dropout, bidirectional=bidirectional_encoder)
			self.decoder = model.RNN_Variational_Decoder(input_size, rnn_hidden_size, mlp_hidden_size, feature_size, rnn_type=rnn_type, rnn_layers=rnn_layers, emission_sampler=emission_sampler, self_feedback=decoder_self_feedback)
			self.bag_of_data_decoder = model.MLP_To_k_Vecs(feature_size, mlp_hidden_size, input_size, 2) # Analogous to Zhao et al.'s (2017) "bag-of-words MLP".
			logger.info('Data to be encoded into {feature_size}-dim features.'.format(feature_size=feature_size))
			logger.info('Features are assumed to be distributed according to {feature_distribution}.'.format(feature_distribution=feature_distribution))
			logger.info('Conditioned on the features, data are assumed to be distributed according to {emission_distribution}'.format(emission_distribution=emission_distribution))
			logger.info('Random seed: {seed}'.format(seed = seed))
			logger.info('Type of RNN used: {rnn_type}'.format(rnn_type=rnn_type))
			logger.info("# of RNN hidden layers: {hl}".format(hl=rnn_layers))
			logger.info("# of hidden units in the RNNs: {hs}".format(hs=rnn_hidden_size))
			logger.info("# of hidden units in the MLPs: {hs}".format(hs=mlp_hidden_size))
			logger.info("Encoder is bidirectional: {bidirectional_encoder}".format(bidirectional_encoder=bidirectional_encoder))
			logger.info("Dropout rate in the input to the encoder: {do}".format(do=dropout))
			logger.info("Self-feedback to the decoder: {decoder_self_feedback}".format(decoder_self_feedback=decoder_self_feedback))


		self.encoder.to(self.device)
		self.decoder.to(self.device)
		self.bag_of_data_decoder.to(self.device)
		self.parameters = lambda:itertools.chain(self.encoder.parameters(), self.decoder.parameters(), self.bag_of_data_decoder.parameters())




	def train(self, dataloader):
		"""
		Training phase. Updates weights.
		"""
		self.encoder.train() # Turn on training mode which enables dropout.
		self.decoder.train()
		self.bag_of_data_decoder.train()

		emission_loss = 0
		emission_loss_BOD = 0
		end_prediction_loss = 0
		kl_loss = 0

		num_batches = dataloader.get_num_batches()

		for batch_ix,(batched_input, is_offset, _) in enumerate(dataloader, 1):
			batched_input = batched_input.to(self.device)
			is_offset = is_offset.to(self.device)

			self.optimizer.zero_grad()

			feature_params = self.encoder(batched_input)
			features = self.feature_sampler(*feature_params)
			_, lengths = torch.nn.utils.rnn.pad_packed_sequence(batched_input, batch_first=True)
			emission_params,flatten_offset_prediction,_ = self.decoder(features, lengths, self.device)
			params_BOD = self.bag_of_data_decoder(features)
			params_BOD = [torch.nn.utils.rnn.pack_sequence([p[ix].expand(l,-1) for ix,l in enumerate(lengths)]).data
									for p in params_BOD] # Can/should be .expand() rather than .repeat() for aurograd. cf. https://discuss.pytorch.org/t/torch-repeat-and-torch-expand-which-to-use/27969

			emission_loss_per_batch = -self.log_pdf_emission(batched_input.data, *emission_params)
			emission_loss_BOD_per_batch = -self.log_pdf_emission(batched_input.data, *params_BOD)
			end_prediction_loss_per_batch = self.bce_with_logits_loss(flatten_offset_prediction, is_offset.data)
			kl_loss_per_batch = self.kl_func(*feature_params)
			loss = emission_loss_per_batch + end_prediction_loss_per_batch + kl_loss_per_batch + emission_loss_BOD_per_batch
			loss.backward()

			# `clip_grad_norm` helps prevent the exploding gradient problem in RNNs.
			torch.nn.utils.clip_grad_norm_(self.parameters(), self.gradient_clip)

			self.optimizer.step()

			emission_loss += emission_loss_per_batch.item()
			emission_loss_BOD += emission_loss_BOD_per_batch.item()
			end_prediction_loss += end_prediction_loss_per_batch.item()
			kl_loss += kl_loss_per_batch.item()

			logger.info('{batch_ix}/{num_batches} training batches complete.'.format(batch_ix=batch_ix, num_batches=num_batches))

		num_strings = len(dataloader.dataset)
		emission_loss /= num_strings
		emission_loss_BOD /= num_strings
		end_prediction_loss /= num_strings
		kl_loss /= num_strings
		mean_loss = emission_loss + emission_loss_BOD + end_prediction_loss + kl_loss
		logger.info('mean training emission negative pdf loss (ORDERED, per string): {:5.4f}'.format(emission_loss))
		logger.info('mean training emission negative pdf loss (BAG-OF-DATA, per string): {:5.4f}'.format(emission_loss_BOD))
		logger.info('mean training end-prediction loss (per string): {:5.4f}'.format(end_prediction_loss))
		logger.info('mean training KL (per string): {:5.4f}'.format(kl_loss))
		logger.info('mean training total loss (per string): {:5.4f}'.format(mean_loss))


	def test_or_validate(self, dataloader):
		"""
		Test/validation phase. No update of weights.
		"""
		self.encoder.eval() # Turn on evaluation mode which disables dropout.
		self.decoder.eval()
		self.bag_of_data_decoder.eval()

		emission_loss = 0
		emission_loss_BOD = 0
		end_prediction_loss = 0
		kl_loss = 0
		sign_loss = 0
		sign_loss_BOD = 0

		num_batches = dataloader.get_num_batches()

		with torch.no_grad():
			for batch_ix, (batched_input, is_offset, _) in enumerate(dataloader, 1):
				batched_input = batched_input.to(self.device)
				is_offset = is_offset.to(self.device)

				feature_params = self.encoder(batched_input)
				features = self.feature_sampler(*feature_params)
				_, lengths = torch.nn.utils.rnn.pad_packed_sequence(batched_input, batch_first=True)
				emission_params,flatten_offset_prediction,_ = self.decoder(features, lengths, self.device)
				params_BOD = self.bag_of_data_decoder(features)
				params_BOD = [torch.nn.utils.rnn.pack_sequence([p[ix].expand(l,-1) for ix,l in enumerate(lengths)]).data
										for p in params_BOD] # Should be .expand() rather than .repeat() for aurograd(?).

				emission_loss += -self.log_pdf_emission(batched_input.data, *emission_params).item()
				emission_loss_BOD += -self.log_pdf_emission(batched_input.data, *params_BOD).item()
				end_prediction_loss += torch.nn.BCEWithLogitsLoss(reduction='sum')(
												flatten_offset_prediction,
												is_offset.data
											).item()
				kl_loss += self.kl_func(*feature_params).item()

				logger.info('{batch_ix}/{num_batches} validation batches complete.'.format(batch_ix=batch_ix, num_batches=num_batches))

		num_strings = len(dataloader.dataset)
		emission_loss /= num_strings
		emission_loss_BOD /= num_strings
		end_prediction_loss /= num_strings
		kl_loss /= num_strings
		mean_loss = emission_loss + emission_loss_BOD + end_prediction_loss + kl_loss
		logger.info('mean validation emission negative pdf loss (ORDERED, per string): {:5.4f}'.format(emission_loss))
		logger.info('mean validation emission negative pdf loss (BAG-OF-DATA, per string): {:5.4f}'.format(emission_loss_BOD))
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
			'decoder':self.decoder.state_dict(),
			'bag_of_data_decoder':self.bag_of_data_decoder.state_dict(),
			'optimizer':self.optimizer.state_dict(),
			'lr_scheduler':self.lr_scheduler.state_dict(),
			'gradient_clip':self.gradient_clip,
			'input_size':int(self.encoder.rnn.rnn.input_size),
			'rnn_type':self.encoder.rnn.rnn.mode,
			'rnn_hidden_size':self.encoder.rnn.rnn.hidden_size,
			'rnn_layers':self.encoder.rnn.rnn.num_layers,
			'bidirectional_encoder':self.encoder.rnn.rnn.bidirectional,
			'mlp_hidden_size':self.encoder.to_parameters.mlps[0].hidden_size,
			'feature_size':self.decoder.feature2hidden.in_features,
			'feature_distribution':self.feature_distribution,
			'emission_distribution':self.emission_distribution,
			'model_random_state':torch.get_rng_state(),
		}
		if torch.cuda.is_available():
			checkpoint['model_random_state_cuda'] = torch.cuda.get_rng_state_all()
		torch.save(checkpoint, os.path.join(self.save_dir, 'checkpoint.pt'))
		logger.info('Config successfully saved.')


	def retrieve_model(self, checkpoint_path = None, device='cpu'):
		if checkpoint_path is None:
			checkpoint_path = os.path.join(self.save_dir, 'checkpoint.pt')
		checkpoint = torch.load(checkpoint_path, map_location=device)

		input_size = checkpoint['input_size']
		rnn_type = checkpoint['rnn_type']
		rnn_hidden_size = checkpoint['rnn_hidden_size']
		rnn_layers = checkpoint['rnn_layers']
		bidirectional_encoder = checkpoint['bidirectional_encoder']
		feature_size = checkpoint['feature_size']
		mlp_hidden_size = checkpoint['mlp_hidden_size']

		self.feature_distribution = checkpoint['feature_distribution']
		self.emission_distribution = checkpoint['emission_distribution']
		self.feature_sampler,_,self.kl_func = model.choose_distribution(self.feature_distribution)
		emission_sampler,self.log_pdf_emission,_ = model.choose_distribution(self.emission_distribution)

		self.encoder = model.RNN_Variational_Encoder(input_size, rnn_hidden_size, mlp_hidden_size, feature_size, rnn_type=rnn_type, rnn_layers=rnn_layers, bidirectional=bidirectional_encoder)
		self.decoder = model.RNN_Variational_Decoder(input_size, rnn_hidden_size, mlp_hidden_size, feature_size, rnn_type=rnn_type, rnn_layers=rnn_layers, emission_sampler=emission_sampler)
		self.bag_of_data_decoder = model.MLP_To_k_Vecs(feature_size, mlp_hidden_size, input_size, 2)
		self.encoder.load_state_dict(checkpoint['encoder'])
		self.decoder.load_state_dict(checkpoint['decoder'])
		self.bag_of_data_decoder.load_state_dict(checkpoint['bag_of_data_decoder'])


		self.parameters = lambda:itertools.chain(self.encoder.parameters(), self.decoder.parameters(), self.bag_of_data_decoder.parameters())
		self.optimizer = torch.optim.SGD(self.parameters(), lr=0.1)
		self.optimizer.load_state_dict(checkpoint['optimizer'])

		self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer)
		self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

		self.gradient_clip = checkpoint['gradient_clip']
		

		torch.set_rng_state(checkpoint['model_random_state'])
		if device=='cuda':
			torch.cuda.set_rng_state_all(checkpoint['model_random_state_cuda'])
		return checkpoint['epoch']





def get_parameters():
	par_parser = argparse.ArgumentParser()

	par_parser.add_argument('input_root', type=str, help='Path to the root directory under which inputs are located.')
	par_parser.add_argument('annotation_file', type=str, help='Path to the annotation csv file.')
	par_parser.add_argument('-e', '--epochs', type=int, default=40, help='# of epochs to train the model.')
	par_parser.add_argument('-b', '--batch_size', type=int, default=10, help='Batch size for training.')
	par_parser.add_argument('-l', '--learning_rate', type=float, default=1.0, help='Initial learning rate.')
	par_parser.add_argument('-f', '--feature_size', type=int, default=13, help='# of dimensions of features into which data are encoded.')
	par_parser.add_argument('-M', '--momentum', type=float, default=0.0, help='Momentum for the storchastic gradient descent.')
	par_parser.add_argument('-c', '--clip', type=float, default=1.0, help='Gradient clipping.')
	par_parser.add_argument('-D', '--dropout', type=float, default=0.0, help='Dropout rate.')
	par_parser.add_argument('--validation_batch_size', type=int, default=None, help='Batch size for validation. Same as for training b y default.')
	par_parser.add_argument('-R', '--rnn_type', type=str, default='GRU', help='Name of RNN to be used.')
	par_parser.add_argument('--rnn_layers', type=int, default=1, help='# of hidden layers.')
	par_parser.add_argument('--rnn_hidden_size', type=int, default=100, help='# of the RNN units.')
	par_parser.add_argument('--mlp_hidden_size', type=int, default=200, help='# of neurons in the hidden layer of the MLP transforms.')
	par_parser.add_argument('--greedy_decoder', action='store_true', help='If selected, decoder becomes greedy and will not receive self-feedback.')
	par_parser.add_argument('-j', '--job_id', type=str, default='NO_JOB_ID', help='Job ID. For users of computing clusters.')
	par_parser.add_argument('-s', '--seed', type=int, default=1111, help='random seed')
	par_parser.add_argument('-d', '--device', type=str, default='cpu', help='Computing device.')
	par_parser.add_argument('-S', '--save_root', type=str, default=None, help='Path to the directory where results are saved.')
	par_parser.add_argument('--fft_frame_length', type=float, default=0.008, help='FFT frame length in sec.')
	par_parser.add_argument('--fft_step_size', type=float, default=0.004, help='FFT step size in sec.')
	par_parser.add_argument('--fft_window_type', type=str, default='hann_window', help='Window type for FFT. "hann_window" by default.')
	par_parser.add_argument('--fft_no_centering', action='store_true', help='If selected, no centering in FFT.')
	par_parser.add_argument('-p', '--patience', type=int, default=0, help='# of epochs before updating the learning rate.')
	par_parser.add_argument('-N','--data_normalizer', type=float, default=1.0, help='Normalizing constant to devide the data.')
	par_parser.add_argument('-E','--epsilon', type=float, default=2**(-15), help='Small positive real number to add to avoid log(0).')
	# par_parser.add_argument('--retrieve', type=str, help='Path to a directory with previous training results. Retrieve previous training.')

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

	data_parser = data_utils.Data_Parser(parameters.input_root, parameters.annotation_file)
	fs = data_parser.get_sample_freq() # Assuming all the wav files have the same fs, get the 1st file's.

	fft_frame_length = int(np.floor(parameters.fft_frame_length * fs))
	fft_step_size = int(np.floor(parameters.fft_step_size * fs))

	# Get a model.
	learner = Learner(
				int(fft_frame_length / 2 + 1),
				parameters.rnn_hidden_size,
				parameters.mlp_hidden_size,
				parameters.feature_size,
				save_dir,
				rnn_type=parameters.rnn_type,
				rnn_layers=parameters.rnn_layers,
				dropout=parameters.dropout,
				device = parameters.device,
				seed = parameters.seed,
				decoder_self_feedback=not parameters.greedy_decoder
				)

	to_tensor = data_utils.ToTensor()
	stft = data_utils.STFT(fft_frame_length, fft_step_size, window=parameters.fft_window_type, centering=not parameters.fft_no_centering)
	log_and_normalize = data_utils.Transform(lambda x: (x + parameters.epsilon).log() / parameters.data_normalizer)
	logger.info("log(abs(STFT(wav))) + {eps}) / {normalizer} will be the input.".format(eps=parameters.epsilon, normalizer=parameters.data_normalizer))
	logger.info("Sampling frequency of data: {fs}".format(fs=fs))
	logger.info("STFT window type: {fft_window}".format(fft_window=parameters.fft_window_type))
	logger.info("STFT frame lengths: {fft_frame_length_in_sec} sec".format(fft_frame_length_in_sec=parameters.fft_frame_length))
	logger.info("STFT step size: {fft_step_size_in_sec} sec".format(fft_step_size_in_sec=parameters.fft_step_size))


	train_dataset = data_parser.get_data(data_type='train', transform=Compose([to_tensor,stft,log_and_normalize]))
	valid_dataset = data_parser.get_data(data_type='valid', transform=Compose([to_tensor,stft,log_and_normalize]))
	

	if parameters.validation_batch_size is None:
		parameters.batch_size = parameters.batch_size

	# Train the model.
	learner.learn(
			train_dataset,
			valid_dataset,
			parameters.epochs,
			parameters.batch_size,
			parameters.batch_size,
			learning_rate=parameters.learning_rate,
			momentum=parameters.momentum,
			gradient_clip = parameters.clip,
			patience = parameters.patience
			)