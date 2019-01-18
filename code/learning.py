# coding: utf-8

import torch
from torchvision.transforms import Compose
import numpy as np
import model, data_utils
from logging import getLogger,FileHandler,DEBUG,Formatter
import os, argparse, random, itertools

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
	def __init__(self, lstm_input_size, lstm_hidden_size, save_dir, num_layers=1, dropout = 0.5, device=False, seed=1111):
		self.retrieval,self.log_file_path = update_log_handler(save_dir)
		if not self.retrieval:
			torch.manual_seed(seed)
			if torch.cuda.is_available():
				if device.startswith('cuda'):
					torch.cuda.manual_seed(seed)
				else:
					print('CUDA is available. Restart with option -C or --cuda to activate it.')

		self.mse_loss = torch.nn.MSELoss(reduction='sum')
		self.cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='sum')

		self.save_dir = save_dir

		self.device = torch.device(device)
		logger.info('Device: {device}'.format(device=device))

		

		self.encoder = model.LSTM(lstm_input_size, lstm_hidden_size, num_layers=num_layers, dropout=dropout)
		self.decoder = model.LSTM(lstm_input_size, lstm_hidden_size, num_layers=num_layers, dropout=dropout, is_decoder=True)

		if self.retrieval:
			self.last_epoch = self.retrieve_model(device)
			logger.info('Model retrieved.')
		else:
			logger.info('Random seed: {seed}'.format(seed = seed))
			logger.info("# of hidden layers: {hl}".format(hl=num_layers))
			logger.info("Dropout rate: {do}".format(do=dropout))

			# logger.info("Sampling frequency of data: {fs}".format(fs=fs))
			# logger.info("STFT window type: {fft_window}".format(fft_window=fft_window))
			# logger.info("STFT frame lengths: {fft_frame_length_in_sec} sec".format(fft_frame_length_in_sec=fft_frame_length_in_sec))
			# logger.info("STFT step size: {fft_step_size_in_sec} sec".format(fft_step_size_in_sec=fft_step_size_in_sec))

		self.encoder.to(self.device)
		self.decoder.to(self.device)




	def train(self, dataloader):
		"""
		Training phase. Updates weights.
		"""
		self.encoder.train() # Turn on training mode which enables dropout.
		self.decoder.train()

		total_loss = 0
		data_size = 0

		num_batches = dataloader.get_num_batches()

		for batch_ix,(batched_input, pseudo_input, is_offset) in enumerate(dataloader, 1):
			batched_input = batched_input.to(self.device)
			pseudo_input = pseudo_input.to(self.device)
			is_offset = is_offset.to(self.device)

			hidden = self.encoder.init_hidden(batched_input.batch_sizes[0])

			self.optimizer.zero_grad()

			hidden = self.encoder(batched_input, hidden)
			flatten_prediction,_,flatten_offset_prediction = self.decoder(pseudo_input, hidden)

			loss = self.mse_loss(flatten_prediction, batched_input.data)
			loss += self.cross_entropy_loss(flatten_offset_prediction, is_offset.data)
			loss.backward()

			# `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
			torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), self.gradient_clip)
			torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), self.gradient_clip)

			self.optimizer.step()

			total_loss += loss.item()
			data_size += batched_input.batch_sizes.sum().item()

			logger.info('{batch_ix}/{num_batches} training batches complete.'.format(batch_ix=batch_ix, num_batches=num_batches))

		mean_loss = total_loss / data_size
		logger.info('mean training loss: {:5.2f}'.format(mean_loss))
		# logger.info('training perplexity: {:8.2f}'.format(np.exp(mean_loss)))


	def test_or_validate(self, dataloader):
		"""
		Test/validation phase. No update of weights.
		"""
		self.encoder.eval() # Turn on evaluation mode which disables dropout.
		self.decoder.eval()

		total_loss = 0
		data_size = 0

		num_batches = dataloader.get_num_batches()

		with torch.no_grad():
			for batch_ix, (batched_input, pseudo_input, is_offset) in enumerate(dataloader, 1):
				batched_input = batched_input.to(self.device)
				pseudo_input = pseudo_input.to(self.device)
				is_offset = is_offset.to(self.device)

				hidden = self.encoder.init_hidden(batched_input.batch_sizes[0])

				hidden = self.encoder(pseudo_input, hidden)
				flatten_prediction,_,flatten_offset_prediction = self.decoder(pseudo_input, hidden)

				total_loss += torch.nn.MSELoss(reduction='sum')(
											flatten_prediction,
											batched_input.data
											).item()
				total_loss += torch.nn.CrossEntropyLoss(reduction='sum')(
												flatten_offset_prediction,
												is_offset.data
											).item()
				data_size += batched_input.batch_sizes.sum().item()

				logger.info('{batch_ix}/{num_batches} validation batches complete.'.format(batch_ix=batch_ix, num_batches=num_batches))

		mean_loss = total_loss / data_size # Average loss
		logger.info('mean validation loss: {:5.2f}'.format(mean_loss))
		# logger.info('validation perplexity: {:8.2f}'.format(np.exp(mean_loss)))
		return mean_loss




	def learn(self, train_dataset, valid_dataset, num_epochs, batch_size_train, batch_size_valid, learning_rate=0.1, momentum= 0.9, gradient_clip = 0.25, patience=0):
		train_dataloader = data_utils.DataLoader(train_dataset, batch_size=batch_size_train)
		valid_dataloader = data_utils.DataLoader(valid_dataset, batch_size=batch_size_valid)
		if self.retrieval:
			initial_epoch = self.last_epoch + 1
			logger.info('To be restarted from the beginning of epoch #: {epoch}'.format(epoch=initial_epoch))
		else:
			self.optimizer = torch.optim.SGD(itertools.chain(self.encoder.parameters(), self.decoder.parameters()), lr=learning_rate, momentum=momentum)
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
			'optimizer':self.optimizer.state_dict(),
			'lr_scheduler':self.lr_scheduler.state_dict(),
			'gradient_clip':self.gradient_clip
		}
		torch.save(checkpoint, os.path.join(self.save_dir, 'checkpoint.pt'))
		logger.info('Config successfully saved.')


	def retrieve_model(self, device, dir_path = None):
		if dir_path is None:
			dir_path = self.save_dir
		checkpoint = torch.load(os.path.join(dir_path, 'checkpoint.pt'))

		self.encoder.load_state_dict(checkpoint['encoder'])
		self.decoder.load_state_dict(checkpoint['decoder'])

		self.optimizer = torch.optim.SGD(itertools.chain(self.encoder.parameters(), self.decoder.parameters()), lr=0.1)
		self.optimizer.load_state_dict(checkpoint['optimizer'])

		self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer)
		self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

		self.gradient_clip = checkpoint['gradient_clip']
		return checkpoint['epoch']





def get_parameters():
	par_parser = argparse.ArgumentParser()

	par_parser.add_argument('input_root', type=str, help='Path to the root directory under which inputs are located.')
	par_parser.add_argument('annotation_file', type=str, help='Path to the annotation csv file.')
	par_parser.add_argument('-e', '--epochs', type=int, default=40, help='# of epochs to train the model.')
	par_parser.add_argument('-b', '--batch_size', type=int, default=10, help='Batch size for training.')
	par_parser.add_argument('-l', '--learning_rate', type=float, default=0.1, help='Initial learning rate.')
	par_parser.add_argument('-M', '--momentum', type=float, default=0.9, help='Momentum for the storchastic gradient descent.')
	par_parser.add_argument('-c', '--clip', type=float, default=0.25, help='Gradient clipping rate.')
	par_parser.add_argument('-D', '--dropout', type=float, default=0.0, help='Dropout rate.')
	par_parser.add_argument('--validation_batch_size', type=int, default=None, help='Batch size for validation. Same as for training b y default.')
	par_parser.add_argument('--layers', type=int, default=1, help='# of hidden layers.')
	par_parser.add_argument('--hidden_size', type=int, default=100, help='# of features in the hidden state of LSTM.')
	par_parser.add_argument('-j', '--job_id', type=str, default='NO_JOB_ID', help='Job ID. For users of computing clusters.')
	par_parser.add_argument('-s', '--seed', type=int, default=1111, help='random seed')
	par_parser.add_argument('-d', '--device', type=str, default='cpu', help='Computing device.')
	par_parser.add_argument('-S', '--save_root', type=str, default=None, help='Path to the directory where results are saved.')
	par_parser.add_argument('--fft_frame_length', type=float, default=0.025, help='FFT frame length in sec.')
	par_parser.add_argument('--fft_step_size', type=float, default=0.010, help='FFT step size in sec.')
	par_parser.add_argument('--fft_window_type', type=str, default='hann_window', help='Window type for FFT. "hann_window" by default.')
	par_parser.add_argument('--fft_no_centering', action='store_true', help='If selected, no centering in FFT.')
	par_parser.add_argument('-p', '--patience', type=int, default=0, help='# of epochs before updating the learning rate.')
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
				fft_frame_length + 2,
				parameters.hidden_size,
				save_dir,
				num_layers=parameters.layers,
				dropout=parameters.dropout,
				device = parameters.device,
				seed = parameters.seed
				)

	to_tensor = data_utils.ToTensor()
	stft = data_utils.STFT(fft_frame_length, fft_step_size, learner.device, window=parameters.fft_window_type, centering=not parameters.fft_no_centering)
	logger.info("Sampling frequency of data: {fs}".format(fs=fs))
	logger.info("STFT window type: {fft_window}".format(fft_window=parameters.fft_window_type))
	logger.info("STFT frame lengths: {fft_frame_length_in_sec} sec".format(fft_frame_length_in_sec=parameters.fft_frame_length))
	logger.info("STFT step size: {fft_step_size_in_sec} sec".format(fft_step_size_in_sec=parameters.fft_step_size))


	train_dataset = data_parser.get_data('train', transform=Compose([to_tensor,stft]))
	valid_dataset = data_parser.get_data('valid', transform=Compose([to_tensor,stft]))


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