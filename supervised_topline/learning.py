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
			mlp_hidden_size,
			num_categories,
			save_dir,
			encoder_rnn_hidden_size=128,
			encoder_rnn_type='LSTM',
			encoder_rnn_layers=1,
			bidirectional_encoder=True,
			encoder_hidden_dropout = 0.0,
			device='cpu',
			seed=1111,
			esn_leak=1.0,
			use_input_mean = False,
			use_input_median = False,
			use_resampling = False,
			num_resampled_frames = 10,
			use_attention = False,
			attention_hidden_size = 512,
			num_attention_heads = 8,
			num_attention_layers = 1,
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

		self.cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='sum')
		self.cross_entropy_loss.to(self.device)

		if self.retrieval:
			self.last_epoch = self.retrieve_model(device=device)
			logger.info('Model retrieved.')
		else:
			torch.manual_seed(seed)
			torch.cuda.manual_seed_all(seed) # According to the docs, "It’s safe to call this function if CUDA is not available; in that case, it is silently ignored."
			logger.info('Random seed: {seed}'.format(seed = seed))
			self.use_input_median = False
			if use_resampling:
				self.encoder = model.Resample(num_resampled_frames)
				classifier_input_size = num_resampled_frames * input_size + 1
				logger.info('Resample the input time series into {} frames.'.format(num_resampled_frames))
				logger.info('Original sequence length is appended to the input.')
			elif use_input_mean:
				self.encoder = model.TakeMean()
				classifier_input_size = input_size + 1
				logger.info('Take the mean of the input over the time dimension.')
				logger.info('Original sequence length is appended to the input.')
			elif use_input_median:
				self.encoder = model.TakeMedian()
				classifier_input_size = input_size + 1
				self.use_input_median = True
				logger.info('Take the median of the input over the time dimension.')
				logger.info('Original sequence length is appended to the input.')
			elif use_attention:
				self.encoder = model.AttentionEncoderToFixedLength(input_size, attention_hidden_size, mlp_hidden_size, num_heads=num_attention_heads, num_layers=num_attention_layers, dropout=encoder_hidden_dropout)
				logger.info('Use attention (alone).')
				logger.info('# of attention layers: {}'.format(num_attention_layers))
				logger.info('# of attention hidden units per layer: {}'.format(attention_hidden_size))
				logger.info('# of attention heads: {}'.format(num_attention_heads))
				logger.info('Dropout rate at the top of the sublayers: {}'.format(encoder_hidden_dropout))
				classifier_input_size = attention_hidden_size
			else:
				logger.info('Type of RNN used for the encoder: {rnn_type}'.format(rnn_type=encoder_rnn_type))
				logger.info("# of RNN hidden layers in the encoder RNN: {hl}".format(hl=encoder_rnn_layers))
				logger.info("# of hidden units in the encoder RNNs: {hs}".format(hs=encoder_rnn_hidden_size))
				logger.info("Encoder is bidirectional: {bidirectional_encoder}".format(bidirectional_encoder=bidirectional_encoder))
				logger.info("Dropout rate in the non-top layers of the encoder RNN: {do}".format(do=encoder_hidden_dropout))
				if encoder_rnn_type == 'ESN':
					logger.info('ESN leak: {leak}'.format(leak=esn_leak))
				if encoder_hidden_dropout > 0.0 and encoder_rnn_layers==1:
					logger.warning('Non-zero dropout cannot be used for the single-layer encoder RNN (because there is no non-top hidden layers).')
					logger.info('encoder_hidden_dropout reset from {do} to 0.0.'.format(do=encoder_hidden_dropout))
					encoder_hidden_dropout = 0.0
				self.encoder = model.RNN_Variational_Encoder(input_size, encoder_rnn_hidden_size, rnn_type=encoder_rnn_type, rnn_layers=encoder_rnn_layers, hidden_dropout=encoder_hidden_dropout, bidirectional=bidirectional_encoder, esn_leak=esn_leak)
				classifier_input_size = self.encoder.hidden_size_total
			self.encoder.to(self.device)
			self.classifier = model.MLP(classifier_input_size, mlp_hidden_size, num_categories)
			self.classifier.to(self.device)
			logger.info("# of hidden units in the MLPs: {hs}".format(hs=mlp_hidden_size))
			self.modules = [self.encoder, self.classifier]
			self.parameters = lambda:itertools.chain(*[m.parameters() for m in self.modules])





	def train(self, dataloader):
		"""
		Training phase. Updates weights.
		"""
		[m.train() for m in self.modules] # Turn on training mode which enables dropout.
		self.cross_entropy_loss.train()

		total_loss = 0
		accuracy = 0

		num_batches = dataloader.get_num_batches()

		for batch_ix,(packed_input, batched_target, _) in enumerate(dataloader, 1):
			packed_input = packed_input.to(self.device)
			batched_target = batched_target.to(self.device)

			self.optimizer.zero_grad()

			last_hidden,_ = self.encoder(packed_input)
			weights = self.classifier(last_hidden)

			loss = self.cross_entropy_loss(weights, batched_target)
			total_loss += loss.item()
			loss /= packed_input.batch_sizes[0]
			loss.backward()

			argmax_category = weights.argmax(dim=-1)
			accuracy += (argmax_category==batched_target).sum().item()

			# `clip_grad_norm` helps prevent the exploding gradient problem in RNNs.
			torch.nn.utils.clip_grad_norm_(self.parameters(), self.gradient_clip)

			self.optimizer.step()

			logger.info('{batch_ix}/{num_batches} training batches complete. mean loss: {loss:5.4f}'.format(batch_ix=batch_ix, num_batches=num_batches, loss=loss.item()))

		num_strings = len(dataloader.dataset)
		mean_loss = total_loss / num_strings
		accuracy = accuracy / num_strings
		logger.info('mean training total loss: {:5.4f}'.format(mean_loss))
		logger.info('mean training accuracy: {:.4f}'.format(accuracy))


	def test_or_validate(self, dataloader):
		"""
		Test/validation phase. No update of weights.
		"""
		[m.eval() for m in self.modules] # Turn on evaluation mode which disables dropout.
		self.cross_entropy_loss.eval()

		total_loss = 0
		accuracy = 0

		num_batches = dataloader.get_num_batches()

		with torch.no_grad():
			for batch_ix, (packed_input, batched_target, _) in enumerate(dataloader, 1):
				packed_input = packed_input.to(self.device)
				batched_target = batched_target.to(self.device)

				last_hidden,_ = self.encoder(packed_input)
				weights = self.classifier(last_hidden)

				total_loss += self.cross_entropy_loss(weights, batched_target).item()

				argmax_category = weights.argmax(dim=-1)
				accuracy += (argmax_category==batched_target).sum().item()

				logger.info('{batch_ix}/{num_batches} validation batches complete.'.format(batch_ix=batch_ix, num_batches=num_batches))

		num_strings = len(dataloader.dataset)
		mean_loss = total_loss / num_strings
		accuracy = accuracy / num_strings
		logger.info('mean validation total loss: {:5.4f}'.format(mean_loss))
		logger.info('mean validation accuracy: {:.4f}'.format(accuracy))
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
			'classifier':self.classifier.state_dict(),
			'classifier_init_args':self.classifier.pack_init_args(),
			'optimizer':self.optimizer.state_dict(),
			'lr_scheduler':self.lr_scheduler.state_dict(),
			'gradient_clip':self.gradient_clip,
			'random_state':torch.get_rng_state(),
			'use_input_median':self.use_input_median
		}
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

		self.use_input_median = False
		if 'rnn_type' in checkpoint['encoder_init_args']:
			self.encoder = model.RNN_Variational_Encoder(**checkpoint['encoder_init_args'])
		elif 'num_samples' in checkpoint['encoder_init_args']:
			self.encoder = model.Resample(**checkpoint['encoder_init_args'])
		elif 'num_heads' in checkpoint['encoder_init_args']:
			self.encoder = model.AttentionEncoderToFixedLength(**checkpoint['encoder_init_args'])
		elif 'use_input_median' in checkpoint and checkpoint['use_input_median']:
			self.encoder = model.TakeMedian()
			self.use_input_median = True
		else:
			self.encoder = model.TakeMean()
		self.encoder.load_state_dict(checkpoint['encoder'])
		self.encoder.to(self.device)
		self.classifier = model.MLP(**checkpoint['classifier_init_args'])
		self.classifier.load_state_dict(checkpoint['classifier'])
		self.classifier.to(self.device)
		self.modules = [self.encoder, self.classifier]

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
	par_parser.add_argument('--encoder_rnn_layers', type=int, default=1, help='# of hidden layers in the encoder RNN.')
	par_parser.add_argument('--encoder_rnn_hidden_size', type=int, default=100, help='# of the RNN units in the encoder RNN.')
	par_parser.add_argument('--mlp_hidden_size', type=int, default=200, help='# of neurons in the hidden layer of the MLP transforms.')
	par_parser.add_argument('--encoder_hidden_dropout', type=float, default=0.0, help='Dropout rate in the non-top layers of the encoder RNN.')
	par_parser.add_argument('--esn_leak', type=float, default=1.0, help='Leak for the echo-state network. Ignored if the RNN type is not ESN.')
	par_parser.add_argument('--fft_frame_length', type=float, default=0.008, help='FFT frame length in sec.')
	par_parser.add_argument('--fft_step_size', type=float, default=0.004, help='FFT step size in sec.')
	par_parser.add_argument('--fft_window_type', type=str, default='hann_window', help='Window type for FFT. "hann_window" by default.')
	par_parser.add_argument('--fft_no_centering', action='store_true', help='If selected, no centering in FFT.')
	par_parser.add_argument('--channel', type=int, default=0, help='Channel ID # (starting from 0) of multichannel recordings to use.')
	par_parser.add_argument('--mfcc', action='store_true', help='Use the MFCCs for the input.')
	par_parser.add_argument('--num_mfcc', type=int, default=20, help='# of MFCCs to use as the input.')
	par_parser.add_argument('--formant', action='store_true', help='Use formants as the input.')
	par_parser.add_argument('--num_formants', type=int, default=2, help='# of formants used.')
	par_parser.add_argument('--use_pitch', action='store_true', help='If selected, use F0 ("Pitch" in Praat) in addition to higher formants.')
	par_parser.add_argument('-N','--data_normalizer', type=float, default=1.0, help='Normalizing constant to devide the data.')
	par_parser.add_argument('-E','--epsilon', type=float, default=2**(-15), help='Small positive real number to add to avoid log(0).')
	par_parser.add_argument('--use_input_mean', action='store_true', help='Pass the mean of the input frames over the time dimension to the MLP classifier.')
	par_parser.add_argument('--use_input_median', action='store_true', help='Pass the median of the input frames over the time dimension to the MLP classifier.')
	par_parser.add_argument('--use_resampling', action='store_true', help='Resample the input frames and pass them to the MLP classifier.')
	par_parser.add_argument('--num_resampled_frames', type=int, default=10, help='# of resampled frames to pass to the MLP decoder. Used only if --use_resampling is selected.')
	par_parser.add_argument('--use_attention', action='store_true', help='If selected, use attention instead of RNN.')
	par_parser.add_argument('--attention_hidden_size', type=int, default=512, help='Dimensionality of the hidden space of the attention.')
	par_parser.add_argument('--num_attention_heads', type=int, default=8, help='# of attention heads.')
	par_parser.add_argument('--num_attention_layers', type=int, default=1, help='# of layers of attention.')
	
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
	elif parameters.formant:
		get_formants = data_utils.Formant(fs, parameters.fft_frame_length, parameters.fft_step_size, num_formants=parameters.num_formants, use_pitch=parameters.use_pitch)
		half_nyquist_freq = fs / 4
		normalize = data_utils.Transform(lambda x: (x / half_nyquist_freq) - 1.0)
		transform = Compose([get_formants, to_tensor, normalize])
		input_size = parameters.num_formants
		if parameters.use_pitch:
			input_size += 1
	else:
		stft = data_utils.STFT(fft_frame_length, fft_step_size, window=parameters.fft_window_type, centering=not parameters.fft_no_centering)
		log_and_normalize = data_utils.Transform(lambda x: (x + parameters.epsilon).log() / parameters.data_normalizer)
		transform = Compose([to_tensor,stft,log_and_normalize])
		input_size = int(fft_frame_length / 2 + 1)

	# Get a model.
	learner = Learner(
				input_size,
				parameters.mlp_hidden_size,
				data_parser.get_num_labels(),
				save_dir,
				encoder_rnn_hidden_size=parameters.encoder_rnn_hidden_size,
				encoder_rnn_type=parameters.encoder_rnn_type,
				encoder_rnn_layers=parameters.encoder_rnn_layers,
				encoder_hidden_dropout=parameters.encoder_hidden_dropout,
				device = parameters.device,
				seed = parameters.seed,
				use_input_mean = parameters.use_input_mean,
				use_input_median = parameters.use_input_median,
				use_resampling = parameters.use_resampling,
				num_resampled_frames = parameters.num_resampled_frames,
				use_attention = parameters.use_attention,
				attention_hidden_size = parameters.attention_hidden_size,
				num_attention_heads = parameters.num_attention_heads,
				num_attention_layers = parameters.num_attention_layers
				)

	logger.info("Sampling frequency of data: {fs}".format(fs=fs))
	logger.info("STFT window type: {fft_window}".format(fft_window=parameters.fft_window_type))
	logger.info("STFT frame lengths: {fft_frame_length_in_sec} sec".format(fft_frame_length_in_sec=parameters.fft_frame_length))
	logger.info("STFT step size: {fft_step_size_in_sec} sec".format(fft_step_size_in_sec=parameters.fft_step_size))
	if parameters.mfcc:
		logger.info("{num_mfcc}-dim MFCCs will be the input.".format(num_mfcc=parameters.num_mfcc))
	elif parameters.formant:
		if parameters.use_pitch:
			lowest_formant = 0
		else:
			lowest_formant = 1
		logger.info("F{lowest_formant}-F{highest_formant} will be the input.".format(lowest_formant=lowest_formant, highest_formant=parameters.num_formants))
		logger.info("Formant freqs are first divided by the half Nyquist freq. and then -1.0 is added s.t. the input range in [-1.0,1.0].")
	else:
		logger.info("log(abs(STFT(wav))) + {eps}) / {normalizer} will be the input.".format(eps=parameters.epsilon, normalizer=parameters.data_normalizer))

	train_dataset = data_parser.get_data(data_type='train', transform=transform, channel=parameters.channel)
	valid_dataset = data_parser.get_data(data_type='valid', transform=transform, channel=parameters.channel)
	

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