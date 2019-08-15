# encoding: utf-8

import torch
from modules.data_utils import Compose
import numpy as np
import pandas as pd
from modules import data_utils
import learning
import os, argparse, itertools


class Predictor(learning.Learner):
	def __init__(self, model_config_path, device = 'cpu'):
		self.device = torch.device(device)
		self.retrieve_model(checkpoint_path = model_config_path, device=device)
		for param in self.parameters():
			param.requires_grad = False
		self.softmax = torch.nn.Softmax(dim=-1)
		self.modules.append(self.softmax)
		[m.eval() for m in self.modules]


	def predict(self, data, is_packed = False, to_numpy = True):
		if not is_packed:
			if not isinstance(data, list):
				data = [data]
			data = torch.nn.utils.rnn.pack_sequence(data)
		with torch.no_grad():
			data = data.to(self.device)
			last_hidden = self.encoder(data)
			weights = self.classifier(last_hidden)
			probs = self.softmax(weights)
		if to_numpy:
			probs = probs.data.cpu().numpy()
		return probs


	def predict_dataset(self, dataset, to_numpy = True, batch_size=1):
		dataloader = data_utils.DataLoader(dataset, batch_size=batch_size)
		class_probs = []
		data_ixs = []
		most_probable_classes = []
		for data, _, ix_in_list in dataloader:
			probs = self.predict(data, is_packed=True, to_numpy=to_numpy)
			class_probs += probs.tolist()
			data_ixs += ix_in_list.tolist()
			most_probable_classes += probs.argmax(-1).tolist()
		df_prob = pd.DataFrame(class_probs)
		df_prob['data_ix'] = data_ixs
		df_prob['most_probable'] = most_probable_classes
		df_prob = df_prob.melt(id_vars=['data_ix','most_probable'], var_name='class_ix', value_name='prob')
		df_prob['is_most_probable'] = df_prob.class_ix==df_prob.most_probable
		df_prob = df_prob.drop(columns=['most_probable'])
		return df_prob

def get_parameters():
	par_parser = argparse.ArgumentParser()

	par_parser.add_argument('model_path', type=str, help='Path to the configuration file of a trained model.')
	par_parser.add_argument('input_root', type=str, help='Path to the root directory under which inputs are located.')
	par_parser.add_argument('annotation_file', type=str, help='Path to the annotation csv file.')
	par_parser.add_argument('-N', '--data_normalizer', type=float, default=1.0, help='Normalizing constant to devide the data.')
	par_parser.add_argument('--annotation_sep', type=str, default=',', help='Separator symbol of the annotation file. Comma "," by default (i.e., csv).')
	par_parser.add_argument('-d', '--device', type=str, default='cpu', help='Computing device.')
	par_parser.add_argument('-S', '--save_path', type=str, default=None, help='Path to the file where results are saved.')
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
	par_parser.add_argument('-E','--epsilon', type=float, default=2**(-15), help='Small positive real number to add to avoid log(0).')
	par_parser.add_argument('-b', '--batch_size', type=int, default=1, help='Batch size.')

	return par_parser.parse_args()

if __name__ == '__main__':
	parameters = get_parameters()

	save_path = parameters.save_path
	if save_path is None:
		save_path = os.path.join(parameters.input_root, 'autoencoded.csv')
	save_dir = os.path.dirname(save_path)
	if not os.path.isdir(save_dir):
		os.makedirs(save_dir)

	data_parser = data_utils.Data_Parser(parameters.input_root, parameters.annotation_file, annotation_sep=parameters.annotation_sep)
	fs = data_parser.get_sample_freq() # Assuming all the wav files have the same fs, get the 1st file's.

	fft_frame_length = int(np.floor(parameters.fft_frame_length * fs))
	fft_step_size = int(np.floor(parameters.fft_step_size * fs))

	# Get a model.
	predictor = Predictor(parameters.model_path, device=parameters.device)

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
	elif parameters.formant:
		get_formants = data_utils.Formant(fs, parameters.fft_frame_length, parameters.fft_step_size, num_formants=parameters.num_formants, use_pitch=parameters.use_pitch)
		half_nyquist_freq = fs / 4
		normalize = data_utils.Transform(lambda x: (x / half_nyquist_freq) - 1.0)
		transform = Compose([get_formants, to_tensor, normalize])
	else:
		stft = data_utils.STFT(fft_frame_length, fft_step_size, window=parameters.fft_window_type, centering=not parameters.fft_no_centering)
		log_and_normalize = data_utils.Transform(lambda x: (x + parameters.epsilon).log() / parameters.data_normalizer)
		transform = Compose([to_tensor,stft,log_and_normalize])

	dataset = data_parser.get_data(transform=transform, channel=parameters.channel)

	df_prob = predictor.predict_dataset(dataset, batch_size=parameters.batch_size)
	df_prob = df_prob.sort_values(['data_ix','class_ix'])
	df_prob['class_label'] = df_prob.class_ix.map(data_parser.get_ix2label())
	if 'label' in data_parser.df_annotation.columns:
		df_prob = df_prob.merge(data_parser.df_annotation, how='left', left_on='data_ix', right_index=True)
		df_prob['is_target'] = df_prob.class_label==df_prob.label
	df_prob = df_prob.drop(columns=[col for col in df_prob.columns if col.startswith('Unnamed:')])
	df_prob.to_csv(save_path, index=False)

