# coding: utf-8

import numpy as np
import pandas as pd
import os.path, argparse, codecs

def randomly_assign_data_type(df, num_samples, random_state = None):
	df_samples = df.sample(n=num_samples, random_state=random_state)
	return df_samples


def split_data(df, valid_ratio, test_ratio, random_state = None):
	# Sample test data.
	num_samples = np.int(np.ceil(df.shape[0] * test_ratio))
	df_test = randomly_assign_data_type(df, num_samples, random_state)
	df.loc[df_test.index,'data_type'] = 'test'
	
	# Sample validation data.
	df_non_test = df[df.data_type!='test']
	num_samples = np.int(np.ceil(df.shape[0] * valid_ratio))
	df_valid = randomly_assign_data_type(df_non_test, num_samples, random_state)
	df.loc[df_valid.index, 'data_type'] = 'valid'

	df.loc[df.data_type.isnull(), 'data_type'] = 'train'
	return df


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('annotation_csv_path', type=str, help='Path to the annotation file.')
	parser.add_argument('-v','--valid_ratio', type=np.float64, default=0.1, help='Ratio of validation data.')
	parser.add_argument('-t','--test_ratio', type=np.float64, default=0.1, help='Ratio of test data per annotation type.')
	parser.add_argument('-s','--seed', type=np.int, default=111, help='Random seed.')

	params = parser.parse_args()

	df = pd.read_csv(params.annotation_csv_path)
	random_state = np.random.RandomState(params.seed)
	df = split_data(df, params.valid_ratio, params.test_ratio, random_state=random_state)

	filepath_wo_ext, ext = os.path.splitext(params.annotation_csv_path)
	df.to_csv('{filepath_wo_ext}_valid-ratio-{valid_ratio}_test-ratio-{test_ratio}_seed-{seed}{ext}'.format(filepath_wo_ext=filepath_wo_ext, valid_ratio = params.valid_ratio, test_ratio=params.test_ratio, seed=params.seed, ext=ext), index=False)
