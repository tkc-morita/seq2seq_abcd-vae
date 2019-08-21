# coding: utf-8

import numpy as np
import pandas as pd
import scipy.stats as sstats
import argparse

def get_accuracy(df):
	df_target = df[df.is_target]
	accuracy = df_target.is_most_probable.sum() / df_target.shape[0]
	return accuracy

def get_f1_score_of_class(df, class_label):
	df_positive = df[(df.class_label==class_label) & (df.is_most_probable)]
	if df_positive.shape[0]:
		precision = df_positive.is_target.sum() / df_positive.shape[0]
	else: # No predicted value. Returns 0.0 following sklearn.
		# print('No positive predicted. Returns precision=0.0 following sklearn.')
		precision = 0.0
	df_target = df[(df.label==class_label) & (df.is_target)]
	recall = df_target.is_most_probable.sum() / df_target.shape[0]
	if precision and recall:
		f1 = sstats.hmean([precision, recall])
	else:
		f1 = 0.0
	return {'f1':f1, 'precision':precision, 'recall':recall}

def main(df, num_samples, random_state=None, seed=None, percentiles=[2.5, 5, 25, 50, 75, 95, 97.5]):
	accuracy = get_accuracy(df)
	results = {'all':{'accuracy':{'score':accuracy,'samples':[]}},}
	class_labels = df.label.unique()
	for class_label in class_labels:
		scores = get_f1_score_of_class(df, class_label)
		results.update({
			class_label:{score_name:{'score':value,'samples':[]}
			for score_name,value in scores.items()}
		})
	
	# bootstrap sampling
	if random_state is None:
		random_state = np.random.RandomState(seed=seed)
	# data_ixs = df.data_ix.drop_duplicates()
	for sample_ix in range(num_samples):
		df_sampled = df.pivot(
							index='data_ix',
							columns='class_label',
							values='is_most_probable'
							# values=['is_most_probable','label','is_target']
						).sample(
							frac=1.0, replace=True, random_state=random_state
						)
		df_sampled = df_sampled.reset_index().melt(
						col_level=0,
						id_vars=['data_ix'],
						var_name='class_label'
						)
		# df_sampled = df_sampled.melt(id_vars=['data_ix'], var_name='class_label')
		df_sampled = df_sampled.merge(
						df.loc[:,['data_ix','class_label','is_most_probable','label','is_target']],
						on=['data_ix','class_label'],
						how='left'
						)
		# df_sampled = pd.concat([
		# 				df[df.data_ix==data_ix]
		# 				for data_ix
		# 				in data_ixs.sample(frac=1.0, replace=True, random_state=random_state).tolist()
		# 				], axis=0, ignore_index=True)
		results['all']['accuracy']['samples'].append(
			get_accuracy(df_sampled)
		)
		for class_label in class_labels:
			scores = get_f1_score_of_class(df_sampled, class_label)
			for score_name,value in scores.items():
				results[class_label][score_name]['samples'].append(value)
	results_reformatted = []
	for class_label, subdict in results.items():
		for score_name, subsubdict in subdict.items():
			entries = [class_label, score_name, subsubdict['score']]
			if subsubdict['samples']:
				entries += [np.percentile(subsubdict['samples'], q) for q in percentiles]
			else:
				entries += [np.nan for q in percentiles]
			results_reformatted.append(entries)
	df_results = pd.DataFrame(results_reformatted, columns=['class_label','score_type','score']+['{}_percentile'.format(q) for q in percentiles])
	return df_results




if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('data_path', type=str, help='Path to the data.')
	parser.add_argument('save_path', type=str, help='Path to the results csv.')
	parser.add_argument('-n', '--num_samples', type=int, default=100000, help='# of bootstrap samples.')
	parser.add_argument('--seed', type=int, default=111, help='Random seed.')
	args = parser.parse_args()

	df = pd.read_csv(args.data_path)
	random_state = np.random.RandomState(args.seed)
	results = []
	for data_type, sub_df in df.groupby('data_type'):
		if data_type=='test':
			num_samples = args.num_samples
		else:
			num_samples = 0
		sub_df_results = main(sub_df, num_samples, random_state=random_state)
		sub_df_results['data_type'] = data_type
		results.append(sub_df_results)
	df_results = pd.concat(results, axis=0, ignore_index=True)
	df_results.to_csv(args.save_path, index=False)