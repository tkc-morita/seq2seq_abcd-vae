# coding: utf-8

import numpy as np
import pandas as pd
# import sklearn.manifold as skman
# import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
import argparse, itertools, os

def plot_scores(df, x, title, hue=None, x_order=None, hue_order=None, show_ci=False, save_path=None):
	ax = sns.barplot(x=x, y='score', hue=hue, data=df, order=x_order, hue_order=hue_order)
	if x=='model':
		ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
	if show_ci:
		if hue is None:
			xs = ax.get_xticks()
			ys = df.score.values
			errors_low = ys-df['2.5_percentile'].values
			errors_high = df['97.5_percentile'].values - ys
		else:
			xs = []
			ys = []
			errors_low = []
			errors_high = []
			for rect,(hue_label,x_label) in zip(ax.patches, itertools.product(hue_order, x_order)):
				# print(rect.get_x(), x_label, hue_label)
				xs.append(rect.get_x() + rect.get_width() * 0.5)
				score = rect.get_y() + rect.get_height()
				ys.append(score)
				e_l,e_h = df.loc[
							(df[x]==x_label)&(df[hue]==hue_label),
							['2.5_percentile','97.5_percentile']].values.reshape(2)
				errors_low.append(score-e_l)
				errors_high.append(e_h-score)
		# scores = df['score'].values
		# errors = (scores-df['2.5_percentile'].values,
		# 			df['97.5_percentile'].values-scores)
		ax.errorbar(
			xs,
			ys,
			yerr=(errors_low, errors_high),
			ecolor='grey',
			ls='none'
			)
	ax.set_title(title)
	ax.legend(bbox_to_anchor=(1, 1.1), loc='upper left')
	if save_path is None:
		plt.show()
	else:
		plt.savefig(save_path, bbox_inches='tight')


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('score_name', type=str, default=None, help='The name of the score to plot.')
	parser.add_argument('-t', '--data_type', type=str, default='test', help='train, valid, or test.')
	parser.add_argument('-S', '--save_path', type=str, default=None, help='Path to the file where the figure is saved.')
	parser.add_argument("data_paths", help="Sequence of model_name_1 data_path_1 model_name_2 data_path_2 ...", nargs=argparse.REMAINDER)
	args = parser.parse_args()

	dfs = []
	for model_name, datapath in zip(args.data_paths[0::2], args.data_paths[1::2]):
		df = pd.read_csv(datapath)
		df['model'] = model_name
		dfs.append(df)
	df = pd.concat(dfs)

	# Order models by test accuracy
	df_test_accuracy = df[(df.score_type=='accuracy') & (df.data_type=='test')]
	df_test_accuracy = df_test_accuracy.sort_values(['class_label','score'], ascending=[True, False])
	model_order = df_test_accuracy.model.drop_duplicates().tolist()
	df = df[(df.score_type==args.score_name) & (df.data_type==args.data_type)]
	df.loc[:,'model'] = pd.Categorical(df.model, categories=model_order, ordered=True)
	df = df.sort_values(['class_label','model'])

	if args.data_type=='test':
		show_ci = True
	else:
		show_ci = False
	if args.score_name=='accuracy':
		x = 'model'
		x_order = model_order
		hue = None
		hue_order = None
	else:
		x = 'class_label'
		x_order = df.class_label.drop_duplicates().tolist()
		hue = 'model'
		hue_order = model_order
	plot_scores(
		df,
		x,
		'{score_name} ({data_type})'.format(score_name=args.score_name, data_type=args.data_type),
		save_path=args.save_path,
		show_ci=show_ci,
		hue=hue,
		x_order=x_order,
		hue_order=hue_order
		)