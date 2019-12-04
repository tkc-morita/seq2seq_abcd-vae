# coding: utf-8

import numpy as np
import pandas as pd
# import sklearn.manifold as skman
# import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
import argparse, itertools, os

def plot_scores(df, category_col, title, hue=None, category_order=None, hue_order=None, show_ci=False, save_path=None, horizontal=True, use_hatch=False):
	if horizontal:
		x = 'score'
		y = category_col
	else:
		x = category_col
		y = 'score'
	if category_col=='model':
		fig,ax = plt.subplots()
	else:
		fig,ax = plt.subplots(figsize=(6,16))
	if use_hatch:
		ax = sns.barplot(x=x, y=y, hue=hue, data=df, order=category_order, hue_order=hue_order, ax=ax, color='w')
		hatches = ['', '///', 'OO', 'xxx', '**', '+++']
		if not hue_order is None:
			hatches = [h for h in hatches for c in category_order]
			# hatches = hatches[:len(hue_order)] * len(category_order)
		for bar,hatch in zip(ax.patches, hatches):
			bar.set_hatch(hatch)
			# bar.set_lw(10)
			bar.set_edgecolor('k')
	else:
		ax = sns.barplot(x=x, y=y, hue=hue, data=df, order=category_order, hue_order=hue_order, ax=ax)
	if category_col=='model':
		if horizontal:
			ax.set_yticklabels(ax.get_yticklabels())
		else:
			ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
	if show_ci:
		if hue is None:
			label_poss = ax.get_yticks() if horizontal else ax.get_xticks()
			scores = df.score.values
			errors_low = scores-df['2.5_percentile'].values
			errors_high = df['97.5_percentile'].values - scores
		else:
			label_poss = []
			scores = []
			errors_low = []
			errors_high = []
			for rect,(hue_label,category_label) in zip(ax.patches, itertools.product(hue_order, category_order)):
				# print(rect.get_x(), category_label, hue_label)
				if horizontal:
					label_poss.append(rect.get_y() + rect.get_height() * 0.5)
					score = rect.get_x() + rect.get_width()
				else:
					label_poss.append(rect.get_x() + rect.get_width() * 0.5)
					score = rect.get_y() + rect.get_height()
				scores.append(score)
				e_l,e_h = df.loc[
							(df[category_col]==category_label)&(df[hue]==hue_label),
							['2.5_percentile','97.5_percentile']].values.reshape(2)
				errors_low.append(score-e_l)
				errors_high.append(e_h-score)
		# scores = df['score'].values
		# errors = (scores-df['2.5_percentile'].values,
		# 			df['97.5_percentile'].values-scores)
		if horizontal:
			xs = scores
			ys = label_poss
			orient2error = {'xerr':(errors_low, errors_high)}
		else:
			xs = label_poss
			ys = scores
			orient2error = {'yerr':(errors_low, errors_high)}
		ax.errorbar(
			xs,
			ys,
			ecolor='black',
			ls='none',
			**orient2error
			)
	ax.set_title(title)
	if not hue is None:
		legend_kwargs = {}
		if horizontal:
			legend_kwargs = {'loc':'lower left','bbox_to_anchor':(1, 0.0)}
			loc = 'lower left'
		else:
			loc = 'upper left'
			legend_kwargs = {'loc':'upper left','bbox_to_anchor':(1, 1.1)}
		ax.legend(**legend_kwargs)
	if save_path is None:
		plt.show()
	else:
		plt.savefig(save_path, bbox_inches='tight')


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('score_name', type=str, default=None, help='The name of the score to plot.')
	parser.add_argument('-t', '--data_type', type=str, default='test', help='train, valid, or test.')
	parser.add_argument('-S', '--save_path', type=str, default=None, help='Path to the file where the figure is saved.')
	parser.add_argument('--use_hatch', action='store_true', help='If selected, plot monochrome with hatches.')
	parser.add_argument("data_paths", help="Sequence of model_name_1 data_path_1 model_name_2 data_path_2 ...", nargs=argparse.REMAINDER)
	args = parser.parse_args()

	dfs = []
	for model_name, datapath in zip(args.data_paths[0::2], args.data_paths[1::2]):
		df = pd.read_csv(datapath)
		df['model'] = model_name.replace(r'\n','\n')
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
		category_order = model_order
		hue = None
		hue_order = None
	else:
		x = 'class_label'
		category_order = df.class_label.drop_duplicates().tolist()
		hue = 'model'
		hue_order = model_order
	if args.score_name=='f1':
		args.score_name = 'F1'
	plot_scores(
		df,
		x,
		'{score_name} ({data_type})'.format(score_name=args.score_name, data_type=args.data_type),
		save_path=args.save_path,
		show_ci=show_ci,
		hue=hue,
		category_order=category_order,
		hue_order=hue_order,
		use_hatch=args.use_hatch
		)