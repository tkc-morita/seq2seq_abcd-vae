# coding: utf-8

import pandas as pd
import sklearn.manifold as skman
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
import argparse, itertools, os

def plot_features(df):
	parameter_types = len(df.parameter_name.unique())
	ndim = len(df.feature_dim.unique())
	# fig, axes = plt.subplots(ndim, parameter_types)
	# for sub_axes,(par_name,pre_sub_df) in zip(axes, df.groupby('parameter_name')):
	# 	for ax, (d, sub_df) in zip(sub_axes, pre_sub_df.groupby('feature_dim')):
	# 		sns.distplot(sub_df.parameter_value, ax=ax)
		# sns.violinplot(x='feature_dim', y='parameter_value', data=sub_df, ax=ax)
	# df['feature_dim'] = pd.Categorical(df.feature_dim)
	# sns.violinplot(x='feature_dim', y='parameter_value', hue='parameter_name', data=df, bw=.02)
	# plt.show()
	for par_name,pre_sub_df in df.groupby('parameter_name'):
		for d, sub_df in pre_sub_df.groupby('feature_dim'):
			sns.distplot(sub_df.parameter_value)
			plt.title('{par_name} ({d}-th dim)'.format(par_name=par_name, d=d))
			plt.show()

def tsne(df, title=None, save_path=None):
	tsne = skman.TSNE(n_components=2, init='pca', random_state=0)
	df_pv = df.pivot(index='data_ix', columns='dim', values='parameter_value')
	embedded = tsne.fit_transform(df_pv.values)
	df_pv = pd.merge(df_pv, df.drop_duplicates(subset='data_ix'), on='data_ix', how='left')
	df_pv.loc[:,'embed_0'] = embedded[:,0]
	df_pv.loc[:,'embed_1'] = embedded[:,1]
	labels = []
	v_included = False
	x_included = False
	for l in sorted(df_pv.label.unique().tolist()):
		if l == 'v':
			v_included = True
		elif l == 'x':
			x_included = True
		else:
			labels.append(l)
	if x_included:
		labels = ['x'] + labels
	if v_included:
		labels = ['v'] + labels
	df_pv.loc[:,'label'] = pd.Categorical(df_pv.label, categories=labels, ordered=True)
	color_x_markers = [('C{color_ix}'.format(color_ix=color_ix),marker)
						for marker,color_ix
						in itertools.product(['o','v','s'],range(10))]
	label2color = {}
	label2marker = {}
	for ix,l in enumerate(labels):
		color,marker = color_x_markers[ix]
		label2color[l] = color
		label2marker[l] = marker
	# df_pv.groupby('label').plot.scatter(x='embed_0', y='embed_1')
	ax = sns.scatterplot(x='embed_0', y='embed_1', data=df_pv, hue='label', style='label', palette=label2color, markers=label2marker)
	ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
	if not title is None:
		ax.set_title(title)
	if save_path is None:
		plt.show()
	else:
		plt.savefig(save_path, bbox_inches="tight")
	ax.clear()


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('data', type=str, help='Path to the csv file containing features.')
	parser.add_argument('-S','--save_dir', type=str, default=None, help='Path to the directory where figures are saved.')
	parser.add_argument('-p', '--parameter', type=str, default=None, help='If given, use only the specified parameter.')
	args = parser.parse_args()

	df = pd.read_csv(args.data)
	if args.parameter is None:
		dim = df.feature_dim.max()+1
		for par_ix,(par,sub_df) in enumerate(df.groupby('parameter_name')):
			df.loc[sub_df.index,'dim'] = sub_df.feature_dim + par_ix*dim
	else:
		df = df[df.parameter_name==args.parameter]
		df.loc[:,'dim'] = df.feature_dim
	# print(df.label.unique())
	# df.loc[:,'label'] = pd.Categorical(df.label, categories=list('vxabcdefghijklmnopqrstuwyz'))

	if (not args.save_dir is None) and (not os.path.isdir(args.save_dir)):
		os.makedirs(args.save_dir)
	# plot_features(df)
	# sub_df = df[df.input_path=='cut_b03.wav']
	for path,sub_df in df.groupby('input_path'):
		if args.save_dir is None:
			save_path = None
		else:
			save_path = os.path.join(args.save_dir, os.path.splitext(path)[0] + '.png')
		tsne(sub_df, title=path, save_path=save_path)