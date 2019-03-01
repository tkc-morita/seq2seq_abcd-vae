# coding: utf-8

import pandas as pd
import sklearn.manifold as skman
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
import argparse, itertools

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

def tsne(df):
	tsne = skman.TSNE(n_components=2, init='pca', random_state=0)
	df_pv = df.pivot(index='data_ix', columns='dim', values='parameter_value')
	embedded = tsne.fit_transform(df_pv.values)
	df_pv = pd.merge(df_pv, df.drop_duplicates(subset='data_ix'), on='data_ix', how='left')
	df_pv.loc[:,'embed_0'] = embedded[:,0]
	df_pv.loc[:,'embed_1'] = embedded[:,1]
	labels = df_pv.label.unique().tolist()
	color_x_markers = [('C{color_ix}'.format(color_ix=color_ix),marker)
						for marker,color_ix
						in itertools.product(['o','v'],range(10))]
	label2color = {}
	label2marker = {}
	for ix,l in enumerate(labels):
		color,marker = color_x_markers[ix]
		label2color[l] = color
		label2marker[l] = marker
	# df_pv.groupby('label').plot.scatter(x='embed_0', y='embed_1')
	sns.scatterplot(x='embed_0', y='embed_1', data=df_pv, hue='label', style='label', palette=label2color, markers=label2marker)
	plt.show()


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('data', type=str, help='Path to the csv file containing features.')
	args = parser.parse_args()

	df = pd.read_csv(args.data)

	# plot_features(df)
	sub_df = df[df.input_path=='cut_b20.wav']
	tsne(sub_df)