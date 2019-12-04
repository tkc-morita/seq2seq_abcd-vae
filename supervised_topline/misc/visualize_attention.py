# coding: utf-8

import numpy as np
import pandas as pd
# import sklearn.manifold as skman
import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
import argparse, itertools, os


def visualize_attention(df, save_dir=None,  step_size_in_sec=None):
	df_length = (df.groupby('data_ix').time_ix.max()+1).to_frame().reset_index().rename(columns={'time_ix':'seq_length'})
	df = df.merge(df_length, how='left', on='data_ix')
	df = df.sort_values('attention_weight', ascending=False).drop_duplicates(['data_ix','head_ix'])
	max_length = df_length.seq_length.max()
	min_length = df_length.seq_length.min()
	if step_size_in_sec is None:
		x = 'time_ix'
		seq_length_title = 'log interval length (#time steps)'
	else:
		x = 'time (ms)'
		df.loc[:,x] = df.time_ix * step_size_in_sec * 1000
		df.loc[:,'seq_length'] = df.seq_length * step_size_in_sec * 1000
		max_length *= step_size_in_sec * 1000
		min_length *= step_size_in_sec * 1000
		seq_length_title = 'log interval length (msec)'
	# df.loc[:,'normalized_pos'] = df.time_ix / df.seq_length
	num_heads = len(df.head_ix.unique())
	if save_dir is not None and not os.path.isdir(save_dir):
		os.makedirs(save_dir)
	cmap = sns.diverging_palette(250, 15, l=65, center="light", as_cmap=True)
	norm = matplotlib.colors.LogNorm(vmin=min_length, vmax=max_length)
	for head_ix, sub_df in df.groupby('head_ix'):
		g = sns.FacetGrid(sub_df,
				row='class',
				col='length',
				ylim=(0.0,1.0),
				xlim=(0,max_length),
				palette='seismic',
				)
		g.map(
			facet_scatter,
			x,
			'attention_weight',
			'seq_length',
			norm=norm,
			# vmin=0,
			# vmax=max_length,
			alpha=0.5,
			cmap=cmap
			)
		# cax = g.fig.add_axes([.94, .25, .02, .6])
		ax = g.fig.get_axes()[1]
		points = plt.scatter([], [], c=[], norm=norm, cmap=cmap)
		cbar = g.fig.colorbar(points, ax=ax)
		cbar.set_label(seq_length_title)
		if save_dir is None:
			plt.tight_layout()
			plt.show()
		else:
			plt.savefig(os.path.join(save_dir, 'head-{}.png'.format(head_ix)), bbox_inches='tight')
		plt.clf()
	# for label,sub_df in df.groupby('class_label'):
	# 	fig,axes = plt.subplots(num_heads)
		# plt.subplots_adjust(hspace=1.0)
		# for ax,(head_ix,subsub_df) in zip(axes,sub_df.groupby('head_ix')):
		# 	ax = sns.scatterplot(x='time_ix', y='attention_weight', data=subsub_df, ax=ax)
		# 	ax.set_title('Predicted {label} ({head_ix}-th head)'.format(label=label, head_ix=head_ix))
		# 	ax.set_ylabel('')
		# 	ax.set_ylim((0.0,1.0))
		# if save_dir is None:
		# 	plt.tight_layout()
		# 	plt.show()
		# else:
		# 	plt.savefig(os.path.join(save_dir, label+'.png'), bbox_inches='tight')
		# plt.clf()

def facet_scatter(x,y,c,**kwargs):
	kwargs.pop('color')
	plt.scatter(x, y, c=c, **kwargs)

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("dependency_path", help="Path to the dependency data.")
	parser.add_argument("classification_path", help="Path to the classification data.")
	parser.add_argument('-S', '--save_dir', type=str, default=None, help='Path to the directory where the figure is saved.')
	parser.add_argument('--step_size', type=float, default=None, help='Step size of the data sequence in sec.')
	args = parser.parse_args()

	df_data = pd.read_csv(args.dependency_path)

	df_class = pd.read_csv(args.classification_path)
	df_class = df_class[df_class.is_most_probable]
	df_data = df_data.merge(df_class, how='left', on='data_ix')

	df_data.loc[:,'length'] = df_data.class_label.map(lambda s: 'long' if 'H' in s else 'short')
	df_data.loc[:,'length'] = pd.Categorical(df_data.length, ['short', 'long'], ordered=True)
	df_data.loc[:,'class'] = df_data.class_label.map(lambda s: s[0])

	visualize_attention(df_data, save_dir=args.save_dir, step_size_in_sec=args.step_size)