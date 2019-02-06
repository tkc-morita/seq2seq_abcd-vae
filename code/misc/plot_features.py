# coding: utf-8

import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

def plot_features(df):
	# parameter_types = len(df.parameter_name.unique())
	# f, axes = plt.subplots(parameter_types, 1)
	# for ax,(par_name,sub_df) in zip(axes, df.groupby('parameter_name')):
		# sns.violinplot(x='feature_dim', y='parameter_value', data=sub_df, ax=ax)
	df['feature_dim'] = pd.Categorical(df.feature_dim)
	sns.violinplot(x='feature_dim', y='parameter_value', hue='parameter_name', data=df, bw=.02)
	plt.show()


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('data', type=str, help='Path to the csv file containing features.')
	args = parser.parse_args()

	df = pd.read_csv(args.data)

	plot_features(df)