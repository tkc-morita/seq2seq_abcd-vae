# coding: utf-8

"""
Extract intervals with non-empty labels from a Praat .TextGrid file, and save them in a csv.
"""

import pandas as pd
import os, glob, argparse

def parse_TextGrid(path):
	with open(path, 'r') as f:
		tier_name = None
		interval_id = None
		entries = []
		for line in f.readlines():
			line = line.rstrip()
			if 'name = ' in line:
				tier_name = line.split('name = ')[1].replace('"','')
				interval_id = None
			elif tier_name and 'intervals [' in line:
				interval_id = line.split('intervals [')[1].split(']')[0]
			elif interval_id and 'xmin = ' in line:
				interval_min = line.split('xmin = ')[1].replace(' ','')
			elif interval_id and 'xmax = ' in line:
				interval_max = line.split('xmax = ')[1].replace(' ','')
			elif interval_id and 'text = ' in line:
				label = line.split('text = ')[1].replace('"','')
				entries.append([os.path.splitext(os.path.basename(path))[0]+'.wav',tier_name,interval_id,interval_min,interval_max,label])
	return entries


def main_loop(dir_path):
	entries = []
	for file_path in glob.glob(os.path.join(dir_path, '*.TextGrid')):
		entries += parse_TextGrid(file_path)
	return pd.DataFrame(entries, columns=['input_path','tier_name','interval_id','onset','offset','label'])

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('data_dir', type=str, help='Path to the directory containing .TextGrid files')
	parser.add_argument('save_path', type=str, help='Path to the csv file where results are saved.')
	params = parser.parse_args()

	df = main_loop(params.data_dir)
	# for input_path, sub_df in df.groupby('input_path'):
	# 	print(input_path)
	# 	as_list = sub_df.label.tolist()
	# 	print(set(as_list[0:len(as_list):2]))
	# 	print(set(as_list[1:len(as_list):2]))
	if not os.path.isdir(os.path.split(params.save_path)[0]):
		os.makedirs(os.path.split(params.save_path)[0])

	df[df.label!=''].to_csv(params.save_path, index=False)
