import argparse
import csv

import pandas as pd

parser = argparse.ArgumentParser('Pajigsaw training and evaluation script', add_help=False)
parser.add_argument('--similarity_file', type=str, required=True, help='path to similarity matrix', )

args = parser.parse_args()

similarity_map = pd.read_csv(args.similarity_file, index_col=0)
gt_data = []
for key in similarity_map.index:
    group = key.split("_")[0]
    gt_data.append({'file': key, 'group': group})

with open('gt.csv', 'w') as f:
    writer = csv.DictWriter(f, fieldnames=['file', 'group'])
    writer.writerows(gt_data)

distance_map = 1. - similarity_map
distance_map.to_csv('distance_matrix.csv')