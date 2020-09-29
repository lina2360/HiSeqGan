import pandas as pd
import argparse
import get_ghsom_dim

parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('--name', type=str, default = None)
args = parser.parse_args()

prefix = args.name
seq_name = '%s-item-seq' % prefix

layers,max_layer,number_of_digits = get_ghsom_dim.layers(seq_name)
# layers,max_layer,number_of_digits = get_ghsom_dim.layers(prefix)
layer = sum(number_of_digits)
df = pd.read_csv('./applications/%s/data/rnn_input_item_seq_with_cluster.csv' % prefix)
# df = pd.read_csv('./applications/%s/data/%s_with_clustered_label.csv' % (prefix,prefix))

df['clustered_label'] = df['clustered_label']*pow(10,layer)
df['clustered_label']  = df['clustered_label'].astype('int64')
# df = df[['id','week1', 'week2', 'week3','week4','week5', 'week6', 'week7', 'week8', 'week9', 'week10', 'week11', 'week12',  'week13',  'week14', 'week15', 'week16', 'week17','week18','week19','week20','week21','week22','week23','week24','week25','week26','week27','week28','week29','week30','week31','week32','week33','week34','week35','week36','clustered_label'  ]]
df = df[['id','week1', 'week2', 'week3','week4','week5', 'week6', 'week7', 'week8', 'week9', 'week10', 'week11', 'week12',  'week13',  'week14', 'week15', 'week16', 'week17','week18','clustered_label']]
df.to_csv('./applications/%s/data/rnn_input_item_seq_with_cluster_integer.csv'  % prefix,index=False)
# df.to_csv('./applications/%s/data/rnn_input_data_integer.csv'  % prefix,index=False)