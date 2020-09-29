import pandas as pd
import argparse

parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('--name', type=str, default = None)
parser.add_argument('--target', type=str, default = None)

args = parser.parse_args()
prefix = args.name
target = args.target

df = pd.read_csv('./applications/%s/data/rnn_input_item_seq_with_cluster_integer.csv' % prefix)
# df = pd.read_csv('./applications/%s/data/rnn_input_data_integer.csv' % prefix)

df_seqgan = df.drop([target,'clustered_label'],axis=1)

df_seqgan.to_csv('./applications/%s/SeqGAN/seqGAN_input_data.txt'  % prefix,sep=' ',header = False,index = False)

