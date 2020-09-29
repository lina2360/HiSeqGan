import pandas as pd
import numpy as np
import argparse
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('--name', type=str, default = None)

args = parser.parse_args()
prefix = args.name

raw_data = pd.read_csv('./applications/%s/data/rnn_input_item_seq_with_cluster_integer.csv' % prefix)
# raw_data = pd.read_csv('./applications/%s/data/rnn_input_data_integer.csv' % prefix)

rnn_data = raw_data

# delete ItemShortName column and reset index
raw_data= raw_data.drop(['id'], axis=1)
raw_data.columns = range(raw_data.shape[1])

# drop duplicates rows and reset index
raw_data = raw_data.head(110).drop_duplicates()
raw_data.index = range(raw_data.shape[0])
print("Raw data : ",raw_data)
# read seqgan generate sequence data
result_data = pd.read_csv('./applications/%s/SeqGAN/seqGAN_output_data.txt'  % prefix,header=None,sep=' ')
print("SeqGAN data shape : ",result_data.shape)

# print(result_data.iloc[0][:91]-raw_data.iloc[0][:91])
# sum((result_data.iloc[0][:91] - raw_data.iloc[0][:91])**2)


result = result_data.loc[0].to_frame().T
for i in range(result_data.shape[0]):
# for i in range(10): 
    print("i=",i)
    row = result_data.loc[0].to_frame().T
    min_error = 100000000000000

    for j in range(raw_data.shape[0]):
    # for j in range(1):
        # print("min_error : ",min_error)
        # print('j=',j)
        # print("now : ",result_data.iloc[j][:91] - raw_data.iloc[i][:91])
        if min_error >= (sum((result_data.iloc[i][:13] - raw_data.iloc[j][:13])**2)) :
            row = raw_data.loc[j].to_frame().T
            min_error = sum((result_data.iloc[i][:13] - raw_data.iloc[j][:13])**2)
            # print('cluster')
            #print(sum((result_data.iloc[i] - raw_data.iloc[j][:96])**2))
    result = pd.concat([result,row])
    # print(result)

result_add_item = result.reset_index(drop=True)
# delete first row
result_add_item = result_add_item.drop(result_add_item.index[[0]])
# change cluster datatype from float to int64
seqgan_data = result_add_item.astype('int64')
# add item column in seqGAN data and modify column names for merge
seqgan_data.insert(0,'id','none')
seqgan_data.columns = rnn_data.columns
# export data
pd.concat([rnn_data,seqgan_data]).to_csv('./applications/%s/data/rnn_input_item_seq_with_cluster_label_seqgan_train.csv'  % prefix,index=False)

# result_add_item.astype('int64').to_csv('../data/seqGAN_sequence_with_clustering.csv',index=False,header=False)






