import numpy as np
import pandas as pd
import random
import math
from random import sample
# from RNN_tensorflow import WPG_RNN\
from RNN_tensorflow_all_center_point import WPG_RNN
#from RNN_tensorflow_all import WPG_RNN
from data_preprocess_all_center_point import format_row_to_layerformat
import datetime
import os
import datetime
import get_ghsom_dim
import argparse

parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('--name', type=str, default = None)
parser.add_argument('--input_type', type=str, default = None)

args = parser.parse_args()
prefix = args.name
input_type = args.input_type
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

print("========== Start Training =============")
print("Start Training Time:",datetime.datetime.now())
outpu_format = datetime.datetime.now().strftime('%Y-%m-%d_%H_%M_%S')
input_file_path = ''
if input_type =='rnn':
    input_file_path = './applications/%s/data/rnn_input_item_seq_with_cluster_integer.csv' % prefix
else:
    input_file_path = './applications/%s/data/rnn_input_item_seq_with_cluster_label_seqgan.csv' % prefix
# input_file_path = '../data/WPG_item_sequence_with_cluster_label.csv'
df = pd.read_csv(input_file_path, low_memory=False)
df = df.head(10)
# df = df[0:2].reset_index(drop=True)
# print(df)

layers,max_layer,number_of_digits = get_ghsom_dim.layers(prefix)
layer = sum(number_of_digits)

# ======
# Here train RNN with all items
# =====
base_length = 100
not_satisify_cluster = []

final_train_results = np.zeros((1,layer+3))
final_results = np.zeros((1,layer+3))
# final_results = np.zeros((1,3))

df_nominal = df.loc[:, ['clustered_label']]
df_input = df.drop(['ItemShortName', 'clustered_label'], axis=1)

print("===== format row to layerformat =====")
satisify_input_ornot, input_sequence, max_sequence_length, min_sequence_length = format_row_to_layerformat(df_input, base_length, prefix,layer,number_of_digits)
print(satisify_input_ornot)


if satisify_input_ornot:
    print("===== input sequence =====")
    print(input_sequence)

    # ======== Training and Testing Part ===========
    RNN = WPG_RNN(input_sequence, max_sequence_length, min_sequence_length, 100, 0.001, 10000, 0.01, 1, 0, len(number_of_digits), prefix, outpu_format)
    # RNN = WPG_RNN(input_sequence, max_sequence_length, min_sequence_length, 100, 0.001, 10, 0.01, 1, 0)
    # (self, df, max_sequence_length, min_sequence_length, num_neurons, learning_rate, max_num_train_iterations, accept_loss, batch_size, training_start_point)
    train_result = RNN.train(input_sequence)
    result = RNN.test(input_sequence)
    # correct_layer = np.add(correct_layer, result)
    # ========= end training and testing part =======

    print('===== final accuracy =====')
    train_result_df = pd.DataFrame(data=train_result)
    result_df = pd.DataFrame(data=result)
    # print(final_results_df)
    final_train_results = pd.concat([df_nominal, train_result_df], axis=1)
    final_results = pd.concat([df_nominal, result_df], axis=1)

    print(final_results)
    final_results.to_csv('./applications/%s/RNN/result/all_item_prediction_wMSE_%s.csv' % (prefix, outpu_format), index=False)
# else:
#     not_satisify_cluster.append(cluster)


# final_results = final_results[1:, :]
# columns = ['cluster_label', 'layer1_correctness', 'layer2_correctness', 'layer3_correctness', 'delay_correctness']
# final_results_df = pd.DataFrame(data=final_results, columns=columns)
# print(final_results_df)
# final_results[['cluster_label']] = final_results[['cluster_label']].astype(int)
# final_results = final_results.fillna(0)
# final_results = final_results.astype(int)


print("======= total accuracy =======")
print(final_results)

final_train_results.to_csv('./applications/%s/RNN/result/%s_All_wMSE_layers_train_center_point_%s.csv' % (prefix,input_type,  outpu_format), index=False)
final_results.to_csv('./applications/%s/RNN/result/%s_All_wMSE_layers_center_point_%s.csv' % (prefix,input_type,  outpu_format), index=False)

print("======== total not satisify cluster count ===========")
print(not_satisify_cluster)

print("========== End Training =============")
print("End Training Time:",datetime.datetime.now())