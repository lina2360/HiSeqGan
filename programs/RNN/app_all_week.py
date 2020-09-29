import numpy as np
import pandas as pd
import random
import math
from random import sample
# from RNN_tensorflow import WPG_RNN\
from RNN_tensorflow_all_week import WPG_RNN
from data_preprocess_all import format_row_to_layerformat
import os
import datetime
import argparse
import get_ghsom_dim

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


days = 5
output_format = datetime.datetime.now().strftime('%Y-%m-%d_%H_%M_%S')
input_file_path = ''
if input_type =='rnn':
    input_file_path = './applications/%s/data/rnn_input_item_seq_with_cluster_integer.csv' % prefix
else:
    input_file_path = './applications/%s/data/rnn_input_item_seq_with_cluster_label_seqgan.csv' % prefix
df = pd.read_csv(input_file_path, low_memory=False)
df = df.head(10)
print(df)

layers,max_layer,number_of_digits = get_ghsom_dim.layers(prefix)
layer = sum(number_of_digits)
# df = df[0:2].reset_index(drop=True)
# print(df)

# ===== 
# Here train RNN by each cluster
# =====
# clusters_label = df['clustered_label']
# unique_cluster = clusters_label.unique()
# slice_unique_cluster = unique_cluster[52:104]
# # slice_unique_cluster = unique_cluster[0:25]
# # slice_unique_cluster = unique_cluster[0:51]
# base_length = 100
# not_satisify_cluster = []

# # df = df.drop(['ItemShortName', 'clustered_label'], axis=1)
# # df_input = df[0:3].reset_index(drop=True)

# # final_results = np.zeros((1,3))
# final_results = np.zeros((1,5))
# final_train_results = np.zeros((1,5))

# for cluster in slice_unique_cluster:
#     print(cluster)
#     df_cluster = df.loc[df['clustered_label'] == cluster]
#     df_input = df_cluster.drop(['ItemShortName', 'clustered_label'], axis=1)

#     print("====== current cluster =======")
#     print(cluster)

#     satisify_input_ornot, input_sequence, max_sequence_length, min_sequence_length = format_row_to_layerformat(df_input, base_length)

#     if satisify_input_ornot:
#         print("===== input sequence =====")
#         print(input_sequence)

#         # ======== Training and Testing Part ===========
#         RNN = WPG_RNN(input_sequence, max_sequence_length, min_sequence_length, 100, 0.001, 13000, 0.005, 1, 0)
#         # RNN = WPG_RNN(input_sequence, max_sequence_length, min_sequence_length, 100, 0.001, 30, 0.001, 1, 0)
#         train_result = RNN.train(input_sequence)
#         result = RNN.test(input_sequence)
#         # correct_layer = np.add(correct_layer, result)
#         # ========= end training and testing part =======

#         print('===== final training accuracy =====')
#         cluster_name = np.repeat(cluster, train_result.shape[0])
#         train_result = np.column_stack((cluster_name, train_result))
#         # result = np.append(cluster,result)
#         # result = np.expand_dims(np.asarray(result), axis=0)
        
#         final_train_results = np.vstack((final_train_results, train_result))
#         print(final_train_results)


#         print('===== final accuracy =====')
#         cluster_name = np.repeat(cluster, result.shape[0])
#         result = np.column_stack((cluster_name, result))
#         # result = np.append(cluster,result)
#         # result = np.expand_dims(np.asarray(result), axis=0)
        
#         final_results = np.vstack((final_results, result))
#         print(final_results)

#     else:
#         not_satisify_cluster.append(cluster)


# final_train_results = final_train_results[1:, :]
# final_results = final_results[1:, :]

# # columns = ['cluster_label', 'distance', 'delay_correctness']
# columns = ['cluster_label', 'layer1_correctness', 'layer2_correctness', 'layer3_correctness', 'delay_correctness']
# final_train_results_df = pd.DataFrame(data=final_train_results, columns=columns)
# final_train_results_df[['cluster_label']] = final_train_results_df[['cluster_label']].astype(int)

# final_results_df = pd.DataFrame(data=final_results, columns=columns)
# final_results_df[['cluster_label']] = final_results_df[['cluster_label']].astype(int)

# # print("======= total accuracy =======")
# # print(final_results)

# final_results_df.to_csv('./result/layers/new/C_lMSE_layers_2.csv', index=False)
# final_train_results_df.to_csv('./result/layers/new/train/C_lMSE_layers_train_2.csv', index=False)

# print("======== total not satisify cluster count ===========")
# print(not_satisify_cluster)


# ======
# Here train RNN with all items
# =====
base_length = 40
not_satisify_cluster = []

final_train_results = np.zeros((1,(layer+1)))
final_results = np.zeros((1,(layer+1)))
# final_results = np.zeros((1,3))

df_nominal = df.loc[:, ['clustered_label']]
df_input = df.drop(['ItemShortName', 'clustered_label'], axis=1)

satisify_input_ornot, input_sequence, max_sequence_length, min_sequence_length, df_nominal = format_row_to_layerformat(df_input, base_length, layer,number_of_digits,df_nominal)
# print(satisify_input_ornot)

if satisify_input_ornot:
    print("===== input sequence =====")
    print(input_sequence)

    # ======== Training and Testing Part ===========
    RNN = WPG_RNN(input_sequence, max_sequence_length, min_sequence_length, 100, 0.01, 10, 0.01, 1, 0, len(number_of_digits),prefix,output_format)
    # RNN = WPG_RNN(input_sequence, max_sequence_length, min_sequence_length, 100, 0.001, 10, 0.01, 1, 0)
    # (self, df, max_sequence_length, min_sequence_length, num_neurons, learning_rate, max_num_train_iterations, accept_loss, batch_size, training_start_point)
    train_result = RNN.train(input_sequence)
    result91,data91,predict_data_1 = RNN.test(input_sequence)
    result92,data92,predict_data_2 = RNN.test_multi(data91)
    result93,data93,predict_data_3 = RNN.test_multi(data92)
    result94,data94,predict_data_4 = RNN.test_multi(data93)
    result95,data95,predict_data_5 = RNN.test_multi(data94)
    # correct_layer = np.add(correct_layer, result)
    # ========= end training and testing part =======


#------------------------------------

predict_data = pd.DataFrame(data=predict_data_1)
predict_data = pd.concat([predict_data, pd.DataFrame(data=predict_data_2)], axis=1)
predict_data = pd.concat([predict_data, pd.DataFrame(data=predict_data_3)], axis=1)
predict_data = pd.concat([predict_data, pd.DataFrame(data=predict_data_4)], axis=1)
predict_data = pd.concat([predict_data, pd.DataFrame(data=predict_data_5)], axis=1)
print(predict_data)
final_results_91 = pd.concat([df_nominal, pd.DataFrame(data=result91)], axis=1)
final_results_92 = pd.concat([df_nominal, pd.DataFrame(data=result92)], axis=1)
final_results_93 = pd.concat([df_nominal, pd.DataFrame(data=result93)], axis=1)
final_results_94 = pd.concat([df_nominal, pd.DataFrame(data=result94)], axis=1)
final_results_95 = pd.concat([df_nominal, pd.DataFrame(data=result95)], axis=1)


predict_data.to_csv('./applications/%s/RNN/result/predict_91-95.csv' % (prefix), index=False)
final_results_91.to_csv('./applications/%s/RNN/result/91_%s_All_wMSE_layers_%s.csv' % (prefix, input_type,output_format), index=False)
final_results_92.to_csv('./applications/%s/RNN/result/92_%s_All_wMSE_layers_%s.csv' % (prefix, input_type,output_format), index=False)
final_results_93.to_csv('./applications/%s/RNN/result/93_%s_All_wMSE_layers_%s.csv' % (prefix, input_type,output_format), index=False)
final_results_94.to_csv('./applications/%s/RNN/result/94_%s_All_wMSE_layers_%s.csv' % (prefix, input_type,output_format), index=False)
final_results_95.to_csv('./applications/%s/RNN/result/95_%s_All_wMSE_layers_%s.csv' % (prefix, input_type,output_format), index=False)
#-------------------------------------
#     print('===== final accuracy =====')
#     train_result_df = pd.DataFrame(data=train_result)
#     result_df = pd.DataFrame(data=result)
#     # print(final_results_df)
#     final_train_results = pd.concat([df_nominal, train_result_df], axis=1)
#     final_results = pd.concat([df_nominal, result_df], axis=1)

#     print(final_results)
#     # final_results.to_csv('./result/all_item_prediction_wMSE.csv', index=False)
# # else:
# #     not_satisify_cluster.append(cluster)


# # final_results = final_results[1:, :]
# # columns = ['cluster_label', 'layer1_correctness', 'layer2_correctness', 'layer3_correctness', 'delay_correctness']
# # final_results_df = pd.DataFrame(data=final_results, columns=columns)
# # print(final_results_df)
# # final_results[['cluster_label']] = final_results[['cluster_label']].astype(int)
# # final_results = final_results.fillna(0)
# # final_results = final_results.astype(int)


# print("======= total accuracy =======")
# print(final_results)

# final_train_results.to_csv('./applications/%s/RNN/result/%s_All_wMSE_layers_train_%s.csv' % (prefix, input_type, output_format), index=False)
# final_results.to_csv('./applications/%s/RNN/result/%s_All_wMSE_layers_%s.csv' % (prefix, input_type,output_format), index=False)

print("======== total not satisify cluster count ===========")
print(not_satisify_cluster)

print("========== End Training =============")
print("End Training Time:",datetime.datetime.now())