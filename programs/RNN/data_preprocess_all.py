import numpy as np
import pandas as pd
import random
import math
from random import sample
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def getClusterList(cluster,layer):
    cluster_list = []
    layer.reverse()
    # print(layer)
    digits = 0
    for i in layer:
        digits = i
        # print(digits)
        if cluster < 10:
            cluster_list.append(cluster)
            break
        else:
            value = cluster%(pow(10,digits))
            cluster_list.append(int(value))
#             if(cluster>1):
#                 cluster = int(cluster/10)
#             else:
#                 break
        cluster = cluster // pow(10,digits)
    
    cluster_list.reverse()
    return cluster_list


def format_row_to_layerformat(df_input, base_length, layer,number_of_digits,df_nominal):

    base_shape = np.zeros((len(number_of_digits))*base_length).reshape(len(number_of_digits), -1)
    input_sequence = np.zeros((len(number_of_digits), base_length))
    # =====
    # Below training all input data with separate levels number and zero 
    # =====
    # correct_layer = [0, 0, 0, 0]
    total_number = 0
    not_satisified_sequence_count = 0
    max_sequence_length = 0
    min_sequence_length = 1000000
    sequences_length = np.empty((len(number_of_digits),0), int)
    check_sequence_length_enough_ornot = True
    # print('layer:',len(number_of_digits))
    remove = []
    for index in range(df_input.shape[0]):
        row = df_input.iloc[[index]]
        zero_count = 0
        single_cluster = np.empty((len(number_of_digits),0), int)
        # print('row:',row)
        # print('single_cluster:',single_cluster)
        for column in row:
            # encoding delay zero count
            if row[column].iloc[0] < pow(10,layer-1) :
                zero_count += 1
            else:
                cluster_number = row[column].iloc[0]

                # # ===== without decimal=====
                # first_layer = str(cluster_number)[:1]
                # second_layer = str(cluster_number)[1:2]
                # if str(cluster_number)[2:3] == '':
                #     third_layer = 0
                # else:
                #     third_layer = str(cluster_number)[2:3]
                # # ==================
                # # ---with decimal---
                # # ==================
                # # split_num = str(cluster_number).split('.')
                # # decimal_part = int(split_num[1])

                # # first_layer = str(decimal_part)[:1]
                # # second_layer = str(decimal_part)[1:2]
                # # if str(decimal_part)[2:3] == '':
                # #     third_layer = 0
                # # else:
                # #     third_layer = str(decimal_part)[2:3]
                cluster_number_list = getClusterList(cluster_number,number_of_digits)
                # print('cluster_number_list:',cluster_number_list)
                # cluster_number_list.append(zero_count)
                
                current_cluster = np.expand_dims(cluster_number_list, axis=1)
                # current_cluster = np.expand_dims(np.asarray([int(first_layer), int(second_layer), int(third_layer), zero_count]), axis=1)
                # print('current_cluster:',current_cluster)
                # print('single_cluster:',single_cluster)
                single_cluster = np.append(single_cluster, current_cluster, axis=1)
                # print(single_cluster)
                zero_count = 0
        # print('===== raw input =====')
        # print(single_cluster)
        # input_row = pd.DataFrame(data=single_cluster)
        # get the maxinum length of sequence
        
       
        # if single_cluster.shape[1] >= len(number_of_digits):
        if single_cluster.shape[1] >= 0:
            # get max sequence length
            if single_cluster.shape[1] > max_sequence_length:
                max_sequence_length = single_cluster.shape[1]

            # get min sequence length
            if single_cluster.shape[1] < min_sequence_length:
                min_sequence_length = single_cluster.shape[1]
            
            single_length = single_cluster.shape[1]
            single_length_array = np.empty(len(number_of_digits)+1)
            single_length_array.fill(single_length)

            # sequences_length = np.append(sequences_length,  single_length_array).astype(int)
            # result = np.hstack((single_cluster, base_shape))
            result = np.hstack((base_shape, single_cluster))
            result = result[:, single_cluster.shape[1]:].astype(int)
            result = result[:, 0: base_length].astype(int)

            total_number+=1
            print(input_sequence.shape)
            print(result.shape)
            input_sequence = np.vstack((input_sequence, result))
            print(input_sequence)

        else:
            remove.append(index)
            not_satisified_sequence_count += 1
            # print(not_satisified_sequence_count)
            pass

    if not_satisified_sequence_count > df_input.shape[0]:
    # if not_satisified_sequence_count > total_number:
        check_sequence_length_enough_ornot = False



    # delete first layer (4 rows) of redundent zero value and reshape, one input one layer
    input_sequence = input_sequence.astype(int)
    input_sequence = input_sequence[:, base_length-max_sequence_length:]
    # input_sequence = input_sequence[:, 0:max_sequence_length]
    input_sequence = np.delete(input_sequence, list(range(0, len(number_of_digits))), 0)
    # input_sequence = input_sequence.transpose()
    # print(input_sequence)
    # print(input_sequence.shape)
    # input_sequence = input_sequence.reshape(total_number, 4, -1)
    input_sequence = pd.DataFrame(data=input_sequence)
    print('not_satisified_sequence_count:',not_satisified_sequence_count)

    df_nominal = df_nominal.drop(remove, axis=0)
    print('df_nominal:',df_nominal)

    return check_sequence_length_enough_ornot, input_sequence, max_sequence_length, min_sequence_length, df_nominal