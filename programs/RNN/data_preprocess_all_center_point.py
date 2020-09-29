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

def format_row_to_layerformat(df_input, base_length, prefix, layer,number_of_digits):

    base_shape = np.zeros((len(number_of_digits)+3)*base_length).reshape(len(number_of_digits)+3, -1)
    input_sequence = np.zeros((len(number_of_digits)+3, base_length))
    # =====
    # Below training all input data with separate levels number and zero 
    # =====
    # correct_layer = [0, 0, 0, 0]
    total_number = 0
    not_satisified_sequence_count = 0
    max_sequence_length = 0
    min_sequence_length = 1000000
    sequences_length = np.empty((len(number_of_digits)+3,0), float)
    check_sequence_length_enough_ornot = True

    for index in range(df_input.shape[0]):
        row = df_input.iloc[[index]]
        zero_count = 0
        single_cluster = np.empty((len(number_of_digits)+3,0), float)


        for column in row:
            # encoding delay zero count
            if row[column].iloc[0] == 0:
                zero_count += 1
            else:
                cluster_number = row[column].iloc[0]
                df_center_point = pd.read_csv('./applications/%s/data/merge_point_cluster_data.csv' % prefix)
                # print("cluster:",str(cluster_number))
                # print("cluster df:",df_center_point.clustered_label.astype(str).str[0:2])
                point_x = df_center_point[df_center_point.clustered_label == cluster_number].head(1).point_x.values[0]
                point_y = df_center_point[df_center_point.clustered_label == cluster_number].head(1).point_y.values[0]
                # print("point x: {0} , point y: {1} ".format( point_x, point_y))
                # 4/8 : 4-> molecule, 8 -> denominator
                point_x_molecule = 0
                point_x_denominator = 0
                point_y_molecule = 0
                point_y_denominator = 0
                if point_x != '1':
                    point_x_molecule = int(str(point_x).split('/')[0])
                    point_x_denominator = int(str(point_x).split('/')[1])
                else:
                    point_x_molecule = 1
                    point_x_denominator = 1
                
                if point_y != '1':
                    point_y_molecule = int(str(point_y).split('/')[0])
                    point_y_denominator = int(str(point_y).split('/')[1])
                else:
                    point_y_molecule = 1
                    point_y_denominator = 1

                print('x: %s/%s, y: %s/%s' % (point_x_molecule,point_x_denominator,point_y_molecule,point_y_denominator))
                # # ===== without decimal=====
                # first_layer = str(cluster_number)[:1]
                # second_layer = str(cluster_number)[1:2]
                # if str(cluster_number)[2:3] == '':
                #     third_layer = 0
                # else:
                #     third_layer = str(cluster_number)[2:3]
                # ==================
                # ---with decimal---
                # ==================
                # split_num = str(cluster_number).split('.')
                # decimal_part = int(split_num[1])

                # first_layer = str(decimal_part)[:1]
                # second_layer = str(decimal_part)[1:2]
                # if str(decimal_part)[2:3] == '':
                #     third_layer = 0
                # else:
                #     third_layer = str(decimal_part)[2:3]
                cluster_number_list = getClusterList(cluster_number,number_of_digits)
                # print('cluster_number_list:',cluster_number_list)
                cluster_number_list.append(zero_count)
                cluster_number_list.append(point_x_molecule/point_x_denominator)
                cluster_number_list.append(point_y_molecule/point_y_denominator)

                current_cluster = np.expand_dims(cluster_number_list, axis=1)
                # current_cluster = np.expand_dims(np.asarray([int(first_layer), int(second_layer), int(third_layer), zero_count, point_x_molecule/point_x_denominator, point_y_molecule/point_y_denominator]), axis=1)
                single_cluster = np.append(single_cluster, current_cluster, axis=1)
                # print(current_cluster)
                zero_count = 0
        # print('===== raw input =====')
        # print(single_cluster)
        # input_row = pd.DataFrame(data=single_cluster)
        # get the maxinum length of sequence

        if single_cluster.shape[1] >= len(number_of_digits):
            # get max sequence length
            if single_cluster.shape[1] > max_sequence_length:
                max_sequence_length = single_cluster.shape[1]

            # get min sequence length
            if single_cluster.shape[1] < min_sequence_length:
                min_sequence_length = single_cluster.shape[1]
            
            single_length = single_cluster.shape[1]
            single_length_array = np.empty(len(number_of_digits)+3)
            single_length_array.fill(single_length)

            # sequences_length = np.append(sequences_length,  single_length_array).astype(int)
            # result = np.hstack((single_cluster, base_shape))
            result = np.hstack((base_shape, single_cluster))
            result = result[:, single_cluster.shape[1]:].astype(float)
            result = result[:, 0: base_length].astype(float)

            total_number+=1
            
            input_sequence = np.vstack((input_sequence, result))


        else:
            not_satisified_sequence_count += 1
            # print(not_satisified_sequence_count)
            pass

    if not_satisified_sequence_count > df_input.shape[0]:
    # if not_satisified_sequence_count > total_number:
        check_sequence_length_enough_ornot = False



    # delete first layer (4 rows) of redundent zero value and reshape, one input one layer
    input_sequence = input_sequence.astype(float)
    input_sequence = input_sequence[:, base_length-max_sequence_length:]
    # input_sequence = input_sequence[:, 0:max_sequence_length]
    input_sequence = np.delete(input_sequence, list(range(0, len(number_of_digits)+3)), 0)
    # input_sequence = input_sequence.transpose()
    # print(input_sequence)
    # print(input_sequence.shape)
    # input_sequence = input_sequence.reshape(total_number, 4, -1)
    input_sequence = pd.DataFrame(data=input_sequence)
    # print(input_sequence)
    print('not_satisified_sequence_count:',not_satisified_sequence_count)

    return check_sequence_length_enough_ornot, input_sequence, max_sequence_length, min_sequence_length