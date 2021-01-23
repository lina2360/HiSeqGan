import numpy as np
import pandas as pd
from pandas import ExcelWriter
# import openpyxl
# from pymongo import MongoClient
import argparse
import os

def layers(name):
    source_path = name.replace('-item-seq','')
    # init an array to store every layers 
    layer = []
    max_layer = 1

    # list all files in /output folder
    for file in os.listdir('./applications/%s/GHSOM/output/%s' % (source_path,name)):
        # get all .unit files in /output folder
        if file.endswith('.unit'):
            
            # get file path
            unit_file_path = os.path.join('./applications/%s/GHSOM/output/%s' % (source_path,name), file)
            print(unit_file_path)

            # get attr from content
            text_file = open(unit_file_path).read().split()

            # get each layer dimension info
            x_dim = int(text_file[text_file.index('$XDIM')+1])
            y_dim = int(text_file[text_file.index('$YDIM')+1])
            # print('$XDIM:',text_file[text_file.index('$XDIM')+1])
            # print('$YDIM:',text_file[text_file.index('$YDIM')+1])

            # first layer
            if 'lvl' not in file:
                layer.append(x_dim*y_dim)
            # other layers
            else:
                layer_index = int(file.split('lvl')[1][0])
                max_layer = layer_index if max_layer < layer_index else max_layer
                
                if len(layer) != layer_index:
                    layer.append(x_dim*y_dim)
                else:
                    if (x_dim*y_dim) > layer[layer_index-1]:
                        layer[layer_index-1] = x_dim*y_dim
    print('layer:',layer)
    print('max_layer:',max_layer)
    
    layer_number = layer.copy()
    number_of_digits = [0] * len(layer_number)

    for i in range(len(layer_number)):
        digit = 1
        while True:
            if (layer_number[i] // 10) != 0 :
                digit = digit + 1 
                layer_number[i] = layer_number[i] // 10
            else:
                break
        number_of_digits[i] = digit
    print('number_of_digits:',number_of_digits)
    return layer,max_layer,number_of_digits

# print('wpg-1',layers('wpg-1'))
# print('wpg-2',layers('wpg-2'))
# print('wpg-3',layers('wpg-3'))
# print('wpg-4',layers('wpg-4'))
# print('wpg-5',layers('wpg-5'))