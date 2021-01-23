import numpy as np
import pandas as pd
from pandas import ExcelWriter
# import openpyxl
# from pymongo import MongoClient
import argparse
import get_ghsom_dim

parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('--name', type=str, default = None)
parser.add_argument('--target', type=str, default = None)
parser.add_argument('--date_column', type=str, default = None)
# parser.add_argument('--index', type=str, default = None)
args = parser.parse_args()

prefix = args.name
date_column = args.date_column

layers,max_layer,number_of_digits = get_ghsom_dim.layers(prefix)
layer = sum(number_of_digits)

# read source file to get data attribute
target = args.target
df = pd.read_csv('./applications/%s/data/%s_with_clustered_label.csv' % (prefix,prefix), low_memory=False)
df['clustered_label'] =  round(df['clustered_label'],layer)
print(df)
df_source = df[[date_column, target, 'clustered_label']]
# df_source[date_column] = pd.to_datetime(df_source[date_column])
# df_source[date_column] = df_source[date_column].dt.date
df_source = df_source.dropna()

# #get unuique item name 
# dates = df_source['ReportDate'].unique()

# sequence_df = pd.DataFrame()
# for date in dates:
#     df = df_source[df_source['ReportDate'] == date]
#     df = df.rename(columns={'clustered_label': date})
#     df = df.drop(['ReportDate'], axis=1)
#     df = df.reset_index(drop=True)
#     sequence_df = pd.concat([df, sequence_df])


targets = df_source[target].unique()

result = pd.DataFrame()

for target_name in targets:
    df = df_source[df_source[target] == target_name]
    
    dates = df[date_column].unique()
    sequence = pd.DataFrame()

    for date in dates:
        df_sequence = df[df[date_column] == date]
        df_sequence = df_sequence.drop([target], axis=1)
        df_sequence.rename(columns={'clustered_label': date}, inplace=True)
        df_sequence = df_sequence.drop(date_column, axis=1)
        df_sequence = df_sequence.reset_index(drop=True)
        sequence = pd.concat([df_sequence, sequence], axis=1)
    
    # sort column by date
    sequence = sequence.reindex(sorted(sequence.columns), axis=1)
    sequence.insert(loc=0, column=target, value=target_name)
    sequence = sequence.fillna(-1)

    result = pd.concat([sequence, result], axis=0, ignore_index=True)
    # print(result)

result = result.fillna(-1)
# result = result.reindex(sorted(result.columns), axis=1)
result = result.replace(-1, 0)
print(result.columns)

result.to_csv('./applications/%s/data/tmp_float.csv' % (prefix), index=False)

df = pd.read_csv('./applications/%s/data/tmp_float.csv' % (prefix), low_memory=False)
print(df.columns)
df = df.reindex(sorted(df.columns), axis=1)    
#df = df[['id','week1', 'week2', 'week3','week4','week5', 'week6', 'week7', 'week8', 'week9', 'week10', 'week11', 'week12',  'week13',  'week14', 'week15', 'week16', 'week17','week18','week19','week20','week21','week22','week23','week24','week25','week26','week27','week28','week29','week30','week31','week32','week33','week34','week35','week36'  ]]
df = df[['id','week1', 'week2', 'week3','week4','week5', 'week6', 'week7', 'week8', 'week9', 'week10', 'week11', 'week12',  'week13',  'week14', 'week15', 'week16', 'week17','week18'  ]]
df.to_csv('./applications/%s/data/rnn_input_data_float.csv'  % (prefix), index=False)

# reset ItemShortName index from 0 to shape[1]-1
# df[target] = df.index
df.to_csv('./raw-data/%s-item-seq.csv'  % (prefix), index=False)