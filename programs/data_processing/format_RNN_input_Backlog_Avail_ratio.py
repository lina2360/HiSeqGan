import numpy as np
import pandas as pd
from pandas import ExcelWriter
import openpyxl
from pymongo import MongoClient
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
df_source = df[[date_column, target, 'Backlog_Avail_ratio']]
# df_source['AWU_Avial_ratio'] = df_source['AWU_Avial_ratio']*pow(10,layer)
# df_source['clustered_label'] = df_source['clustered_label'].astype('int64')
# df['clustered_label'] = df['clustered_label']*pow(10,layer)
# df['clustered_label'] = df['clustered_label'].astype('int64')
# df.to_csv('./applications/%s/data/%s_with_clustered_label_integer.csv' % (prefix,prefix),index=False)
df_source[date_column] = pd.to_datetime(df_source[date_column])
df_source[date_column] = df_source[date_column].dt.date
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
        df_sequence.rename(columns={'Backlog_Avail_ratio': date}, inplace=True)
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

result.to_csv('./applications/%s/data/tmp_integer_Backlog_Avail_ratio.csv' % (prefix), index=False)

df = pd.read_csv('./applications/%s/data/tmp_integer_Backlog_Avail_ratio.csv' % (prefix), low_memory=False)
print(df.columns)
df = df.reindex(sorted(df.columns), axis=1)
# df = df[['id','week1', 'week2', 'week3','week4','week5', 'week6', 'week7', 'week8', 'week9', 'week10', 'week11', 'week12',  'week13',  'week14', 'week15', 'week16', 'week17','week18'  ]]
# float to int 
columns = df.columns.tolist()
columns.remove(target)
# for column in columns:
#     df[column] = df[column].astype('int64')
    

df.to_csv('./applications/%s/data/rnn_input_data_Backlog_Avail_ratio.csv'  % (prefix), index=False)
