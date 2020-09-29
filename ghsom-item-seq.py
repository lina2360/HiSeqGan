import os
import sys
import argparse
import csv 
from programs.data_processing.format_ghsom_input_vector import format_ghsom_input_vector

parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('--tau1', type=float, default = 0.1)
parser.add_argument('--tau2', type=float, default=0.01)
parser.add_argument('--data', type=str, default=None)
parser.add_argument('--train_column', type=str, default = None)
parser.add_argument('--index', type=str, default=None)
# parser.add_argument('--target', type=str, default = None)
# parser.add_argument('--date_column', type=str, default = None)
args = parser.parse_args()
seq_name = '%s-item-seq'  % args.data

print('tau1 = %s, tau2 = %s' % (args.tau1, args.tau2))
print('data = %s, index = %s' % (seq_name,args.index))
current_path = os.getcwd()
print('Current:',current_path)
##############################
# GHSOM Settings
##############################
# create ghsom input vector file 
#Ref: http://www.ifs.tuwien.ac.at/~andi/somlib/download/SOMLib_Datafiles.html#input_vectors
def create_ghsom_input_file(name, index, train_column):
    try:
        format_ghsom_input_vector(name, index, train_column)
        # os.system('python ./programs/data_processing/format_ghsom_input_vector.py --name=%s --index=%s --train_column=%s' % (name, index, train_column))
        print('Success to create ghsom input file.')
    except Exception as e:
        print('Failed to create ghsom input file.')
        print('Error:',e)

# create ghsom prop file 
# Ref: http://www.ifs.tuwien.ac.at/dm/somtoolbox/examples/PROPERTIES
def create_ghsom_prop_file(name, tau1 = 0.1, tau2 = 0.01, sparseData ='yes', isNormalized = 'false', randomSeed = 7, xSize = 2, ySize = 2, learnRate = 0.7,numIterations = 20000):
    source_path = name.replace('-item-seq','')
    with open('./applications/%s/GHSOM/%s_ghsom.prop' % (source_path,name), 'w', newline='',encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        
        # Parameter settings
        writer.writerow(['workingDirectory=./'])
        writer.writerow(['outputDirectory=./output/%s' % name])
        writer.writerow(['namePrefix=%s' % name])
        writer.writerow(['vectorFileName=./data/%s_ghsom.in' % name])
        writer.writerow(['sparseData=%s' % sparseData])
        writer.writerow(['isNormalized=%s' % isNormalized])
        writer.writerow(['randomSeed=%s' % randomSeed])
        writer.writerow(['xSize=%s' % xSize])
        writer.writerow(['ySize=%s' % ySize])
        writer.writerow(['learnRate=%s' % learnRate])
        writer.writerow(['numIterations=%s' % numIterations])
        writer.writerow(['tau=%s' % tau1])
        writer.writerow(['tau2=%s' % tau2])

# clustering high-dimensional data
def ghsom_clustering(name):
    source_path = name.replace('-item-seq','')
    try:
        print('cmd=','.\programs\GHSOM\somtoolbox.bat GHSOM .\\applications\%s\GHSOM\%s_ghsom.prop -h' % (source_path,name))
        os.system('.\programs\GHSOM\somtoolbox.bat GHSOM .\\applications\%s\GHSOM\%s_ghsom.prop -h' % (source_path,name))
    except Exception as e:
        print('Error:',e)
# extract output data
def extract_ghsom_output(name, current_path):
    source_path = name.replace('-item-seq','')
    print('cmd=','7z e applications\%s\GHSOM\output\%s -o%s\\applications\%s\GHSOM\output\%s' % (source_path,name,current_path,source_path,name))
    os.system('7z e applications\%s\GHSOM\output\%s -o%s\\applications\%s\GHSOM\output\%s' % (source_path,name,current_path,source_path,name))

def save_ghsom_cluster_label(name):
    os.system('python ./programs/data_processing/save_cluster_with_clustered_label_sequence.py --name=%s' % name)
    print('Success transfer cluster label of item sequence data.')

def save_ghsom_cluster_label_with_coordinate(name):
    output = os.popen('python ./programs/data_processing/save_cluster_with_coordinate_representation.py --name=%s' % name)
    logging.info(output)
    output = os.popen('python ./programs/data_processing/GHSOM_center_point.py --name=%s' % name)
    logging.info(output)
    print('Success transfer cluster label into coordinate.')

def format_rnn_input_integer(name):
    # os.system('python ./programs/data_processing/format_RNN_input_sequence_integer.py --name=%s' % (name))
    output = os.popen('python ./programs/data_processing/format_RNN_input_sequence_integer.py --name=%s' % (name)).read()
    logging.info(output)
    print('Success transfer cluster label into integer.')

# def format_RNN_input_float(name,target,date_column):
#     os.system('python ./programs/data_processing/format_RNN_input_float.py --name=%s --target=%s --date_column=%s' % (name,target,date_column))
#     print('Success transfer cluster label into float.')

import datetime
import logging

if os.path.exists('%s/log/%s' % (current_path,args.data)):
    print('Path : %s/log/%s already exist.' % (current_path,args.data) )
else:
    os.makedirs('%s/log/%s' % (current_path,args.data))
path = 'log/%s' % args.data
log_filename = datetime.datetime.now().strftime(path +'/%Y-%m-%d_%H_%M_%S.log')
logging.basicConfig(level=logging.DEBUG,
            format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
            datefmt='%m-%d %H:%M:%S',
            filename=log_filename)
# 定義 handler 輸出 sys.stderr
console = logging.StreamHandler()
console.setLevel(logging.INFO)
# 設定輸出格式
formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
# handler 設定輸出格式
console.setFormatter(formatter)
# 加入 hander 到 root logger
logging.getLogger('').addHandler(console)

# check a new application folder is exist in /applications or not
if os.path.exists('%s/applications/%s/GHSOM/output/%s' % (current_path, args.data, seq_name)):
    print('Already clustering item seqence with ghsom.' )
else:
    print('Creating /applications/%s ....' % seq_name)
    try:
        # # create a new application folder in /applications
        # os.makedirs('%s/applications/%s' % (current_path, args.data))
        # print('Success to create /applications/%s folder.' % (args.data))
        
        # ##############################
        # # data folders settings
        # ##############################
        # # create a folder for data
        # os.makedirs('%s/applications/%s/data' % (current_path, args.data))

        # ##############################
        # # GHSOM folders settings
        # ##############################
        # # create a folder for GHSOM
        # os.makedirs('%s/applications/%s/GHSOM' % (current_path, args.data))

        # # # create a folder for GHSOM prop
        # # os.makedirs('%s/applications/%s/GHSOM/prop' % (current_path, args.data))

        # # create a folder for GHSOM input vector
        # os.makedirs('%s/applications/%s/GHSOM/data' % (current_path, args.data))

        # create a folder for GHSOM output
        os.makedirs('%s/applications/%s/GHSOM/output/%s' % (current_path, args.data,seq_name))

        create_ghsom_input_file(seq_name, args.index, args.train_column)
        create_ghsom_prop_file(seq_name, args.tau1, args.tau2)
        ghsom_clustering(seq_name)
        extract_ghsom_output(seq_name, current_path)
        save_ghsom_cluster_label(args.data)
        # save_ghsom_cluster_label_with_coordinate(seq_name)
        format_rnn_input_integer(args.data)

    except Exception as e:
        print('Failed to create /applications/%s folder due to :%s' % (args.data, str(e)))
# save_ghsom_cluster_label(args.data)
format_rnn_input_integer(args.data)
