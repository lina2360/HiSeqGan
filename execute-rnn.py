import os
import sys
import argparse
import csv 
import datetime
import logging

parser = argparse.ArgumentParser(description='manual to this script')
# parser.add_argument('--tau1', type=float, default = 0.1)
# parser.add_argument('--tau2', type=float, default=0.01)
parser.add_argument('--data', type=str, default=None)
parser.add_argument('--target', type=str, default = None)
parser.add_argument('--generated_num', type=int, default=960)
parser.add_argument('--total_batch', type=int, default = 100)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--seq_length', type=int, default = 96)
# parser.add_argument('--input_type', type=int, default = 96)


args = parser.parse_args()
prefix = args.data
target = args.target
generated_num = args.generated_num
total_batch = args.total_batch
batch_size = args.batch_size
seq_length = args.seq_length
# input_type = args.input_type
current_path = os.getcwd()
print('Current:',current_path)


if os.path.exists('%s/log/%s' % (current_path,prefix)):
    print('Path : %s/log/%s already exist.' % (current_path,prefix) )
else:
    os.makedirs('%s/log/%s' % (current_path,prefix))
path = 'log/%s' % prefix
log_filename = datetime.datetime.now().strftime(path +'/rnn-%Y-%m-%d_%H_%M_%S.log')
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

##############################
# RNN Settings
##############################
# Run RNN 
def rnn_all_wmse(name,input_type):
    try:
        # os.system('python ./programs/RNN/app_all.py --name=%s' % (name))
        os.system('python ./programs/RNN/app_all.py --name=%s --input_type=%s' % (name,input_type))
        # logging.info(output)
    except Exception as e:
        print('Error:',e)

def rnn_all_center(name,input_type):
    try:
        # os.system('python ./programs/RNN/app_all.py --name=%s' % (name))
        os.system('python ./programs/RNN/app_all_item_center_point_real.py --name=%s --input_type=%s' % (name,input_type))
        # logging.info(output)
    except Exception as e:
        print('Error:',e)

def rnn_all_week(name,input_type):
    try:
        # os.system('python ./programs/RNN/app_all.py --name=%s' % (name))
        os.system('python ./programs/RNN/app_all_week.py --name=%s --input_type=%s' % (name,input_type))
        # logging.info(output)
    except Exception as e:
        print('Error:',e)


# create SeqGAN input file 
def create_seqgan_input_file(name, target):
    try:
        os.system('python ./programs/data_processing/format_SeqGAN_input.py --name=%s --target=%s' % (name, target))
        # logging.info(output)
        print('Success to SeqGAN ghsom input file.')
    except Exception as e:
        print('Failed to create SeqGAN input file.')
        print('Error:',e)

def execute_sequence_gan(name,generated_num,total_batch,batch_size,seq_length):
    try:
        # generated_num = input('generated_num=')
        # total_batch = input('total_batch=')
        # batch_size = input('batch_size=')
        # seq_length = input('seq_length=')
        print('python ./programs/SeqGAN/sequence_gan.py --name=%s --generated_num=%s --total_batch=%s --batch_size=%s --seq_length=%s' % (name, generated_num, total_batch, batch_size, seq_length))
        os.system('python ./programs/SeqGAN/sequence_gan.py --name=%s --generated_num=%s --total_batch=%s --batch_size=%s --seq_length=%s' % (name, generated_num, total_batch, batch_size, seq_length))
        # logging.info(output)
        print('Success to generate SeqGAN output file.')
    except Exception as e:
        print('Failed to generate SeqGAN output file.')
        print('Error:',e)

def clustering_sequence_gan_output(name):
    try:
        os.system('python ./programs/data_processing/sequence_clustering.py --name=%s' % (name))
        # logging.info(output)
        print('Success to clustering SeqGAN output file.')
    except Exception as e:
        print('Failed to clustering SeqGAN output file.')
        print('Error:',e)


# check a new application folder is exist in /applications or not
if os.path.exists('%s/applications/%s/RNN' % (current_path, prefix)):
    print('Warning : /applications/%s/RNN is exist.' % prefix)
else:
    print('Creating /applications/%s/RNN ....' % prefix)

    try:
        # create a new application folder in /applications
        os.makedirs('%s/applications/%s/RNN' % (current_path, prefix))
        print('Success to create /applications/%s/RNN folder.' % (prefix))

        # create a new application folder in /applications
        os.makedirs('%s/applications/%s/RNN/result' % (current_path, prefix))
        print('Success to create /applications/%s/RNN/result folder.' % (prefix))

        # create a new application folder in /applications
        os.makedirs('%s/applications/%s/RNN/model' % (current_path, prefix))
        print('Success to create /applications/%s/RNN/model folder.' % (prefix))

        # rnn_all_wmse(prefix)
        # rnn_all_center(prefix)
    except Exception as e:
        print('Failed to create /applications/%s/RNN folder due to :%s' % (args.data, str(e)))

if os.path.exists('%s/applications/%s/SeqGAN' % (current_path, args.data)):
    print('Warning : /applications/%s/SeqGAN is exist.' % args.data)
else:
    print('Creating /applications/%s/SeqGAN ....' % args.data)

    # create a new application folder in /applications
    os.makedirs('%s/applications/%s/SeqGAN' % (current_path, args.data))
    print('Success to create /applications/%s/SeqGAN folder.' % (args.data))

    # create a new application folder in /applications
    os.makedirs('%s/applications/%s/SeqGAN/save' % (current_path, args.data))
    print('Success to create /applications/%s/SeqGAN/save folder.' % (args.data))


# create_seqgan_input_file(prefix, target)
execute_sequence_gan(prefix,generated_num,total_batch,batch_size,seq_length)
# clustering_sequence_gan_output(prefix)

# rnn_all_wmse(prefix,'rnn')
# rnn_all_center(prefix,'rnn')
# rnn_all_wmse(prefix,'seqgan')
# rnn_all_center(prefix,'seqgan')

# rnn_all_week(prefix,'rnn')
# rnn_all_week(prefix,'seqgan')