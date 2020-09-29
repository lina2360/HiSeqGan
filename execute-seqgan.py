import os
import sys
import argparse
import csv 

parser = argparse.ArgumentParser(description='manual to this script')
# parser.add_argument('--tau1', type=float, default = 0.1)
# parser.add_argument('--tau2', type=float, default=0.01)
parser.add_argument('--data', type=str, default=None)
parser.add_argument('--target', type=str, default = None)
parser.add_argument('--generated_num', type=int, default=960)
parser.add_argument('--total_batch', type=int, default = 100)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--seq_length', type=int, default = 96)

args = parser.parse_args()
seq_name = '%s-item-seq'  % args.data
prefix = args.data
target = args.target
generated_num = args.generated_num
total_batch = args.total_batch
batch_size = args.batch_size
seq_length = args.seq_length
current_path = os.getcwd()
print('Current:',current_path)
##############################
# SeqGAN Settings
##############################
# create SeqGAN input file 
def create_seqgan_input_file(name, target):
    try:
        os.system('python ./programs/data_processing/format_SeqGAN_input.py --name=%s --target=%s' % (name, target))
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
        # generated_num = 3840
        # total_batch = 100
        # batch_size = 96
        # seq_length = 96
        # generated_num = 1
        # total_batch = 1
        # batch_size = 1
        # seq_length = 96
        os.system('python ./programs/SeqGAN/sequence_gan.py --name=%s --generated_num=%s --total_batch=%s --batch_size=%s --seq_length=%s' % (name, generated_num, total_batch, batch_size, seq_length))
        print('Success to generate SeqGAN output file.')
    except Exception as e:
        print('Failed to generate SeqGAN output file.')
        print('Error:',e)

def compute_sequence_predict_accuracy(name):
    try:
        os.system('python ./programs/data_processing/sequance_predict_accuracy.py --name=%s' % (name))
        print('Success to compute SeqGAN accuracy.')
    except Exception as e:
        print('Failed to compute SeqGAN accuracy.')
        print('Error:',e)

def rnn_all_week(name,input_type):
    try:
        # os.system('python ./programs/RNN/app_all.py --name=%s' % (name))
        os.system('python ./programs/RNN/app_all_week.py --name=%s --input_type=%s' % (name,input_type))
        # logging.info(output)
    except Exception as e:
        print('Error:',e)

# check a new application folder is exist in /applications or not
if os.path.exists('%s/applications/%s/SeqGAN' % (current_path, args.data)):
    print('Warning : /applications/%s/SeqGAN is exist.' % args.data)
else:
    print('Creating /applications/%s/SeqGAN ....' % args.data)

    try:
        # create a new application folder in /applications
        os.makedirs('%s/applications/%s/SeqGAN' % (current_path, args.data))
        print('Success to create /applications/%s/SeqGAN folder.' % (args.data))

        # create a new application folder in /applications
        os.makedirs('%s/applications/%s/SeqGAN/save' % (current_path, args.data))
        print('Success to create /applications/%s/SeqGAN/save folder.' % (args.data))

        # create_seqgan_input_file(args.data, args.target)
        # execute_sequence_gan(args.data)
        # execute_sequence_gan(args.data)
    except Exception as e:
        print('Failed to create /applications/%s/SeqGAN folder due to :%s' % (args.data, str(e)))
        
create_seqgan_input_file(args.data, args.target)
execute_sequence_gan(args.data,generated_num,total_batch,batch_size,seq_length)
# compute_sequence_predict_accuracy(args.data)

# rnn_all_week(args.data,'rnn')