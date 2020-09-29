import numpy as np
import tensorflow as tf
import random
from dataloader import Gen_Data_loader, Dis_dataloader
from generator import Generator
from discriminator import Discriminator
from rollout import ROLLOUT
from target_lstm import TARGET_LSTM
import pickle 
import pandas as pd
import argparse
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

#import codecs
parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('--name', type=str, default = None)
parser.add_argument('--generated_num', type=int, default = 960)
parser.add_argument('--total_batch', type=int, default = 100)
parser.add_argument('--batch_size', type=int, default = 100)
parser.add_argument('--seq_length', type=int, default = 100)

args = parser.parse_args()
prefix = args.name
generated_num = args.generated_num
total_batch = args.total_batch
batch_size = args.batch_size
seq_length = args.seq_length

#########################################################################################
#  Generator  Hyper-parameters
######################################################################################
EMB_DIM = 32 # embedding dimension
HIDDEN_DIM = 32 # hidden state dimension of lstm cell
SEQ_LENGTH = seq_length # sequence length
START_TOKEN = 0
PRE_EPOCH_NUM = 120 # supervise (maximum likelihood estimation) epochs
# PRE_EPOCH_NUM = 10
SEED = 88
BATCH_SIZE = batch_size
# BATCH_SIZE = 2

#########################################################################################
#  Discriminator  Hyper-parameters
#########################################################################################
dis_embedding_dim = 64
dis_filter_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 18]
dis_num_filters = [100, 200, 200, 200, 200, 100, 100, 100, 100, 100, 160, 160]
dis_dropout_keep_prob = 0.75
dis_l2_reg_lambda = 0.2
dis_batch_size = 64
# dis_batch_size = 2

#########################################################################################
#  Basic Training Parameters
#########################################################################################
TOTAL_BATCH = total_batch
# TOTAL_BATCH = 3
# positive_file = 'save/real_data.txt'
positive_file = './applications/%s/SeqGAN/seqGAN_input_data.txt' % prefix
negative_file = './applications/%s/SeqGAN/save/generator_sample.txt' % prefix
# eval_file = 'save/eval_file.txt'
eval_file = './applications/%s/SeqGAN/seqGAN_output_data.txt' % prefix
# generated_num = 10
generated_num = generated_num


def generate_samples(sess, trainable_model, batch_size, generated_num, output_file):
    # Generate Samples
    generated_samples = []
    for _ in range(int(generated_num / batch_size)):
        generated_samples.extend(trainable_model.generate(sess))
        # print('============================generated_samples============================')
        # print(generated_samples)
        # print('=========================================================================')
    with open(output_file, 'w') as fout:
        for poem in generated_samples:
            buffer = ' '.join([str(x) for x in poem]) + '\n'
            fout.write(buffer)


# def target_loss(sess, target_lstm, data_loader):
#     # target_loss means the oracle negative log-likelihood tested with the oracle model "target_lstm"
#     # For more details, please see the Section 4 in https://arxiv.org/abs/1609.05473
#     nll = []
#     data_loader.reset_pointer()

#     for it in range(data_loader.num_batch):
#         batch = data_loader.next_batch()
#         g_loss = sess.run(target_lstm.pretrain_loss, {target_lstm.x: batch})
#         nll.append(g_loss)

#     return np.mean(nll)


def pre_train_epoch(sess, trainable_model, data_loader):
    # Pre-train the generator using MLE for one epoch
    supervised_g_losses = []
    data_loader.reset_pointer()

    for it in range(data_loader.num_batch):
        batch = data_loader.next_batch()
        _, g_loss = trainable_model.pretrain_step(sess, batch)
        supervised_g_losses.append(g_loss)

    return np.mean(supervised_g_losses)


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    assert START_TOKEN == 0

    gen_data_loader = Gen_Data_loader(BATCH_SIZE)
    likelihood_data_loader = Gen_Data_loader(BATCH_SIZE) # For testing
    # for get maximium value
    df = pd.read_csv(positive_file, sep=' ',header=None)
    max_values_of_columns = df.max()
    print('max cluster:',int(max(max_values_of_columns[0:df.shape[1]])))
    vocab_size = int(max(max_values_of_columns[0:df.shape[1]]))+1
    dis_data_loader = Dis_dataloader(BATCH_SIZE)

    generator = Generator(vocab_size, BATCH_SIZE, EMB_DIM, HIDDEN_DIM, SEQ_LENGTH, START_TOKEN)
    generator_last = Generator(vocab_size, BATCH_SIZE, EMB_DIM, HIDDEN_DIM, seq_length, START_TOKEN)
    # ~~op = './save/target_params_py3.pkl'
    #with open(op, 'rb') as f:
    #   target_params = pickle.load(f)
    # ~~target_params = pickle.load(open(op, 'rb'), encoding = 'bytes')

    # print("------*******************", np.array(target_params).shape)    
    # print("------*******************", target_params )    
    # ~~target_lstm = TARGET_LSTM(vocab_size, BATCH_SIZE, EMB_DIM, HIDDEN_DIM, SEQ_LENGTH, START_TOKEN, target_params) # The oracle model

    discriminator = Discriminator(sequence_length=SEQ_LENGTH, num_classes=2, vocab_size=vocab_size, embedding_size=dis_embedding_dim, 
                                filter_sizes=dis_filter_sizes, num_filters=dis_num_filters, l2_reg_lambda=dis_l2_reg_lambda)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    # First, use the oracle model to provide the positive examples, which are sampled from the oracle data distribution
    # generate_samples(sess, target_lstm, BATCH_SIZE, generated_num, positive_file)
    gen_data_loader.create_batches(positive_file)
    
    #
    log = open('./applications/%s/SeqGAN/save/experiment-log.txt' % prefix, 'w')
    #  pre-train generator
    print ('Start pre-training...')
    log.write('pre-training...\n')
    for epoch in range(PRE_EPOCH_NUM):
        loss = pre_train_epoch(sess, generator, gen_data_loader)
        if epoch % 5 == 0:
            generate_samples(sess, generator, BATCH_SIZE, generated_num, eval_file)
            likelihood_data_loader.create_batches(eval_file)
            # test_loss = target_loss(sess, target_lstm, likelihood_data_loader)
            print('pre-train epoch ', epoch, 'test_loss ', loss)
            buffer = 'epoch:\t'+ str(epoch) + '\tnll:\t' + str(loss) + '\n'
            log.write(buffer)

    print('Start pre-training discriminator...')
    # Train 3 epoch on the generated data and do this for 50 times
    for _ in range(50):
        generate_samples(sess, generator, BATCH_SIZE, generated_num, negative_file)
        dis_data_loader.load_train_data(positive_file, negative_file)
        for _ in range(3):
            dis_data_loader.reset_pointer()
            for it in range(dis_data_loader.num_batch):
                x_batch, y_batch = dis_data_loader.next_batch()
                # print("x_batch:",x_batch.shape)
                # print("y_batch:",y_batch.shape)
                feed = {
                    discriminator.input_x: x_batch,
                    discriminator.input_y: y_batch,
                    discriminator.dropout_keep_prob: dis_dropout_keep_prob
                }
                # print('x_batch:',x_batch)
                # print('y_batch:',y_batch)
                _ = sess.run(discriminator.train_op, feed)
    
    rollout = ROLLOUT(generator, 0.8)

    print('#########################################################################')
    print('Start Adversarial Training...')
    log.write('adversarial training...\n')
    for total_batch in range(TOTAL_BATCH):
        # Train the generator for one step
        # for it in range(1):
        #     samples = generator.generate(sess)
        #     rewards = rollout.get_reward(sess, samples, 16, discriminator)
        #     feed = {generator.x: samples, generator.rewards: rewards}
        #     print(feed)
        #     # _ = sess.run(generator.g_updates, feed_dict=feed)
        #     # _ = sess.run(generator.g_updates)

        # Test
        if total_batch % 5 == 0:
            samples = generator.generate(sess)
            rewards = rollout.get_reward(sess, samples, 16, discriminator)
            generate_samples(sess, generator, BATCH_SIZE, generated_num, eval_file)
            likelihood_data_loader.create_batches(eval_file)
            buffer = 'epoch:\t' + str(total_batch) + '\trewards:\t' + str(rewards) + '\n'
            print('total_batch: ', total_batch)
            log.write(buffer)

        # if total_batch != TOTAL_BATCH - 1:
        #     samples = generator.generate(sess)
        #     rewards = rollout.get_reward(sess, samples, 16, discriminator)
        #     generate_samples(sess, generator, BATCH_SIZE, generated_num, eval_file)
        #     likelihood_data_loader.create_batches(eval_file)
        #     buffer = 'epoch:\t' + str(total_batch) + '\trewards:\t' + str(rewards) + '\n'
        #     print('total_batch: ', total_batch)
        #     log.write(buffer)

        elif total_batch == TOTAL_BATCH - 1:
            samples = generator.generate(sess)
            print('samples:',samples)
            rewards = rollout.get_reward(sess, samples, 16, discriminator)
            generate_samples(sess, generator, BATCH_SIZE, generated_num, eval_file)
            likelihood_data_loader.create_batches(eval_file)
            buffer = 'epoch:\t' + str(total_batch) + '\trewards:\t' + str(rewards) + '\n'
            print('total_batch: ', total_batch)
            print('reward:', rewards)
            
            log.write(buffer)

            # test_loss = target_loss(sess, target_lstm, likelihood_data_loader)
            # buffer = 'epoch:\t' + str(total_batch) + '\tnll:\t' + str(test_loss) + '\n'
            # print('total_batch: ', total_batch, 'test_loss: ', test_loss)
            # log.write(buffer)

        # Update roll-out parameters
        rollout.update_params()

        # Train the discriminator
        for _ in range(5):
            generate_samples(sess, generator, BATCH_SIZE, generated_num, negative_file)
            dis_data_loader.load_train_data(positive_file, negative_file)

            for _ in range(3):
                dis_data_loader.reset_pointer()
                for it in range(dis_data_loader.num_batch):
                    x_batch, y_batch = dis_data_loader.next_batch()
                    feed = {
                        discriminator.input_x: x_batch,
                        discriminator.input_y: y_batch,
                        discriminator.dropout_keep_prob: dis_dropout_keep_prob
                    }
                    # print('y_batch:',y_batch)
                    _ = sess.run(discriminator.train_op, feed)

        saver = tf.train.Saver()
        save_path = saver.save(sess, "/tmp/model.ckpt")
        # tf.python.tools.inspect_checkpoint.print_tensors_in_checkpoint_file(file_name='./tmp/model.ckpt',tensor_name='',all_tensors='')
            
    log.close()


if __name__ == '__main__':
    main()
