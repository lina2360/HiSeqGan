import numpy as np
import pandas as pd
import tensorflow as tf
import random
import math
from random import sample
from random import randint
import datetime
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class WPG_RNN(object):

    def __init__(self, df, df_test, max_sequence_length, min_sequence_length, num_neurons, learning_rate, max_num_train_iterations, accept_loss, batch_size, training_start_point, layer, prefix,output_format):
        # data frame
        self._df = df
        # test data frame
        self._df_test = df_test
        # number of time step
        self._num_time_steps = df.shape[1]-2
        # self._num_time_steps = min_sequence_length - 2
        # Just one feature, the time series
        self._num_inputs = df.shape[0]
        # 100 neuron layer, play with this
        self._num_neurons = num_neurons
        # Just one output, predicted time series
        self._num_outputs = df.shape[0]
        # learning rate, 0.0001 default, but you can play with this
        self._learning_rate = learning_rate
        # how many iterations to go through (training steps), you can play with this
        self._num_train_iterations = max_num_train_iterations
        # stop unitl loss below some value
        self._accept_loss = accept_loss
        # Size of the batch of data
        self._batch_size = batch_size
        # training input start point
        self._training_start_point = training_start_point
        # rnn input ghsom layer
        self._layer = layer
        # second start point
        self._training_start_point2 = max_sequence_length - min_sequence_length
        # testing start point
        self._testing_start_point = training_start_point + 1
        # window size
        self._frame_size = min_sequence_length - 2
        # random startpoint
        self._rand_start_point = randint(0, 9)
        # totalrow
        self._total_input_rows = df.shape[0]
        # layer1 row index
        self._l1_index = np.arange(0, df.shape[0], layer ) # 4 = layer + 1
        # layer2 row index
        self._l2_index = np.arange(1, df.shape[0], layer )
        # layer3 row index
        self._l3_index = np.arange(2, df.shape[0], layer )
        # layer4 row index
        # self._zero_index = np.arange(3, df.shape[0], layer )     
        self._prefix = prefix
        self._output_format = output_format
        ##INITIALIZE GRAPH
        self._graph = tf.Graph()

        ##POPULATE GRAPH WITH NECESSARY COMPONENTS
        with self._graph.as_default():

            self._X = tf.placeholder(tf.float32, [None,  self._num_inputs, self._num_time_steps])
            self._y = tf.placeholder(tf.float32, [None, self._num_outputs, self._num_time_steps])

            # rnn.BasicRNNCell
            # rnn.BasicLSTMCell
            # rnn.GRUCell
            cell = tf.contrib.rnn.OutputProjectionWrapper(
                tf.contrib.rnn.BasicRNNCell(num_units=self._num_neurons, activation=tf.nn.relu),
                output_size=self._num_time_steps)

            self._outputs, states = tf.nn.dynamic_rnn(cell, self._X, dtype=tf.float32)
            # self._outputs, states = tf.nn.dynamic_rnn(cell, self._X, dtype=tf.float32, sequence_length=tf.constant([max_sequence_length]))

            #------------------------
            # loss function
            # change this to fit model
            #------------------------

            loss_value = 0
            for i in range(0,self._layer+1):
                print(i)
                index = np.arange(i, df.shape[0], self._layer+1)
                layer_diff = (tf.gather(tf.gather(self._outputs, 0), index) - tf.gather(tf.gather(self._y, 0), index))
                loss_value = loss_value + tf.reduce_mean(tf.square(layer_diff))*pow(10,self._layer-i)

            l1_diff = (tf.gather(tf.gather(self._outputs, 0), self._l1_index) - tf.gather(tf.gather(self._y, 0), self._l1_index))
            l2_diff = (tf.gather(tf.gather(self._outputs, 0), self._l2_index) - tf.gather(tf.gather(self._y, 0), self._l2_index))
            l3_diff = (tf.gather(tf.gather(self._outputs, 0), self._l3_index) - tf.gather(tf.gather(self._y, 0), self._l3_index))
            # zero_diff = (tf.gather(tf.gather(self._outputs, 0), self._zero_index) - tf.gather(tf.gather(self._y, 0), self._zero_index))
            self._loss = tf.reduce_mean(loss_value)

            optimizer = tf.train.AdamOptimizer(learning_rate=self._learning_rate)
            self._train = optimizer.minimize(self._loss)

            init = tf.global_variables_initializer()
            gpu_options = tf.GPUOptions(allow_growth=True)
            ##INITIALIZE SESSION
            self._sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
            
            self._sess.run(init)
            
            self._saver = tf.train.Saver()

    def train(self, input_vects):
        X_input, y_input = self.get_seq(self._df, self._training_start_point, self._training_start_point+self._num_time_steps)
        # X_input, y_input = self.get_seq(self._df, self._training_start_point2, self._training_start_point2+self._num_time_steps)
        X_batch = X_input.reshape(1, -1,  self._num_time_steps)
        y_batch = y_input.reshape(1, -1, self._num_time_steps)
        # print('===== training data ====')
        # print(X_batch)
        # print(y_batch)

        iteration = 0
        while True:
            iteration += 1

            self._sess.run(self._train, feed_dict={self._X: X_batch, self._y: y_batch})
            # print('*****************************')
            # print(self._sess.run(self._l2_diff, feed_dict={self._X: X_batch, self._y: y_batch}))
            
            loss = self._loss.eval(session=self._sess, feed_dict={self._X: X_batch, self._y: y_batch})

            if iteration % 1000 == 0:
                print("========== Start Training =============")
                print("Start Training Time:",datetime.datetime.now())
                print(iteration, "\tloss:", loss)

            if iteration <= self._num_train_iterations:
                if loss <= self._accept_loss:
                    print(iteration, "\tloss:", loss)
                    break
                else:
                    pass
            else:
                break

        # save training accuracy
        y_train_pred = self._sess.run(self._outputs, feed_dict={self._X: X_batch})
        train_pred_result = y_train_pred.reshape(1, -1, self._num_time_steps)
        train_true_value = self._df.ix[:, self._training_start_point+self._num_time_steps]

        # Save Model for Later
        self._saver.save(self._sess, "./applications/%s/RNN/model/high_all_wMSE_layers_%s"  % (self._prefix,self._output_format))

        train_accuracy = self.model_accuracy(train_true_value, train_pred_result)
        return train_accuracy

    def test(self, input_vects):
        # RUN TRAINED MODEL AND ESTIMATED RESULT
        with tf.Session() as sess:
            # print('*****************************')
            self._saver.restore(self._sess, "./applications/%s/RNN/model/high_all_wMSE_layers_%s" % (self._prefix,self._output_format))
            
            X_new, y_batch = self.get_seq(self._df, self._testing_start_point, self._testing_start_point + self._num_time_steps)
            # X_new, y_batch = self.get_seq(self._df, self._training_start_point2+1, self._training_start_point2+self._num_time_steps+1)
            X_new = X_new.reshape(1, -1, self._num_time_steps)
            print('===== test x data =====')
            print(X_new.shape)
            y_pred = self._sess.run(self._outputs, feed_dict={self._X: X_new})
            print('===== predition y data ======')
            y_pred = np.around(y_pred, decimals=0)
            print(y_pred)
            
        # # get last colum of output then absolute then apply math floor
        pred_result = y_pred.reshape(1, -1, self._num_time_steps)

        true_value = self._df.ix[:, self._testing_start_point+self._num_time_steps]
        # print(true_value)
        # accurate to which level of ghsom map
        accuracy = self.model_accuracy(true_value, pred_result)
        print('----------Accuracy----------')
        return accuracy


    def get_seq(self, df, sequence_start_point, sequence_end_point):
        X_input = df.iloc[:, sequence_start_point:sequence_end_point].values
        y_input = df.iloc[:, sequence_start_point+1:sequence_end_point+1].values

        return X_input, y_input


    def floor_to_given_decimal(self, np_array, decimal):
        # certain_decimal = int(math.pow(10, decimal))
        # np_array = np_array*certain_decimal
        # np_array = int(np.floor(np_array)) / certain_decimal
        certain_decimal = int(math.pow(10, decimal))
        np_array = np_array*certain_decimal
        result = np.floor(np_array).astype(int) / certain_decimal

        return result


    def model_accuracy(self, true_df, pred_array):
        print(pred_array.shape)
        pred_array = np.absolute(np.squeeze(pred_array, axis=0)[:, -1]).reshape(-1, ((self._layer)))
        # pred_array = pred_array.astype(float32)
        print('===== Pred Array information =====')
        print(pred_array)
        print(pred_array.shape)
        pred_df = pd.DataFrame(data = pred_array,columns=range(self._layer))
        pred_df['clustered_label'] = ''
        for i in range(0,self._layer):
            pred_df['clustered_label'] = pred_df['clustered_label'] + pred_df[i].astype(int).astype(str)
        pred_df.to_csv('./applications/%s/RNN/result/cluster_predict_%s.csv' % (self._prefix,self._output_format),index=False)
        true_array = true_df.as_matrix().reshape(-1, (self._layer))
        print('===== True Array information =====')
        print(true_array)
        print(true_array.shape)

        #=====
        # if trian is seires of array
        #=====
        layer1_correct = 0
        layer2_correct = 0
        layer3_correct = 0
        zero_correct = 0


        difference = pred_array - true_array
        difference = difference.astype(int)
        total_number = difference.shape[0]
        print('===== difference =====')
        print(difference)

        for row in difference:
            for i in range(len(row)):
                if(row[i] == 0):
                    row[i] = 1
                else:
                    for j in range(i,len(row)) :
                        row[j] = 0
                    break
        # =====
        # Here is RNN with item cluster as input
        # =====
        # print('===== layer correct result =====')
        # print(difference)
        # result =  np.array([layer1_correct, layer2_correct, layer3_correct, zero_correct])/total_number
        # print(result)
        return difference


        # =====
        # Here is RNN with all items as input
        # result =  np.array([layer1_correct, layer2_correct, layer3_correct, zero_correct])/total_number


        #=====
        # if accuracy is tree structure traversal
        #=====
        # def first_nonzero(arr, axis, invalid_val=-1):
        #     mask = arr!=0
        #     return np.where(mask.any(axis=axis), mask.argmax(axis=axis), invalid_val)

        # difference = pred_array - true_array
        # print('===== cluster diff =====')
        # cluster_diff = difference[:,:-1]

        # diff_layer = first_nonzero(cluster_diff, axis=1, invalid_val=-1)

        # each_distance_within_same_cluster = []

        # for i, first_nonzero_index in enumerate(diff_layer):
        #     if first_nonzero_index != -1:
        #         c_pred = pred_array[i,:-1]
        #         c_true = true_array[i,:-1]
        #         c_stack = np.vstack((c_pred, c_true))

        #         distance = np.trim_zeros(np.squeeze(c_stack[:, first_nonzero_index:].reshape(1,-1)))
        #         distance = distance[distance != 0]
        #         distance = distance.size

        #     else:
        #         distance = 0

        #     each_distance_within_same_cluster.append(distance)
        
        # print("====== zero_accuracy =====")
        # zero_accuracy = (difference[:, -1].shape[0] - np.count_nonzero(difference[:, -1])) / difference[:, -1].shape[0]
        # zero_accuracy = difference[:, -1]
        # zero_accuracy[zero_accuracy != 0] = 1
        # print(zero_accuracy)

        # print(each_distance_within_same_cluster)
        # cluster_distance_mean = np.mean(np.asarray(each_distance_within_same_cluster))
        # result = np.column_stack((each_distance_within_same_cluster, zero_accuracy))
        
        # return result


