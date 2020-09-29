from keras.layers import Dense, Dropout, LSTM, Embedding, SimpleRNN
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn import metrics
# input_file = './raw-data/wm5-grade-clean.csv'
input_file = './applications/wm5-normalize-1/data/rnn_input_item_seq_with_cluster_integer_test.csv'

def load_data(test_split = 0.2):
    print ('Loading data...')
    df = pd.read_csv(input_file)
    # df_features = df.drop(['id', 'Group','Group Project','Exam1','Exam2','Final','Attendance','TA','original','semester','grade_class','passed','failed'], axis=1)
    df_features = df.drop(['id', 'clustered_label','passed','failed'], axis=1)
    df_label = df.loc[:, ['passed']]
    x_train, x_test, y_train, y_test = train_test_split(df_features, df_label, test_size=test_split, random_state=42)

    return np.array(x_train.reset_index(drop=True)), np.array(y_train.reset_index(drop=True)), np.array(x_test.reset_index(drop=True)), np.array(y_test.reset_index(drop=True))

def load_data_v2(test_split = 0.2):
    print ('Loading data...')
    df = pd.read_csv(input_file)
    # df_features = df.drop(['id', 'Group','Group Project','Exam1','Exam2','Final','Attendance','TA','original','semester','grade_class','passed','failed'], axis=1)
    df_features = df.drop(['id', 'clustered_label','passed','failed'], axis=1)
    df_label = df.loc[:, ['passed']]
    x_train, x_test, y_train, y_test = train_test_split(df_features, df_label, test_size=test_split, random_state=42)

    return x_train.reset_index(drop=True), y_train.reset_index(drop=True), x_test.reset_index(drop=True), y_test.reset_index(drop=True)


def create_model(shape):
    print ('Creating model...')
    model = Sequential()
    model.add(SimpleRNN(50, input_shape=(1, 18),unroll=True))
    model.add(Dense(50, activation='tanh'))
    # model.add(LSTM(100,input_shape=(None, 1),return_sequences=True))
    model.add(Dropout(0.3))
    # model.add(LSTM(output_dim=50, activation='sigmoid', inner_activation='hard_sigmoid', return_sequences=True))
    # model.add(Dropout(0.5))
    # model.add(LSTM(output_dim=50, activation='sigmoid', inner_activation='hard_sigmoid'))
    # model.add(Dropout(0.5))
    model.add(Dense(1, activation='softmax'))

    print ('Compiling...')
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    model.summary()
    return model

# ============================
# RNN
# ============================
# x_train, y_train, x_test, y_test = load_data()
# x_train = np.reshape(x_train, (x_train.shape[0], 1, x_train.shape[1]))
# x_test = np.reshape(x_test, (x_test.shape[0], 1, x_test.shape[1]))
# print('==========x_train==========')
# print('x_train shape:',x_train.shape)
# print(x_train)

# print('==========y_train==========')
# print('y_train shape:',y_train.shape)
# print(y_train)

# print('==========x_test==========')
# print('x_test shape:',x_test.shape)
# print(x_test)

# print('==========y_test==========')
# print('y_test shape:',y_test.shape)
# print(y_test)


# model = create_model(x_train.shape)

# print ('Fitting model...')
# hist = model.fit(x_train, y_train,epochs = 3, batch_size=1, verbose = 1)

# score, acc = model.evaluate(x_test, y_test, batch_size=1)
# print('Test score:', score)
# print('Test accuracy:', acc)

# predict = model.predict(x_test)
# print(predict)
# print(y_test)

# ============================
# Decision Tree
# ============================
x_train, y_train, x_test, y_test = load_data_v2()

print('==========x_train==========')
print('x_train shape:',x_train.shape)
print(x_train)

print('==========y_train==========')
print('y_train shape:',y_train.shape)
print(y_train)

print('==========x_test==========')
print('x_test shape:',x_test.shape)
print(x_test)

print('==========y_test==========')
print('y_test shape:',y_test.shape)
print(y_test)


clf = tree.DecisionTreeClassifier()
wm5_clf = clf.fit(x_train, y_train)

test_y_predicted = wm5_clf.predict(x_test)
print('predict value:',test_y_predicted)

# 標準答案
print('actual value:',y_test)

# 績效
accuracy = metrics.accuracy_score(y_test, test_y_predicted)
print('accuracy:',accuracy)