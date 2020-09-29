import pandas as pd
import numpy as np
import get_ghsom_dim
import argparse
#import codecs
parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('--name', type=str, default = None)
args = parser.parse_args()
prefix = args.name

# raw_data = pd.read_csv('./applications/%s/data/rnn_input_item_seq_with_cluster_integer.csv' % prefix)

# rnn_data = raw_data

# # delete ItemShortName column and reset index
# raw_data= raw_data.drop(['CUSTOMER_NUM'], axis=1)
# raw_data.columns = range(raw_data.shape[1])

# # drop duplicates rows and reset index
# # raw_data = raw_data.drop_duplicates()
# raw_data.index = range(raw_data.shape[0])

# # read seqgan generate sequence data
# result_data = pd.read_csv('./applications/%s/SeqGAN/seqGAN_output_data.txt' % prefix,header=None,sep=' ')
# print("SeqGAN data shape : ",result_data.shape)
# # result_data = result_data.head(10)
# # print(result_data.iloc[0][:91]-raw_data.iloc[0][:91])
# # sum((result_data.iloc[0][:91] - raw_data.iloc[0][:91])**2)


# result = result_data.loc[0].to_frame().T
# print('raw_data shape:',raw_data.shape)
# for i in range(raw_data.shape[0]):
# # for i in range(1000): 
#     print("i=",i)
#     row = result_data.loc[0].to_frame().T
#     min_error = 100000000000000

#     for j in range(result_data.shape[0]):
#     # for j in range(1):
#         # print("min_error : ",min_error)
#         # print('j=',j)
#         # print("now : ",result_data.iloc[j][:91] - raw_data.iloc[i][:91])
#         if min_error >= (sum((result_data.iloc[j][:14] - raw_data.iloc[i][:14])**2)) :
#             row = result_data.loc[j].to_frame().T
#             min_error = sum((result_data.iloc[j][:14] - raw_data.iloc[i][:14])**2)
#             # print('cluster')
#             #print(sum((result_data.iloc[i] - raw_data.iloc[j][:96])**2))
#     result = pd.concat([result,row])
#     # print(result)

# result_add_item = result.reset_index(drop=True)
# # delete first row
# result_add_item = result_add_item.drop(result_add_item.index[[0]])
# # change cluster datatype from float to int64
# seqgan_data = result_add_item.astype('int64')

# # add item column in seqGAN data and modify column names for merge
# seqgan_data.insert(18,'CUSTOMER_NUM','none')
# seqgan_data.insert(19,'clustered_label','none')
# print(seqgan_data)
# # print(seqgan_data.columns)
# print(rnn_data.columns)
# seqgan_data.columns = rnn_data.columns
# seqgan_data['clustered_label'] = rnn_data['clustered_label']
# seqgan_data.to_csv('./applications/%s/SeqGAN/seqgan_simliar_rnn_output.csv' % prefix,index=False)

predict_days = 5

# compute seqGAN accuracy
raw_data = pd.read_csv('./applications/%s/data/rnn_input_item_seq_with_cluster_integer.csv' % prefix)
raw_data = raw_data.iloc[:,13:18]
raw_data.columns=range(0,predict_days)
print("============ raw_data ============")
print(raw_data)

seqgan_data = pd.read_csv('./applications/%s/SeqGAN/seqgan_simliar_rnn_output.csv' % prefix )
seqgan_data = seqgan_data.iloc[:,13:18]
seqgan_data.columns=range(0,predict_days)
print("============ seqgan_data ============")
print(seqgan_data)

layers,max_layer,number_digits = get_ghsom_dim.layers(prefix)
number_digits.append(0)
print('Layer:',number_digits.reverse())
layer = 0
# All layer 
accuracy_list = np.zeros((len(number_digits)-1, 5)) # 取代 count92,93,94,95,96
for layer_value in range(len(number_digits)-1):
    layer = number_digits[layer_value] + layer 
    for i in range(int(seqgan_data.shape[0])):
        
        for day in range(accuracy_list.shape[1]):
            print('i=',i)
            print('layer_value=',layer_value,'\t day=',day)
            print(raw_data.iloc[i][day]//pow(10,layer),'-',seqgan_data.iloc[i][day]//pow(10,layer),'=',raw_data.iloc[i][day]//pow(10,layer)-seqgan_data.iloc[i][day]//pow(10,layer))
            if (raw_data.iloc[i][day]//pow(10,layer)-seqgan_data.iloc[i][day]//pow(10,layer))==0:
                accuracy_list[layer_value][day] = accuracy_list[layer_value][day] + 1
        print(accuracy_list)

df_layer = pd.DataFrame(accuracy_list/seqgan_data.shape[0])
# print(df_layer)

layer_name = []
for i in range(len(number_digits)-1,0,-1):
    print(i)
    layer_display = list(range(i, 0,-1))
    layer_name.append('Layer %s '% layer_display)
    
print(layer_name)    
df_layer['layer'] = layer_name
print(df_layer)
df_layer.to_csv('./applications/%s/SeqGAN/seqgan_accuracy.csv'  % prefix , index=False)