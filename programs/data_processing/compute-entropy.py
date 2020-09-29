from scipy.stats import entropy
import pandas as pd
import argparse

parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('--name', type=str, default = None)
parser.add_argument('--tau1', type=float, default = 0.1)
parser.add_argument('--tau2', type=float, default = 0.01)

args = parser.parse_args()

prefix = args.name
tau1 = args.tau1
tau2 = args.tau2

def compute_entropy(cluster_list):
    # print(cluster_list)
    return entropy(cluster_list,base=2)


#df = pd.read_csv('./applications/%s/data/%s_with_clustered_label_integer.csv' %(prefix,prefix))
# print(df['clustered_label'].unique())
df = pd.read_csv('./applications/%s/data/rnn_input_item_seq_with_cluster_integer.csv' %(prefix))
# print(df.shape[0])
p_list = []
for cluster in df['clustered_label'].unique():
    count = df['clustered_label'].tolist().count(cluster)
    p = count/df.shape[0]
    p_list.append(p) 
# print(df['clustered_label'].tolist().count(270))
print('tau1=',tau1)
print('tau2=',tau2)
print('clusters=',len(df['clustered_label'].unique()))
# print('p_list:',p_list)
# print('all p_list:',sum(p_list))

print('entropy=',compute_entropy(p_list))


f = open("entropy.txt", "a")
f.write("%s %s %s %s %s-item-seq\n" % (tau1,tau2,len(df['clustered_label'].unique()),compute_entropy(p_list), prefix))
f.close()