import pandas as pd
import argparse
import get_ghsom_dim

parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('--name', type=str, default = None)

args = parser.parse_args()
prefix = args.name
# seq_name = '%s-item-seq' % prefix
layers,max_layer,number_of_digits = get_ghsom_dim.layers(prefix)
layer = sum(number_of_digits)

point_label_path = './applications/%s/data/%s_with_point_label.csv' % (prefix,prefix)
df_point_label = pd.read_csv(point_label_path, low_memory=False)
df_point_label


clustered_label_path = './applications/%s/data/%s_with_clustered_label.csv' % (prefix,prefix)
df_clustered_label = pd.read_csv(clustered_label_path, low_memory=False)


df_merge = pd.DataFrame((df_clustered_label.clustered_label*pow(10,layer)).astype(int))
df_merge["point_x"] = df_point_label.point_x
df_merge["point_y"] = df_point_label.point_y
# df_merge['points'] = df_point_label.clustered_label
df_merge.drop_duplicates()



df_merge.drop_duplicates().to_csv('./applications/%s/data/merge_point_cluster_data.csv' % (prefix),index=False)
