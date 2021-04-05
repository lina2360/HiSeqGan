import plotly.express as px
import plotly.offline as of
import pandas as pd
import numpy as np
import argparse
import get_ghsom_dim


parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('--name', type=str, default = None)
# parser.add_argument('--index', type=str, default = None)
args = parser.parse_args()

prefix = args.name
layers,max_layer,number_of_digits = get_ghsom_dim.layers(prefix)
pathlist = list()


label_file = pd.read_csv('./applications/%s/data/%s_with_clustered_label.csv' % (prefix, prefix), encoding='utf-8')
for i in range(1,max_layer+1):
    label_file['clusterL'+str(i)] = np.nan
    pathlist.append('clusterL'+str(i))
print(label_file)


for a in range(len(label_file['x_label'])):
  strings_x = label_file['x_label'][a].split('-')
  strings_y = label_file['y_label'][a].split('-')
  for j in range(1,len(strings_x)):
    label_file['clusterL'+str(j)][a] = strings_x[j]+'x'+strings_y[j]
label_file.to_csv('./applications/%s/data/%s_with_clustered_label.csv'% (prefix, prefix),index = False)
label_file = label_file.fillna('')

fig = px.treemap(label_file, path=pathlist,
  color = 'median',
  color_continuous_scale = 'RdBu',
  branchvalues = 'remainder'
)

fig.show()
of.plot(fig, filename=('./applications/%s/graphs/%s_map.html' % (prefix, prefix)))
