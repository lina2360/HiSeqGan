from fractions import Fraction
import pandas as pd
import argparse

parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('--name', type=str, default = None)
# parser.add_argument('--index', type=str, default = None)
args = parser.parse_args()

prefix = args.name
source_path = args.name.replace('-item-seq','')

def GHSOM_center_point(df):
    Bx = By = 1
    Bx_list = []
    By_list = []
    Point_list = []
    for i in range(df.shape[0]):
        Bx = Bx * (Fraction(1, df.loc[i]['XDIM']))
        Bx_list.append(Bx)
        By = By * (Fraction(1, df.loc[i]['YDIM']))
        By_list.append(By)

        Point = [ Bx * df.loc[0]['X'], By * df.loc[0]['Y']]
        Point_list.append(Point)

    Px = Fraction(0, 1)
    for j in Point_list:
        Px = Px + j[0]
    Px = Px + Bx_list[-1]* 1/2

    Py = Fraction(0, 1)
    for j in Point_list:
        Py = Py + j[0]
    Py = Py + By_list[-1]* 1/2
    return ([Px,Py])

df_raw = pd.read_csv('./raw-data/%s.csv' % (prefix))
df_source = pd.read_csv('./applications/%s/data/%s_with_coordinate_representation.csv' % (source_path,prefix))
def map_cluster_to_ghsom(df_source):
    point_label_x = []
    point_label_y = []
    for i in df_source['clustered_label']:
        # print(i.split(';'))
        pointarray = i.split(';')
        point = []
        dimension_list = []
        for i in range(0,len(pointarray),4):
            # pointarray[i] = int(pointarray[i])
            dimension_list.append([pointarray[i],pointarray[i+1],pointarray[i+2],pointarray[i+3]])
        # print(dimension_list)
        dimension_df = pd.DataFrame(dimension_list,columns=['XDIM', 'YDIM', 'X', 'Y'])
        # print(dimension_df)

        point = GHSOM_center_point(dimension_df.astype('int64'))
        # print([point])
        point_label_x.append(point[0])
        point_label_y.append(point[1])
    df_source['point_x'] = point_label_x
    df_source['point_y'] = point_label_y
    
    return(df_source)
df_source = map_cluster_to_ghsom(df_source)
df_raw['point_x'] = df_source['point_x']
df_raw['point_y'] = df_source['point_y']
df_raw.to_csv('./applications/%s/data/%s_with_point_label.csv' % (source_path, prefix), index=False)
print(df_raw)




