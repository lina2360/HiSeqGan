# csv data to ghsom .in format
import pandas as pd
import csv
import numpy as np
# import argparse

# parser = argparse.ArgumentParser(description='manual to this script')
# parser.add_argument('--name', type=str, default = None)
# parser.add_argument('--index', type=str, default = None)
# parser.add_argument('--train_column', type=str, default = None)
# args = parser.parse_args()

def format_ghsom_input_vector(name, index, train_column):
    source_path = name.replace('-item-seq','')
    
    train_column = train_column.split(',')

    df = pd.read_csv('./raw-data/%s.csv' % name,encoding='utf-8')
    df = df[train_column].fillna(0)

    # data shape
    print('rows=',df.shape[0])
    print('columns=',df.shape[1])

    rows_amount = df.shape[0]
    columns_amount = df.shape[1]

    df[index] = range(0,rows_amount)
    print('./applications/%s/GHSOM/data/%s_ghsom.csv' % (source_path,name))
    df.to_csv('./applications/%s/GHSOM/data/%s_ghsom.csv' % (source_path,name),sep=' ',header=False,index=False)


    # set ghsom input data format
    # format information : http://www.ifs.tuwien.ac.at/~andi/somlib/download/SOMLib_Datafiles.html#input_vectors
    data_type = 'inputvec'
    x_dim = rows_amount
    y_dim = 1
    vec_dim = columns_amount

    with open('./applications/%s/GHSOM/data/%s_ghsom.in' % (source_path,name), 'w', newline='',encoding='utf-8') as csvfile:
        # 建立 CSV 檔寫入器 , 設定空白切割
        writer = csv.writer(csvfile)

        # Parameter settings
        writer.writerow(['$TYPE %s' % data_type])
        writer.writerow(['$XDIM %s' % x_dim])
        writer.writerow(['$YDIM %s' % y_dim])
        writer.writerow(['$VECDIM %s' % vec_dim])
        
        # Data settings
        with open('./applications/%s/GHSOM/data/%s_ghsom.csv' % (source_path,name),'r', newline='',encoding='utf-8') as rawfile:
            # 讀取 CSV 檔案內容
            rows = csv.reader(rawfile)
            writer.writerow([])
        
            # 以迴圈輸出每一列
            for row in rows:
                #print(row)
                writer.writerow(row)
        rawfile.close()
