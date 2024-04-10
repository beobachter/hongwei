# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from model import Xgboost_lk
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from utils import data_merge
from dataset import data_loader_hongwei
import argparse

def single_train():
    df = pd.read_csv('peise.CSV', header=0)
    # df = data_merge("231229_all.CSV", "data.CSV")
    # df = pd.read_excel("/home/user/liangk/hongwei/配色数据—鸿之微.xlsx")

    # df.fillna(0, inplace=True)
    num_feature_columns = 90 + 37
    num_label_columns = 8


    # convert 'Findings' to 'CIE DF', '不合格' to 2, '合格' to 0, '警告' to 1
    df['Findings'] = df['Findings'].replace('unqualified', 1)
    df['Findings'] = df['Findings'].replace('qualified', 0)
    df['Findings'] = df['Findings'].replace('warnings', 2)

    # normalize the features and labels
    data_loader = data_loader_hongwei(df)
    train_data, test_data, train_label ,test_label = data_loader.pop_data()
    print(train_label.columns)
    label = 'CIE DE'
    test_label = (test_label[label]).to_frame(name=label)
    train_label = (train_label[label]).to_frame(name=label)


    Xgboost = Xgboost_lk()
    Xgboost.train_xgboost(train_data, train_label)
    
    predicted_labels = Xgboost.predict_valid_xgboost(test_data)
    
    predicted_labels = data_loader.numpy_to_pandas(predicted_labels,columns=label)
    # show errors of the predicted labels in each column
    data_loader.valuation(predicted_labels, test_label)
    
    # Xgboost.plo_show_significance(predicted_labels, test_label,"Batch CIE L")

    Xgboost.save_model("src/model/xgboost.model")

    df.to_csv('_1_predicted.csv', index=False)


def mul_train(input_path):
    df = pd.read_csv(input_path, header=0)
    # df = data_merge("231229_all.CSV", "data.CSV")
    # df = pd.read_excel("/home/user/liangk/hongwei/配色数据—鸿之微.xlsx")

    # df.fillna(0, inplace=True)
    num_feature_columns = 63 + 36
    num_label_columns = 9


    # convert 'Findings' to 'CIE DF', '不合格' to 2, '合格' to 0, '警告' to 1
    df['Findings'] = df['Findings'].replace('unqualified', 1)
    df['Findings'] = df['Findings'].replace('qualified', 0)
    df['Findings'] = df['Findings'].replace('warnings', 2)

    # normalize the features and labels
    data_loader = data_loader_hongwei(df)
    train_data, test_data, train_label ,test_label = data_loader.pop_data()
    print(train_data)
    print(train_label)
    print(test_data)
    print(test_label)


    Xgboost = Xgboost_lk()
    Xgboost.train_xgboost(train_data, train_label)

    predicted_labels = Xgboost.predict_valid_xgboost(test_data)

    predicted_labels = data_loader.numpy_to_pandas(predicted_labels, columns= test_label.columns.tolist())
    # show errors of the predicted labels in each column
    data_loader.valuation(predicted_labels, test_label)
    Xgboost.plo_show_significance(predicted_labels, test_label,"Batch CIE L")

    Xgboost.save_model("src/model/xgboost.model")

    df.to_csv('_1_predicted.csv', index=False)
# plt.show()



def single_train1(input_path,label):
    df = pd.read_csv(input_path, header=0)
    # df = data_merge("231229_all.CSV", "data.CSV")
    # df = pd.read_excel("/home/user/liangk/hongwei/配色数据—鸿之微.xlsx")

    # df.fillna(0, inplace=True)
    num_feature_columns = 63 + 36
    num_label_columns = 8


    # convert 'Findings' to 'CIE DF', '不合格' to 2, '合格' to 0, '警告' to 1
    df['Findings'] = df['Findings'].replace('unqualified', 1)
    df['Findings'] = df['Findings'].replace('qualified', 0)
    df['Findings'] = df['Findings'].replace('warnings', 2)

    # normalize the features and labels
    data_loader = data_loader_hongwei(df,train_rate=0.8)
    train_data, test_data, train_label ,test_label = data_loader.pop_data()


    df = pd.read_csv('peise.CSV', header=0)
    # df = data_merge("231229_all.CSV", "data.CSV")
    # df = pd.read_excel("/home/user/liangk/hongwei/配色数据—鸿之微.xlsx")

    # df.fillna(0, inplace=True)
    num_feature_columns = 63 + 36
    num_label_columns = 8


    # convert 'Findings' to 'CIE DF', '不合格' to 2, '合格' to 0, '警告' to 1
    df['Findings'] = df['Findings'].replace('unqualified', 1)
    df['Findings'] = df['Findings'].replace('qualified', 0)
    df['Findings'] = df['Findings'].replace('warnings', 2)

    # normalize the features and labels
    # data_loader = data_loader_hongwei(df,train_rate=0)
    # _, test_data, _ ,test_label = data_loader.pop_data()



    print(train_label.columns)
    test_label = (test_label[label]).to_frame(name=label)
    train_label = (train_label[label]).to_frame(name=label)


    Xgboost = Xgboost_lk()
    Xgboost.train_xgboost(train_data, train_label)
    
    predicted_labels = Xgboost.predict_valid_xgboost(test_data)
    print(predicted_labels)
    predicted_labels = data_loader.numpy_to_pandas(predicted_labels,columns=label)

    # show errors of the predicted labels in each column
    data_loader.valuation(predicted_labels, test_label)
    
    Xgboost.plo_show_significance(predicted_labels, test_label,label=label)

    Xgboost.save_model("src/model/xgboost"+label+".model")

    df.to_csv('_1_predicted.csv', index=False)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='train type and label',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input', '-i', default=None,
                        help="train data file path")
    parser.add_argument('--model', '-m', default='xgboost',
                        help="models: xgboost")
    parser.add_argument('--multi',  action='store_true', 
                        help='In case of multivariate prediction, it is True')
    parser.add_argument('--label', '-l',  type=str, default=None,
                        help='label of single variables')
    args = parser.parse_args()
    if args.multi:
        mul_train( args.input)
    else:
        single_train1(args.input, args.label)


