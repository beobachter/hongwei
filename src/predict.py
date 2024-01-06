# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from model import Xgboost_lk
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

from dataset import data_loader_hongwei


# importance = model_final.get_fscore()
# importance_sorted = sorted(importance.items(), key=operator.itemgetter(1))
# feature_importance_plot(importance_sorted, 'feature importance')


# plt.show()

def single_predict():
    df = pd.read_csv('231229_all.CSV', header=0)
    
    # df = pd.read_excel("/home/user/liangk/hongwei/配色数据—鸿之微.xlsx")

    df.fillna(0, inplace=True)
    num_feature_columns = 63 + 36
    num_label_columns = 8


    # convert 'Findings' to 'CIE DF', '不合格' to 2, '合格' to 0, '警告' to 1
    df['Findings'] = df['Findings'].replace('unqualified', 1)
    df['Findings'] = df['Findings'].replace('qualified', 0)
    df['Findings'] = df['Findings'].replace('warnings', 2)

    # normalize the features and labels
    data_loader = data_loader_hongwei(df,train_rate=0)
    _ , test_data , _ , test_label = data_loader.pop_data()
    label = 'CIE DE'
    test_label = (test_label[label]).to_frame(name=label)

    Xgboost = Xgboost_lk()
    Xgboost.load_model("src/model/xgboost.model")
    predicted_labels = Xgboost.predict_xgboost(test_data)

    predicted_labels = data_loader.numpy_to_pandas(predicted_labels,columns=label)
    # show errors of the predicted labels in each column
    data_loader.valuation(predicted_labels, test_label)
    for i in test_label.columns:
        Xgboost.plo_show_significance(predicted_labels, test_label,i)

    df.to_csv('_1_predicted.csv', index=False)

def mul_predict():
    df = pd.read_csv('231229_all.CSV', header=0)
    
    # df = pd.read_excel("/home/user/liangk/hongwei/配色数据—鸿之微.xlsx")

    df.fillna(0, inplace=True)
    num_feature_columns = 63 + 36
    num_label_columns = 9


    # convert 'Findings' to 'CIE DF', '不合格' to 2, '合格' to 0, '警告' to 1
    df['Findings'] = df['Findings'].replace('unqualified', 1)
    df['Findings'] = df['Findings'].replace('qualified', 0)
    df['Findings'] = df['Findings'].replace('warnings', 2)

    # normalize the features and labels
    data_loader = data_loader_hongwei(df,train_rate=0)
    _ , test_data , _ , test_label = data_loader.pop_data()


    Xgboost = Xgboost_lk()
    Xgboost.load_model("src/model/xgboost.model")
    predicted_labels = Xgboost.predict_xgboost(test_data)

    predicted_labels = data_loader.numpy_to_pandas(predicted_labels, columns= test_label.columns.tolist())
    # show errors of the predicted labels in each column
    data_loader.valuation(predicted_labels, test_label)
    for i in test_label.columns:
        Xgboost.plo_show_significance(predicted_labels, test_label,i)

    df.to_csv('_1_predicted.csv', index=False)


if __name__ == '__main__':
    mul_predict()

