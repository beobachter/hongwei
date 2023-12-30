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

if __name__ == '__main__':


    df = pd.read_csv('/home/user/liangk/hongwei/data.CSV', header=0)
    
    # df = pd.read_excel("/home/user/liangk/hongwei/配色数据—鸿之微.xlsx")

    df.fillna(0, inplace=True)
    num_feature_columns = 63 + 36
    num_label_columns = 9


    # convert 'Findings' to 'CIE DF', '不合格' to 2, '合格' to 0, '警告' to 1
    df['Findings'] = df['Findings'].replace('unqualified', 1)
    df['Findings'] = df['Findings'].replace('qualified', 0)
    df['Findings'] = df['Findings'].replace('warnings', 2)

    # normalize the features and labels
    data_loader = data_loader_hongwei(df)
    train_data, test_data, train_label ,test_label = data_loader.pop_data()


    # # create a random forest regressor
    # regr = RandomForestRegressor(max_depth=10, random_state=0)
    # # train the model
    # regr.fit(train_data, train_label)
    # # predict the labels
    # predicted_labels = regr.predict(test_data)

    Xgboost = Xgboost_lk()
    Xgboost.train_xgboost(train_data, train_label)
    
    predicted_labels = Xgboost.predict_valid_xgboost(test_data)

    predicted_labels = data_loader.numpy_to_pandas(predicted_labels)
    # show errors of the predicted labels in each column
    data_loader.valuation(predicted_labels, test_label)
    Xgboost.plo_show_significance(predicted_labels, test_label,"Batch CIE L")

    Xgboost.save_model("src/model/xgboost.model")
    # scaler = StandardScaler()
    # scaler.fit(df_features)
    # df_features = scaler.transform(df_features)
    # scaler.fit(df_labels)
    # df_labels = scaler.transform(df_labels)

    
    # for i in range(num_label_columns):
    #     df['Batch CIE L pred'] = predicted_labels[:, 0]
    #     df['Batch CIE a pred'] = predicted_labels[:, 1]
    #     df['Batch CIE b pred'] = predicted_labels[:, 2]
    #     df['Batch CIE DL pred'] = predicted_labels[:, 3]
    #     df['Batch CIE Da pred'] = predicted_labels[:, 4]
    #     df['Batch CIE Db pred'] = predicted_labels[:, 5]
    #     df['Batch CIE DE pred'] = predicted_labels[:, 6]
    #     df['Findings pred'] = predicted_labels[:, 7]

    # # round df['Findings pred'] to the nearest integer
    # df['Findings pred'] = df['Findings pred'].round()
    # # df['Findings pred'], convert the predicted labels back to '不合格', '合格', '警告'
    # df['Findings pred'] = df['Findings pred'].replace(1, 'unqualified')
    # df['Findings pred'] = df['Findings pred'].replace(0, 'qualified')
    # df['Findings pred'] = df['Findings pred'].replace(2, 'warnings')

    # # show only the labels and the predicted labels are shown
    # df.iloc[:, num_feature_columns + 1 :]
    # save the dataframe to a csv file
    df.to_csv('_1_predicted.csv', index=False)
