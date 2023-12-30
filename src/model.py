

import xgboost
# First XGBoost model for Pima Indians dataset
from numpy import loadtxt
import pandas as pd
import xgboost as xgb
from xgboost import XGBClassifier,  XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import math
import numpy as np

import operator
from xgboost import plot_importance 
from utils import pearson, MinMaxScaler, timestamp,evaluate_accuracy,accuracy
from dataset import data_loader_xgb

import matplotlib.pyplot as plt



def ceate_feature_map(features):
    outfile = open('xgb.fmap', 'w')
    i = 0
    for feat in features:
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
        i = i + 1
    outfile.close()

def feature_importance_plot(importance_sorted, title):
    df = pd.DataFrame(importance_sorted, columns=['feature', 'fscore'])
    df['fscore'] = df['fscore'] / df['fscore'].sum()

    plt.figure()
    # df.plot()
    df.plot(kind='barh', x='feature', y='fscore',
            legend=False, figsize=(12, 10))
    plt.title('XGBoost Feature Importance')
    plt.xlabel('relative importance')
    plt.tight_layout()
    plt.savefig(title + '.png', dpi=300)
    plt.show()

class Xgboost_lk():
    def __init__(self):
        None
    # seed = 7
    # test_size = 0
    #X_train, X_test, y_train, y_test = train_test_split(train_X, train_Y, test_size=test_size, random_state=seed)
    # test_path='data/test.csv'
    # X_test, y_test=data_loader_xgb(test_path)
    # seed = 7
    # test_size = 0
    #X_train, X_test, y_train, y_test = train_test_split(train_X, train_Y, test_size=test_size, random_state=seed)


    # cat_sel = [n for n in dataset.columns if n.startswith('cat')]  #类别特征数值化
    # for column in cat_sel:
    #     dataset[column] = pd.factorize(dataset[column].values , sort=True)[0] + 1
    #
    # fit model no training datagoogle.csv

    # model = XGBClassifier(
    #     booster='gbtree',          # 使用的是树模型还是线性模型（gbtree，gblinear）
    #     learning_rate=0.3,
    #     n_estimators=100,         # 树的个数--1000棵树建立xgboost
    #     max_depth=6,               # 树的深度
    #     min_child_weight = 1,      # 叶子节点最小权重
    #     gamma=0.,                  # 惩罚项中叶子结点个数前的参数
    #     subsample=1,             # 随机选择80%样本建立决策树
    #     colsample_btree=1,       # 随机选择80%特征建立决策树
    #     objective='multi:softmax', # 指定损失函数
    #     scale_pos_weight=1,        # 解决样本个数不平衡的问题
    #     random_state=0            # 随机数
    # )

    model = XGBRegressor(
        booster='gbtree',
        learning_rate=0.3,
        n_estimators=100,
        max_depth=6,
        min_child_weight=1,
        gamma=0,
        subsample=1,
        colsample_btree=1,
        objective='reg:squarederror',
        random_state=42
    )

    def train_xgboost(self,X_train, y_train):

        model_final = self.model.fit(X_train, y_train)
        
    # make predictions for test data


    def predict_valid_xgboost(self, X_test):
        y_pred = self.model.predict(X_test)

        return y_pred
    
    def save_model(self, model_path):
        self.model.save_model(model_path)

    def load_model(self, model_path):
        booster = xgb.Booster()
        booster.load_model(model_path)
        self.model = booster

    def predict_xgboost(self, X_test):
        X_test = xgb.DMatrix(X_test)
        y_pred = self.model.predict(X_test)

        return y_pred

    def plo_show_significance(self,y_pred,y_test,label):
        y_pred = pd.DataFrame(y_pred, index=y_test.index, columns=[label])

        y_test = pd.DataFrame(y_test, index=y_test.index, columns=[label])

        y_pred=y_pred.values
        y_test=y_test.values
        acc=accuracy(y_pred,y_test)
        pes = pearson(y_pred,y_test)
        print(acc)
        print(pes)
        x = range(len(y_test))
        y = 1 * y_test
        z = 1 * y_pred
        plt.title("accuracy " + label)
        plt.ylabel("y axis caption")
        plt.xlabel("x axis caption")
        plt.plot(x, y)
        plt.plot(x, z)
        plt.show()


        fig,ax = plt.subplots(figsize=(15,15))
        plot_importance(self.model,
                    height=0.5,
                    ax=ax,
                    max_num_features=64)
        plt.show()