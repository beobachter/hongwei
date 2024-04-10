

# First XGBoost model for Pima Indians dataset
from numpy import loadtxt
import pandas as pd
import xgboost as xgb
from xgboost import XGBClassifier,  XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import math
import numpy as np
from sklearn.model_selection import RandomizedSearchCV, train_test_split
import operator
from xgboost import plot_importance 
from utils import pearson, MinMaxScaler, timestamp,evaluate_accuracy,accuracy
from dataset import data_loader_xgb
from sklearn.model_selection import KFold, cross_val_score
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import make_scorer
import joblib

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
        self.model = XGBRegressor(
            booster='gbtree',
            learning_rate=0.3,
            n_estimators=100,
            max_depth=6,
            min_child_weight=1,
            gamma=0,
            subsample=1,
            # colsample_btree=1,
            objective='reg:squarederror',
            random_state=42
        )
        self.kfold = KFold(n_splits=5, shuffle=True, random_state=42)
        self.param_space = {
            'n_estimators': [30, 50, 100, 200],
            'max_depth': [3, 5, 6, 7, 10, 12],
            'learning_rate': [0.01, 0.1, 0.2, 0.3, 0.4, 0.5]
        }
        self.mse_scorer = make_scorer(mean_squared_error, greater_is_better=False)
        

    def train_xgboost(self,X_train, y_train):
        self.model = RandomizedSearchCV(
            estimator=self.model,
            param_distributions=self.param_space,
            n_iter=10,
            cv=self.kfold,
            scoring=self.mse_scorer,
            random_state=42
        )
        self.model.fit(X_train, y_train)
        print("Best Parameters:", self.model.best_params_)
        print("Best Score:", self.model.best_score_)
        # scores = cross_val_score(self.model, X_train, y_train, cv=kfold)
        # for fold, score in enumerate(scores, 1):
        #     print(f'Fold {fold}: {score}')
        # model_final = self.model.fit(X_train, y_train)
        
    # make predictions for test data


    def predict_valid_xgboost(self, X_test):
        y_pred = self.model.best_estimator_.predict(X_test)
        return y_pred
    
    def save_model(self, model_path):
        joblib.dump(self.model.best_estimator_, model_path)

    def load_model(self, model_path):
        self.model.best_estimator_ = joblib.load(model_path)

    def predict_xgboost(self, X_test):
        y_pred = self.model.best_estimator_.predict(X_test)
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
        try:
            plot_importance( self.model.best_estimator_,
                        height=0.5,
                        ax=ax,
                        max_num_features=64)
        except:
            plot_importance( self.model.best_estimator_,
                        height=0.5,
                        ax=ax,
                        max_num_features=64)
        plt.show()

    def plot_learning_curve(self, X_train, y_train, X_valid, y_valid):
        # 训练和验证误差随训练样本数量的变化曲线
        train_errors, valid_errors = [], []

        for m in range(1, len(X_train) + 1):
            self.model.fit(X_train[:m], y_train[:m])
            y_train_pred = self.model.predict(X_train[:m])
            y_valid_pred = self.model.predict(X_valid)

            train_errors.append(mean_squared_error(y_train[:m], y_train_pred))
            valid_errors.append(mean_squared_error(y_valid, y_valid_pred))

        plt.plot(np.sqrt(train_errors), "r-+", linewidth=2, label="train")
        plt.plot(np.sqrt(valid_errors), "b-", linewidth=3, label="valid")
        plt.legend(loc="upper right", fontsize=14)
        plt.xlabel("Training set size", fontsize=14)
        plt.ylabel("RMSE", fontsize=14)
        plt.show()