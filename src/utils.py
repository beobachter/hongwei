import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import time
import pandas as pd
from math import sqrt
import matplotlib.pyplot as plt
from xgboost import plot_importance


def MinMaxScaler(data):
    """Normalizer (MinMax criteria).

  Args:
    - data: original data

  Returns:
    - norm_data: normalized data
  """
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    norm_data = numerator / (denominator + 1e-8)
    return norm_data


def returnScaler(norm_data):
    data = pd.read_csv('data/ep2.csv', sep=',', low_memory=False)
    df = pd.DataFrame(data)
    pre_data = df['WK_ED_R']
    denominator = np.max(pre_data, 0) - np.min(pre_data, 0)
    numerator = norm_data * (denominator + 1e-8)
    oridata = numerator + np.min(pre_data, 0)
    return oridata


def performance(test_y, test_y_hat, metric_name):
    """Evaluate predictive model performance.

  Args:
    - test_y: original testing labels
    - test_y_hat: prediction on testing data
    - metric_name: 'mse' or 'mae'

  Returns:
    - score: performance of the predictive model
  """
    assert metric_name in ['mse', 'mae']

    if metric_name == 'mse':
        score = mean_squared_error(test_y, test_y_hat)
    elif metric_name == 'mae':
        score = mean_absolute_error(test_y, test_y_hat)

    score = np.round(score, 4)

    return score



def timestamp(time1):
    time1 = time.strptime(str(time1), '%Y/%m/%d %H:%M:%S')
    time1 = time.mktime(time1)
    return time1




def MinMaxScaler(data):
    """Normalizer (MinMax criteria).

  Args:
    - data: original data

  Returns:
    - norm_data: normalized data
  """
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    norm_data = numerator / (denominator + 1e-8)
    numerator=np.min(data, 0)
    return norm_data


def returnScaler(path,norm_data):
    dataset = pd.read_csv(path, sep=',', low_memory=False)
    dataset.fillna(0, inplace=True)
    df = pd.DataFrame(dataset)
    pre_data = df.iloc[:,-1]
    denominator = np.max(pre_data, 0) - np.min(pre_data, 0)
    numerator = norm_data * (denominator + 1e-8)
    oridata = numerator + np.min(pre_data, 0)
    return oridata



def timestamp(time1):
    time1 = time.strptime(str(time1), '%Y/%m/%d %H:%M:%S')
    time1 = time.mktime(time1)
    return time1


def pearson(v1, v2):
    n = len(v1)
    #simple sums
    sum1 = sum(float(v1[i]) for i in range(n))
    sum2 = sum(float(v2[i]) for i in range(n))
    #sum up the squares
    sum1_pow = sum([pow(v, 2.0) for v in v1])
    sum2_pow = sum([pow(v, 2.0) for v in v2])
    #sum up the products
    p_sum = sum([v1[i] * v2[i] for i in range(n)])
    #分子num，分母denominator
    num = p_sum - (float(sum1)*float(sum2)/n)
    den = sqrt((sum1_pow-float(pow(sum1, 2.0)/n))*(sum2_pow-float(pow(sum2, 2.0)/n)))
    if den == 0:
        return 0.0
    return (num/den)

def accuracy(y_hat,y):
    return (y_hat==y.astype('float32')).mean()

def evaluate_accuracy(y_pred,y_test):
    acc_sum,n=0.0,0
    for x_p,y_t in y_pred,y_test:
        acc_sum+=(x_p==y_t).sum().asscalar()
        n+=y_t.size
    return acc_sum/n


def MAPE_(v1, v2):
    n = len(v1)
    sum=0
    for i in range(n):
        sum=sum+abs((v1[i]-v2[i])/(v1[i]))
    return (sum/n)

def data_merge(datafile1,datafile2):

    df1 = pd.read_csv(datafile1, header=0)
    print("df1",df1.columns.to_numpy(), df1.shape)
    df2 = pd.read_csv(datafile2, header=0)
    print("df2",df2.columns.to_numpy(), df2.shape)

    merged_df = pd.concat([df1, df2], axis=0, sort=False)
    print("merged_df", merged_df.columns.to_numpy(), merged_df.shape)

    return merged_df