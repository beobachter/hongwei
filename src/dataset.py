# coding : UTF-8


# Necessary Packages

import numpy as np
import pandas as pd
from utils import MinMaxScaler
from utils import timestamp
from sklearn.model_selection import train_test_split

def data_loader(path,train_rate=0.8, seq_len=7):
    """Loads Google stock data.
  
  Args:
    - train_rate: the ratio between training and testing sets
    - seq_len: sequence length
    
  Returns:
    - train_x: training feature
    - train_y: training labels
    - test_x: testing features
    - test_y: testing labels
  """

    # Load data
    # ori_data = np.loadtxt('data/ep3.csv', delimiter=',', skiprows=1)
    # ori_data = pd.read_csv('data/ep1.csv', sep=',',)
    # index_col = 'Day'
    # print(ori_data)
    # Reverse the time order

    # data = pd.read_csv('data/onehot.csv', sep=',', low_memory=False)
    # column_name = data.columns
    # df = data.set_index('date&time')
    sca = True
    Tim = True
    dataset = pd.read_csv(path, sep=',', low_memory=False)
    df = pd.DataFrame(dataset)
    if Tim == False:
        time1 = df['date&time']
        time1 = time1.values
        for i in range(0, len(time1)):
            time1[i] = timestamp(time1[i]) * 1e-8
        df['date&time'] = time1

    def sca_labers(text_labels, labels):
        return [text_labels[int(i)] for i in labels]

    if sca == False:
        sca_ = df['Activity']
        sca_ = sca_.values
        test_labels = ['STANDING', 'SITTING', 'LAYING', 'WALKING', 'WALKING_DOWNSTAIRS', 'WALKING_UPSTAIRS']
        for i in range(len(sca_)):
            for j in range(len(test_labels)):
                if (sca_[i] == test_labels[j]):
                    sca_[i] = j
                    break
        df['Activity'] = sca_


    df=df.astype(float)
    ori_data = df.iloc[::-1]

    ori_data = ori_data.values

    # Normalization

    norm_data  = MinMaxScaler(ori_data)
    norm_data = norm_data[::-1]#反向读取
    print(norm_data)
    # norm_data = reverse_data
    # Build dataset
    data_x = []
    data_y = []

    for i in range(0, len(norm_data[:, 0]) - seq_len-1):
        # Previous seq_len data as features
        temp_x = norm_data[i:i + seq_len, :]  # 取i-i+seq_len个数组
        # print(norm_data[i + seq_len])
        # print("norm_data[i + seq_len]")
        # Values at next time point as labels
        temp_y = norm_data[i + seq_len+1, [-1]]
        temp_x[-1,-1]=0
        data_x = data_x + [temp_x]
        data_y = data_y + [temp_y]
    # print(data_x[0])
    # print("wanc")
    # print(data_y[0])
    # print("wanc")

    data_x = np.asarray(data_x)
    data_y = np.asarray(data_y)

    # Train / test Division
    # idx = np.random.permutation(len(data_x))
    # train_idx = idx[:int(train_rate * len(data_x))]
    # test_idx = idx[int(train_rate * len(data_x)):]
    #
    # train_x, test_x = data_x[train_idx, :, :], data_x[test_idx, :, :]
    # train_y, test_y = data_y[train_idx, :], data_y[test_idx, :]
    # print(train_x.shape)
    #
    # return train_x, train_y, test_x, test_y
    return data_x,data_y


def data_loader_xgb(path,sca=False,Tim=True):
    dataset = pd.read_csv(path, sep=',', low_memory=False)
    df = pd.DataFrame(dataset)
    if Tim==False:
       time1 = df['date&time']
       time1 = time1.values
       for i in range(0, len(time1)):
           time1[i] = timestamp(time1[i]) * 1e-8
       df['date&time'] = time1


    def sca_labers(text_labels, labels):
        return [text_labels[int(i)] for i in labels]

    if sca == False:
        sca_=df['Activity']
        sca_=sca_.values
        test_labels=['STANDING','SITTING','LAYING','WALKING','WALKING_DOWNSTAIRS','WALKING_UPSTAIRS']
        for i in range(len(sca_)):
            for j in range(len(test_labels)):
              if (sca_[i]==test_labels[j]) :
                  sca_[i]=j
                  break

    df['Activity']=sca_

    dataset = df.astype(float)
    columns = dataset.columns
    data = dataset

    print(data.shape)
    # split data into X and y
    X = data.iloc[:, :-1].astype(float)
    Y = data.iloc[:,-1].astype(float)
    print(columns)
    # data_x = np.asarray(X)
    # data_y = np.asarray(Y)
    print(X.shape)
    print(Y.shape)
    # split data into train and test sets
    return X,Y


def data_loader_1(path, train_rate=0.8, seq_len=7):
    """Loads Google stock data.

  Args:
    - train_rate: the ratio between training and testing sets
    - seq_len: sequence length

  Returns:
    - train_x: training feature
    - train_y: training labels
    - test_x: testing features
    - test_y: testing labels
  """

    # Load data
    # ori_data = np.loadtxt('data/ep3.csv', delimiter=',', skiprows=1)
    # ori_data = pd.read_csv('data/ep1.csv', sep=',',)
    # index_col = 'Day'
    # print(ori_data)
    # Reverse the time order

    # data = pd.read_csv('data/onehot.csv', sep=',', low_memory=False)
    # column_name = data.columns
    # df = data.set_index('date&time')
    sca = True
    Tim = True
    dataset = pd.read_csv(path, sep=',', low_memory=False)
    dataset.fillna(0, inplace=True)
    df = pd.DataFrame(dataset)
    if Tim == False:
        time1 = df['date&time']
        time1 = time1.values
        for i in range(0, len(time1)):
            time1[i] = timestamp(time1[i]) * 1e-8
        df['date&time'] = time1

    def sca_labers(text_labels, labels):
        return [text_labels[int(i)] for i in labels]

    if sca == False:
        sca_ = df['Activity']
        sca_ = sca_.values
        test_labels = ['STANDING', 'SITTING', 'LAYING', 'WALKING', 'WALKING_DOWNSTAIRS', 'WALKING_UPSTAIRS']
        for i in range(len(sca_)):
            for j in range(len(test_labels)):
                if (sca_[i] == test_labels[j]):
                    sca_[i] = j
                    break
        df['Activity'] = sca_

    df = df.astype(float)
    ori_data = df.iloc[::-1]

    ori_data = ori_data.values
    ori_data = ori_data.astype(float)


    norm_data = MinMaxScaler(ori_data)
    norm_data = norm_data[::-1]  # 反向读取
    print(norm_data)

    data_x = []
    data_y = []

    for i in range(0, len(norm_data[:, 0]) - seq_len - 1):

        temp_x = norm_data[i:i + seq_len, :]  # 取i-i+seq_len个数组

        temp_y = norm_data[i + seq_len, [-1]]
        temp_x[-1, -1] = 0
        data_x = data_x + [temp_x]
        data_y = data_y + [temp_y]


    data_x = np.asarray(data_x)
    data_y = np.asarray(data_y)
    #Train / test Division
    idx = np.random.permutation(len(data_x))
    train_idx = idx[:int(train_rate * len(data_x))]
    test_idx = idx[int(train_rate * len(data_x)):]

    train_x, test_x = data_x[train_idx, :, :], data_x[test_idx, :, :]
    train_y, test_y = data_y[train_idx, :], data_y[test_idx, :]
    print(train_x.shape)

    return train_x, train_y, test_x, test_y


class data_loader_hongwei():
    def __init__(self, pands_data, train_rate=0.8):
        num_feature_columns = 63 + 36
        num_label_columns = 9
        df_features = pands_data.iloc[:, :num_feature_columns]
        df_labels = pands_data.iloc[:, num_feature_columns + 1 :num_feature_columns + num_label_columns] # "Batch名称" is not a label
        
        self.max_label = df_labels.max()
        self.min_label = df_labels.min()

        data_x = df_features.to_numpy()
        data_y =df_labels.to_numpy()
        
        row_columns = pands_data.columns.to_numpy()
        row_column_train = row_columns[:num_feature_columns]
        row_column_label = row_columns[num_feature_columns + 1 :num_feature_columns + num_label_columns]
        self.row_column_label = row_column_label
        idx = np.random.permutation(len(pands_data))

        train_idx = idx[:int(train_rate * len(data_x))]
        test_idx = idx[int(train_rate * len(data_x)):]

        
        
        train_data, test_data = data_x[train_idx, :], data_x[test_idx, :]
        train_label, test_label = data_y[train_idx, :], data_y[test_idx, :]

        # train_data = np.insert(train_data, 0, row_column_train, axis=0)
        # test_data = np.insert(test_data, 0, row_column_train, axis=0)
        # train_label = np.insert(train_label, 0, row_column_label, axis=0)
        # test_label = np.insert(test_label, 0, row_column_label, axis=0)



        print(train_data.shape,test_data.shape,train_label.shape,test_label.shape)

        self.train_data = pd.DataFrame(train_data,columns=row_column_train)
        self.test_data = pd.DataFrame(test_data,columns=row_column_train)
        self.train_label = pd.DataFrame(train_label,columns=row_column_label)
        self.test_label = pd.DataFrame(test_label,columns=row_column_label)

        print(train_data)
        print(train_label)

    def pop_data(self):
        return self.train_data, self.test_data, self.train_label ,self.test_label
    
    def numpy_to_pandas(self, predicted):
        return pd.DataFrame(predicted,columns=self.row_column_label)

    def valuation(self, predicted_labels,test__labels):
        print(np.mean(np.abs(predicted_labels - test__labels), axis=0)/(self.max_label-self.min_label))

