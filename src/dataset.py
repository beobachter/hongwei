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
        num_label_columns = 8
        # pands_data = self.data_merge(pands_data, num_label_columns)
        num_feature_columns = pands_data.shape[1]
        print(num_feature_columns)
        df_features = pands_data.iloc[:, :num_feature_columns - num_label_columns]
        df_labels = pands_data.iloc[:, num_feature_columns - num_label_columns :num_feature_columns] # "Batch名称" is not a label
        
        self.max_label = df_labels.max()
        self.min_label = df_labels.min()

        data_x = df_features.to_numpy()
        data_y =df_labels.to_numpy()
        
        row_columns = pands_data.columns.to_numpy()
        row_column_train = row_columns[:num_feature_columns - num_label_columns]
        row_column_label = row_columns[num_feature_columns - num_label_columns :num_feature_columns]

        print("row_column_train",row_column_train)
        print("row_column_label",row_column_label)
        
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

        # print("train_data", train_data)
        # print("train_label", train_label)

    def pop_data(self):
        return self.train_data, self.test_data, self.train_label ,self.test_label
    
    def numpy_to_pandas(self, predicted, columns):
        if isinstance(columns, (list)):
            return pd.DataFrame(predicted,columns = columns)
        else:            
            return pd.DataFrame(predicted,columns = [columns])

    def valuation(self, predicted_labels,test_labels):
        print(1-(np.mean(np.abs(predicted_labels - test_labels), axis=0)/(self.max_label-self.min_label)))
        # print((predicted_labels - test__labels).abs().mean(axis=0))


    def data_merge(self, data, num_label_columns):
        columns = [
            'A-01', 'A-02', 'A-03', 'A-04', 'A-05', 'A-06', 'A-07', 'A-08', 'A-09', 'A-10',
            'A-11', 'A-12', 'A-13', 'A-14', 'A-15', 'A-16', 'A-17', 'A-18', 'A-19', 'A-20',
            'A-21', 'A-22', 'A-23', 'A-24', 'A-25', 'A-26', 'A-27', 'A-28', 'A-29', 'A-30',
            'A-31', 'A-32', 'A-33', 'A-34', 'A-35', 'A-36', 'A-37', 'A-38', 'A-39', 'A-40',
            'A-41', 'A-42', 'A-43', 'A-44', 'A-45', 'A-46', 'A-47', 'A-48', 'A-49', 'A-50',
            'A-51', 'A-52', 'A-53', 'A-54', 'A-55', 'A-56', 'A-57', 'A-58', 'A-59', 'A-60',
            'A-61', 'A-62', 'A-63', 'A-64', 'A-65', 'A-66', 'A-67', 'A-68', 'A-69', 'A-70',
            'A-71', 'A-72', 'A-73', 'A-74', 'A-75', 'A-76', 'B-01', 'B-02', 'B-03', 'B-04',
            'B-05', 'B-06', 'B-07', 'B-08', 'B-09', 'B-10', 'B-11', 'B-12', 'B-13', 'B-14',
            'B-15', 'B-16', 'B-17', 'B-18', 'B-19', 'B-20', 'B-21', 'B-22', 'B-23', 'B-24',
            'B-25', 'B-26', 'B-27', 'B-28', 'B-29', 'B-30', 'B-31', 'B-32', 'B-33', 'B-34',
            'B-35'
            ]
        for column_name in columns:
            if(column_name not in data.columns):
                zeros_array = np.zeros(data.shape[0])
                data.insert(loc=data.shape[1]-num_label_columns, column=column_name, value=zeros_array)
                # data[column_name] = np.zeros(data.shape[0])
        return data


