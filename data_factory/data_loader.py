import time
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.linalg import hankel

class OnlineSILoader(object):
    def __init__(self, data_path, win_size, step, prediction_length, mode="train"):
        self.prediction_length = prediction_length
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = pd.read_csv(data_path + '/initialization_train.csv')
        data = data.iloc[:, 1:]
        self.scaler.fit(data)
        data = self.scaler.transform(data)

        valid_data = pd.read_csv(data_path + "/initialization_valid.csv")
        valid_data = valid_data.iloc[:, 1:]
        self.valid = self.scaler.transform(valid_data)

        test_data = pd.read_csv(data_path + "/initialization_valid.csv")
        test_data = test_data.iloc[:, 1:]
        self.test = self.scaler.transform(test_data)

        self.train = data
        self.test_labels = pd.read_csv(data_path + "/initialization_valid_label.csv").iloc[:, 1]
        self.train_labels = pd.read_csv(data_path + "/initialization_train_label.csv").iloc[:, 1]
        self.val_labels = pd.read_csv(data_path + "/initialization_valid_label.csv").iloc[:, 1]

    def __len__(self):

        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step - self.prediction_length
        elif (self.mode == 'val'):
            return (self.valid.shape[0] - self.win_size) // self.step - self.prediction_length
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step - self.prediction_length
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index:index + self.win_size+self.prediction_length]), np.float32(self.train_labels[index:index + self.win_size]), np.float32(self.train_labels[index + self.win_size: index + self.win_size + self.prediction_length])
        elif (self.mode == 'val'):
            return np.float32(self.valid[index:index + self.win_size+self.prediction_length]), np.float32(self.val_labels[index:index + self.win_size]), np.float32(self.val_labels[index + self.win_size: index + self.win_size + self.prediction_length])
        elif (self.mode == 'test'):
            return np.float32(self.test[index:index + self.win_size+self.prediction_length]), np.float32(self.test_labels[index:index + self.win_size]), np.float32(self.test_labels[index + self.win_size: index + self.win_size + self.prediction_length])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])

    def get_basis_from_labels(self, n_delays):
        hankel_matrix = hankel(self.train_labels[:len(self.train_labels)-n_delays], self.train_labels[-n_delays:])
        start = time.perf_counter()
        u, s, _ = np.linalg.svd(hankel_matrix.T)
        end = time.perf_counter()
        elapsed = end - start
        print(elapsed)
        return u


def get_loader_segment(data_path, batch_size, prediction_length=2, win_size=100, step=1, mode='train', dataset='KDD'):

    if dataset in ['Online_Initialization', 'Online_Recursive_0', 'Online_Recursive_1', 'Online_Recursive_2', 'Online_Recursive_3', 'Online_Recursive_4']:
        dataset = OnlineSILoader(data_path, win_size, step, prediction_length, mode)
    shuffle = False
    if mode == 'train':
        shuffle = False

    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             num_workers=0,
                             drop_last=False,
                             pin_memory=True
                             )

    return data_loader

def get_specific_basis_indataloader(n_delays, data_path, prediction_length=2, win_size=100, step=1, mode='train', dataset='KDD'):
    if dataset in ['Online_Initialization', 'Online_Recursive_0', 'Online_Recursive_1', 'Online_Recursive_2', 'Online_Recursive_3', 'Online_Recursive_4']:
        dataset = OnlineSILoader(data_path, win_size, step, prediction_length, mode)
        specific_basis = dataset.get_basis_from_labels(n_delays)
    else:
        raise ValueError(f"Unknown Dataset: {dataset}")

    return specific_basis

# def get_svd_num_hankel(data, window_size, n_delay):
#     length = len(x)
#     num_matrix = np.zeros(shape=(length-window_size))
#     for i in range(length-window_size):
#         hankel_matrix = construct_hankel_matrix(data[i:i+window_size], n_delay)
#         _, s, _ = np.linalg.svd(hankel_matrix.T, full_matrices=False)
#         count = np.sum(s>1)
#         num_matrix[i] = count
#     return num_matrix

def construct_hankel_matrix(data, window_size):
    n_samples = len(data)
    hankel_matrix = np.zeros((n_samples - window_size + 1, window_size))

    for i in range(n_samples - window_size + 1):
        hankel_matrix[i, :] = data[i:i+window_size, 0]

    return hankel_matrix