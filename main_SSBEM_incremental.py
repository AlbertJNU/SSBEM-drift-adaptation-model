import os
import pandas as pd
import torch

from data_factory.data_loader import get_specific_basis_indataloader
from model.SSBEM import SSBEM_B
from model.NBeats import NBEATS
from model_update import Our_method
from data_factory.data_stream import FileStream
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import matplotlib.pyplot as plt
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['Times New Roman']
mpl.rcParams['axes.unicode_minus'] = False

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def construct_hankel_matrix(data, window_size):
    n_samples = len(data)
    hankel_matrix = np.zeros((n_samples - window_size + 1, window_size))

    for i in range(n_samples - window_size + 1):
        hankel_matrix[i, :] = data[i:i+window_size, 0]

    return hankel_matrix

#Comparison model setting
#NBEATS: model = 'NBEATS'; model_load_path = 'checkpoints_NBEATS'; recursive_function = False
#SSBEM_B: model = 'SSBEM'; model_load_path = 'checkpoints_SSBEM_B'; recursive_function = False
#SSBEM_BR: model = 'SSBEM'; model_load_path = 'checkpoints_SSBEM_BR'; recursive_function = True
#SSBEM_BRD: model = 'SSBEM'; model_load_path = 'checkpoints_SSBEM_BRD'; recursive_function = True
#SSBEM_BRN: model = 'SSBEM'; model_load_path = 'checkpoints_SSBEM_BRN'; recursive_function = True
#SSBEM_BRS: model = 'SSBEM'; model_load_path = 'checkpoints_SSBEM_BRS'; recursive_function = True
model = 'SSBEM'
model_load_path = 'checkpoints_SSBEM_BRS'
recursive_function = True

#setup a data stream
win_size = 5
stream = FileStream(filepath="./dataset/data/incredrift.csv",
                    target_idx=-1, n_targets=1, cat_features=None, allow_nan=False, window_size=win_size)

output_size = 1
input_size = 2
n_blocks = [1, 1, 1]
n_layers = 2
block_hidden_size = 64
dropout = 0.1
batch_normalization = False
use_predict_covariate = True
n_delays = 10
svd_low_rank = 6
step_size = 1
prediction_length = 2
dataset = 'Online_Initialization'
data_path = './dataset/' + dataset

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

specific_basis = get_specific_basis_indataloader(
                                            n_delays=n_delays,
                                            data_path=data_path,
                                            win_size=win_size,
                                            step=step_size,
                                            prediction_length=prediction_length,
                                            mode='train',
                                            dataset=dataset)

#setup retrain and recursive variables
n_samples = 0
max_samples = 2993
collect_data = 0
dataset = 'Online_Initialization'
modelset = 'Online_Initialization'
feature_dataset = []
label_dataset = []
y_hat_list = []
y_true_list = []

if model == 'SSBEM':
    base_estimator = SSBEM_B(
        specific_basis=specific_basis,
        output_size=output_size,
        context_length=win_size,
        covariate_size=input_size,
        prediction_length=prediction_length,
        n_blocks=n_blocks,
        n_layers=n_layers,
        hidden_size=block_hidden_size,
        dropout=dropout,
        batch_normalization=batch_normalization,
        use_predict_covariate=use_predict_covariate,
        n_delays=n_delays,
        svd_low_rank=svd_low_rank,
    )
else:
    base_estimator = NBEATS(
        output_size=output_size,
        context_length=win_size,
        covariate_size=input_size,
        prediction_length=prediction_length,
        n_blocks=n_blocks,
        n_layers=n_layers,
        hidden_size=block_hidden_size,
        dropout=dropout,
        batch_normalization=batch_normalization,
    )

online_regressor = Our_method.Online_regressor(base_estimator, model_load_path, device=device)

while n_samples < max_samples and stream.has_more_samples():
    #this data_y doesn't contain the ground truth data, it represents the label values of the previous few moments
    data_x, data_y, index = stream.next_sample()
    print(index)

    if model == 'NBEATS':
        if index >= 700:
            print('predict')
            base_estimator = NBEATS(
                output_size=output_size,
                context_length=win_size,
                covariate_size=input_size,
                prediction_length=prediction_length,
                n_blocks=n_blocks,
                n_layers=n_layers,
                hidden_size=block_hidden_size,
                dropout=dropout,
                batch_normalization=batch_normalization,
            )
            online_regressor = Our_method.Online_regressor(base_estimator, model_load_path, device=device)

            if index == 1300:
                modelset = 'Online_Recursive_0'
                dataset = 'Online_Recursive_0'
            if index == 2300:
                modelset = 'Online_Recursive_1'
                dataset = 'Online_Recursive_1'

            y_predict = \
            online_regressor.predict(data_x, data_y, modelset=modelset, data_path='./dataset/' + dataset, win_size=win_size,
                                     step=1, prediction_length=2)[0][0, 0, 0].item()
            true_label = stream.get_sample(index + 1)[1][-1]
            y_hat_list.append(y_predict)
            y_true_list.append(true_label)
    else:
        if index >= 1000 and index <= 1300:
            print('Recursive update process')
            collect_data = 1
            if recursive_function == True:
                if index < 1100:
                    win_recursive_size = 50
                    lr = -0.4
                if index >= 1100 and index < 1300:
                    win_recursive_size = 50
                    lr = -0.01
            else:
                if index < 1050:
                    win_recursive_size = 50
                    lr = 0
                if index >= 1050 and index < 1200:
                    win_recursive_size = 50
                    lr = 0

            feature_dataset.append(stream.get_sample(index)[0][-3, :])
            label_dataset.append(stream.get_sample(index)[1][-1])
            y = data_y[:, None]
            x_1, y_1 = stream.get_sample(index - win_recursive_size - 1)
            y_1 = y_1[:, None]
            x_pt, y_pt = stream.get_sample(index - 1)
            y_pt = y_pt[:, None]
            y_window = stream.get_window_label(index, win_recursive_size)
            H_new = construct_hankel_matrix(y_window[:, None], n_delays).T
            W_old = base_estimator.model_specific.Weight_layer.weight.data.cpu().detach().numpy()
            B_old = base_estimator.model_specific.Weight_layer.bias.data.cpu().detach().numpy()[:, None]
            W_old_all = np.hstack([W_old, B_old])
            delay_num = n_delays
            x_pt = torch.as_tensor(x_pt, dtype=torch.float32, device=device).unsqueeze(0)
            y_pt = torch.as_tensor(y_pt, dtype=torch.float32, device=device).unsqueeze(0)
            x_1 = torch.as_tensor(x_1, dtype=torch.float32, device=device).unsqueeze(0)
            y_1 = torch.as_tensor(y_1, dtype=torch.float32, device=device).unsqueeze(0)
            y = torch.as_tensor(y, dtype=torch.float32, device=device).unsqueeze(0)
            if index == 1000:
                y_window_old = stream.get_window_label(index - 1, win_recursive_size)
                y_window_old_2d = y_window_old[:, None]
                H_old = construct_hankel_matrix(y_window_old_2d, n_delays).T
                Xpt = []
                for i in range(win_recursive_size):
                    x_i, y_i = stream.get_sample(index-2-i)
                    A_i = online_regressor.predict(x_i, y_i, modelset=dataset, data_path='./dataset/' + dataset, win_size=win_size,
                                             step=1, prediction_length=2)[3].squeeze()[0, :]
                    A_i = torch.cat([A_i, torch.as_tensor([1], dtype=torch.float32, device=device)])
                    Xpt.append(A_i.cpu().detach().numpy())
                Xpt = np.array(Xpt)
                Q_old = np.linalg.inv((Xpt).T @ (Xpt))
                C_old = H_old @ H_old.T
                dataset = 'Online_Initialization'
                modelset = 'Online_Initialization'
            else:
                y_window_old = stream.get_window_label(index - 1, win_recursive_size)
                y_window_old_2d = y_window_old[:, None]
                H_old = construct_hankel_matrix(y_window_old_2d, n_delays).T
                Q_old = Q_new
                C_old = 0
                dataset = 'Online_Initialization'
                modelset = 'Online_Initialization'
            block_num = len(n_blocks)
            Q_new, dataset, modelset = online_regressor.recursive_update(x_1, y_1, x_pt, y_pt, y, W_old_all, Q_old, lr, dataset)

            if index == 1300:
                print('finished collect data in recursive process')
                feature_dataset = np.array(feature_dataset)
                label_dataset = np.array(label_dataset)
                Original_feature, Original_label = stream.get_sample(win_size)
                feature_dataset = np.block([[Original_feature[:-2, :]], [feature_dataset]])
                label_dataset = np.block([Original_label, label_dataset])
                train = pd.DataFrame(feature_dataset[100:-100])
                train_label = pd.DataFrame(label_dataset[100:-100])
                valid = pd.DataFrame(feature_dataset[-100:])
                valid_label = pd.DataFrame(label_dataset[-100:])
                dataset = 'Online_Recursive_0'
                folder = str(dataset)
                path = os.path.join('dataset', folder)
                if not os.path.exists(path):
                    os.makedirs(path)
                train.to_csv(os.path.join('dataset', str(dataset), 'initialization_train.csv'))
                train_label.to_csv(os.path.join('dataset', str(dataset), 'initialization_train_label.csv'))
                valid.to_csv(os.path.join('dataset', str(dataset), 'initialization_valid.csv'))
                valid_label.to_csv(os.path.join('dataset', str(dataset), 'initialization_valid_label.csv'))
                feature_dataset = []
                label_dataset = []

                specific_basis = get_specific_basis_indataloader(
                    n_delays=n_delays,
                    data_path='./dataset/' + dataset,
                    win_size=win_size,
                    step=step_size,
                    prediction_length=prediction_length,
                    mode='train',
                    dataset=dataset)


        elif index >= 2000 and index <= 2300:
            print('Recursive update process')
            collect_data = 1

            if recursive_function == True:
                if index < 2100:
                    win_recursive_size = 100
                    lr = 0.01
                if index >= 2100 and index < 2300:
                    win_recursive_size = 100
                    lr = 0.01
            else:
                if index < 2100:
                    win_recursive_size = 100
                    lr = 0
                if index >= 2100 and index < 2300:
                    win_recursive_size = 100
                    lr = 0

            feature_dataset.append(stream.get_sample(index)[0][-3, :])
            label_dataset.append(stream.get_sample(index)[1][-1])
            y = data_y[:, None]
            x_1, y_1 = stream.get_sample(index - win_recursive_size - 1)
            y_1 = y_1[:, None]
            x_pt, y_pt = stream.get_sample(index - 1)
            y_pt = y_pt[:, None]
            y_window = stream.get_window_label(index, win_recursive_size)
            H_new = construct_hankel_matrix(y_window[:, None], n_delays).T
            W_old = base_estimator.model_specific.Weight_layer.weight.data.cpu().detach().numpy()
            B_old = base_estimator.model_specific.Weight_layer.bias.data.cpu().detach().numpy()[:, None]
            W_old_all = np.hstack([W_old, B_old])
            delay_num = n_delays
            x_pt = torch.as_tensor(x_pt, dtype=torch.float32, device=device).unsqueeze(0)
            y_pt = torch.as_tensor(y_pt, dtype=torch.float32, device=device).unsqueeze(0)
            x_1 = torch.as_tensor(x_1, dtype=torch.float32, device=device).unsqueeze(0)
            y_1 = torch.as_tensor(y_1, dtype=torch.float32, device=device).unsqueeze(0)
            y = torch.as_tensor(y, dtype=torch.float32, device=device).unsqueeze(0)
            if index == 2000:
                y_window_old = stream.get_window_label(index - 1, win_recursive_size)
                y_window_old_2d = y_window_old[:, None]
                H_old = construct_hankel_matrix(y_window_old_2d, n_delays).T
                Xpt = []
                for i in range(win_recursive_size):
                    x_i, y_i = stream.get_sample(index-2-i)
                    A_i = online_regressor.predict(x_i, y_i, modelset=dataset, data_path='./dataset/' + dataset, win_size=win_size,
                                             step=1, prediction_length=2)[3].squeeze()[0, :]
                    A_i = torch.cat([A_i, torch.as_tensor([1], dtype=torch.float32, device=device)])
                    Xpt.append(A_i.cpu().detach().numpy())
                Xpt = np.array(Xpt)
                Q_old = np.linalg.inv((Xpt).T @ (Xpt))
                C_old = H_old @ H_old.T
                basis_old = specific_basis
                dataset = 'Online_Recursive_0'
                modelset = 'Online_Recursive_0'
            else:
                y_window_old = stream.get_window_label(index - 1, win_recursive_size)
                y_window_old_2d = y_window_old[:, None]
                H_old = construct_hankel_matrix(y_window_old_2d, n_delays).T
                Q_old = Q_new
                C_old = 0
                dataset = 'Online_Recursive_0'
                modelset = 'Online_Recursive_0'

            block_num = len(n_blocks)
            Q_new, dataset, modelset = online_regressor.recursive_update(x_1, y_1, x_pt, y_pt, y, W_old_all, Q_old, lr, dataset)

            if index == 2300:
                print('finished collect data in recursive process')
                feature_dataset = np.array(feature_dataset)
                label_dataset = np.array(label_dataset)
                Original_feature, Original_label = stream.get_sample(win_size)
                feature_dataset = np.block([[Original_feature[:-2, :]], [feature_dataset]])
                label_dataset = np.block([Original_label, label_dataset])
                train = pd.DataFrame(feature_dataset[100:-100])
                train_label = pd.DataFrame(label_dataset[100:-100])
                valid = pd.DataFrame(feature_dataset[-100:])
                valid_label = pd.DataFrame(label_dataset[-100:])
                dataset = 'Online_Recursive_1'
                folder = str(dataset)
                path = os.path.join('dataset', folder)
                if not os.path.exists(path):
                    os.makedirs(path)
                train.to_csv(os.path.join('dataset', str(dataset), 'initialization_train.csv'))
                train_label.to_csv(os.path.join('dataset', str(dataset), 'initialization_train_label.csv'))
                valid.to_csv(os.path.join('dataset', str(dataset), 'initialization_valid.csv'))
                valid_label.to_csv(os.path.join('dataset', str(dataset), 'initialization_valid_label.csv'))
                feature_dataset = []
                label_dataset = []

                specific_basis = get_specific_basis_indataloader(
                    n_delays=n_delays,
                    data_path='./dataset/' + dataset,
                    win_size=win_size,
                    step=step_size,
                    prediction_length=prediction_length,
                    mode='train',
                    dataset=dataset)

        if index >= 700:
            print('predict')
            base_estimator = SSBEM_B(
                    specific_basis=specific_basis,
                    output_size=output_size,
                    context_length=win_size,
                    covariate_size=input_size,
                    prediction_length=prediction_length,
                    n_blocks=n_blocks,
                    n_layers=n_layers,
                    hidden_size=block_hidden_size,
                    dropout=dropout,
                    batch_normalization=batch_normalization,
                    use_predict_covariate=use_predict_covariate,
                    n_delays=n_delays,
                    svd_low_rank=svd_low_rank,
                )

            online_regressor = Our_method.Online_regressor(base_estimator, model_load_path, device=device)

            if index == 1300:
                modelset = 'Online_Recursive_0'
                dataset = 'Online_Recursive_0'

            if index == 2300:
                modelset = 'Online_Recursive_1'
                dataset = 'Online_Recursive_1'

            y_predict = online_regressor.predict(data_x, data_y, modelset=modelset, data_path='./dataset/' + dataset, win_size=win_size,
                                                 step=1, prediction_length=2)[0][0, 0, 0].item()
            true_label = stream.get_sample(index + 1)[1][-1]
            y_hat_list.append(y_predict)
            y_true_list.append(true_label)

    n_samples = n_samples + 1

#display results (visualization)
y_hat = np.array(y_hat_list)
y_true = np.array(y_true_list)
sample_num = len(y_hat)

# Prepare the figure
plt.close('all')
plt.ioff()
figprops = dict(figsize=(14, 7), dpi=100)
fig, ax = plt.subplots(**figprops)
all_t = np.arange(700, 700+sample_num, 1)
ax.plot(all_t, y_true[0:], '--', linewidth=1.5)
ax.plot(all_t, y_hat[0:], '-', linewidth=1.5)

y_all = np.column_stack((y_true[0:], y_hat[0:]))
y_all = pd.DataFrame(y_all, columns=['true', 'predict'])

print(f"MAE: {mean_absolute_error(y_true[0:], y_hat[0:])}")
print(f"MSE: {mean_squared_error(y_true[0:], y_hat[0:])}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_true[0:], y_hat[0:]))}")
print(f"决定系数R2: {r2_score(y_true[0:], y_hat[0:])}")

font = {'family': 'sans-serif',
        'color': 'black',
        'weight': 'normal',
        'size': 24
        }

ax.set_title('')
ax.set_xlabel('Sample', fontdict=font)
ax.set_ylabel('Value', fontdict=font)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(True)
ax.spines['bottom'].set_visible(True)
ax.tick_params(axis='both', which='major', labelsize=24)
plt.show()


