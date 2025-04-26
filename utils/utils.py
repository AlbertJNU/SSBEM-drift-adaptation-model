import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from pycwt import wavelet
from torch.autograd import Variable
import numpy as np


def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)


def mkdir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def default_device() -> torch.device:
    """
    PyTorch default device is GPU when available, CPU otherwise.

    :return: Default device.
    """
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def to_tensor(array: np.ndarray) -> torch.Tensor:
    """
    Convert numpy array to tensor on default device.

    :param array: Numpy array to convert.
    :return: PyTorch tensor on default device.
    """
    return torch.tensor(array, dtype=torch.float32).clone().detach().to(default_device())

def get_common_scale(data, t0, dt, wavelet_type):
    '''
    从data中返回common频率/scale
    :param data: input time series data
    :param t0: start time point
    :param dt: time interval
    :param wavelet_type: wavelet type
    :return: common scale in all training data
    '''
    N = data.size
    t = np.arange(0, N) * dt + t0

    # 将趋势项去掉
    p = np.polyfit(t-t0, data, 1)
    data_notrend = data - np.polyval(p, t - t0)
    std = data_notrend.std()
    var = std ** 2
    data_norm = data_notrend / std

    mother = wavelet_type
    s0 = 2 * dt # Starting scale, in this case 2 * dt
    dj = 1 / 12 # Twelve sub-octaves per octaves
    J = 7 / dj # Seven powers of two with dj sub-octaves
    alpha, _, _ = wavelet.ar1(data)  # Lag-1 autocorrelation for red noise, 利用red noise求取置信度

    wave, scales, freqs, coi, fft, fftfreqs = wavelet.cwt(data_norm, dt, dj, s0, J,
                                                          mother)
    power = (np.abs(wave)) ** 2
    fft_power = np.abs(fft) ** 2
    period = 1 / freqs
    signif, fft_theor = wavelet.significance(1.0, dt, scales, 0, alpha,
                                             significance_level=0.95,
                                             wavelet=mother)
    sig95 = np.ones([1, N]) * signif[:, None]
    sig95 = power / sig95

    glbl_power = power.mean(axis=1)
    dof = N - scales  # Correction for padding at edges
    glbl_signif, tmp = wavelet.significance(var, dt, scales, 1, alpha,
                                            significance_level=0.95, dof=dof,
                                            wavelet=mother)

    glbl_power_new = var * glbl_power
    # smaller是较小的交点，bigger是较大的交点
    equal_frequency_smaller = np.array([i for i in range(len(glbl_power_new) - 1) if
                               (glbl_power_new[i] < glbl_signif[i]) and (glbl_power_new[i + 1] > glbl_signif[i + 1])])[:,None]
    equal_frequency_bigger = np.array([i + 1 for i in range(len(glbl_power_new) - 1) if
                              (glbl_power_new[i] > glbl_signif[i]) and (glbl_power_new[i + 1] < glbl_signif[i + 1])])[:,None]

    common_scale = np.concatenate([equal_frequency_smaller, equal_frequency_bigger], axis=1)

    return common_scale

def get_common_data(datapart, t0, dt, wavelet_type, common_scale):
    '''
    从data中返回common频率/scale
    :param datapart: data batch
    :param t0: start time point
    :param dt: time interval
    :param wavelet_type: wavelet type
    :return: common data in the data batch
    '''
    batch, N, _ = datapart.shape
    N = datapart.shape[1]
    t = np.arange(0, N) * dt +t0

    datapart_numpy = datapart.cpu().data.numpy()

    data_after_common_extraction = torch.ones_like(datapart)

    for i in range(batch):
        # 将趋势项去掉
        p = np.polyfit(t-t0, datapart_numpy[i, :, 0], 1) #todo：数据形式是batch*win_size*1, 导致无法正常运行，如何考虑batch？
        data_notrend = datapart_numpy[i, :, 0] - np.polyval(p, t - t0)
        std = data_notrend.std()
        var = std ** 2
        data_norm = data_notrend / std

        mother = wavelet_type
        s0 = 2 * dt # Starting scale, in this case 2 * dt
        dj = 1 / 12 # Twelve sub-octaves per octaves
        J = 7 / dj # Seven powers of two with dj sub-octaves

        wave, scales, freqs, coi, fft, fftfreqs = wavelet.cwt(data_norm, dt, dj, s0, J,
                                                              mother)
        data_common = 0

        for j in range(common_scale.shape[0]):
            iwave_part = wavelet.icwt(wave[common_scale[j, 0]:common_scale[j, 1], :],
                                     scales[common_scale[j, 0]:common_scale[j, 1]], dt, dj, mother) * std

            data_common = data_common + iwave_part

        data_common = torch.tensor(data_common, device=datapart.device)[:, None]
        data_after_common_extraction[i, :, :] = np.real(data_common)

    return data_after_common_extraction

def getModelSize(model):
    param_size = 0
    param_sum = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
        param_sum += param.nelement()
    buffer_size = 0
    buffer_sum = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
        buffer_sum += buffer.nelement()
    all_size = (param_size + buffer_size) / 1024 / 1024
    print('模型总大小为：{:.3f}MB'.format(all_size))
    return (param_size, param_sum, buffer_size, buffer_sum, all_size)