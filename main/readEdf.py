import numpy as np
from mne.io import concatenate_raws, read_raw_edf
import matplotlib.pyplot as plt
import matplotlib
import mne
import math
import warnings
from scipy.io import loadmat
from scipy.signal import butter, lfilter
from scipy.stats import differential_entropy
import scipy.io as sio
import scipy.signal

warnings.filterwarnings("ignore")

def compute_DE(S, len):

    temp = S[len[0]-1:len[1]]
    return np.sum(temp) / (len[1] - len[0])


def decompose(data):
    frequency = 256
    data = np.transpose(data)
    samples = data.shape[0]
    channels = data.shape[1]
    timewindow = frequency * 4

    # 100个采样点计算一个微分熵
    num_sample = int(samples / timewindow)

    bands = 5
    # 微分熵特征
    DE_Characteristics = np.empty([num_sample, channels, bands])

    temp_de = np.empty([0, num_sample])

    for channel in range(channels):

        trail_single = data[:, channel]

        # bandpass for five bands
        # delta (1-4 Hz), theta (4-8 Hz), alpha (8-14 Hz), beta (14-31 Hz), gamma (31-50Hz).
        DE_Delta = np.zeros(shape=[0], dtype=float)
        DE_Theta = np.zeros(shape=[0], dtype=float)
        DE_alpha = np.zeros(shape=[0], dtype=float)
        DE_beta = np.zeros(shape=[0], dtype=float)
        DE_gamma = np.zeros(shape=[0], dtype=float)

        # 依次计算5个频带的微分熵
        for index in range(num_sample):
            tempdata = trail_single[index * timewindow: (index + 1) * timewindow]
            #(f, S) = scipy.signal.periodogram(tempdata, 1024, scaling='density')
            tmp = np.fft.fft(tempdata, 1024)
            tmp = tmp[0:513]
            S = (1/(256*1024))*(np.abs(tmp)**2)
            S[1:-1] = 2 * S[1:-1]
            DE_Delta = np.append(DE_Delta, compute_DE(S, [4, 12]))
            DE_Theta = np.append(DE_Theta, compute_DE(S, [16, 28]))
            DE_alpha = np.append(DE_alpha, compute_DE(S, [32, 52]))
            DE_beta  = np.append(DE_beta , compute_DE(S, [56, 120]))
            DE_gamma = np.append(DE_gamma, compute_DE(S, [124, 200]))

        temp_de = np.vstack([temp_de, DE_Delta])
        temp_de = np.vstack([temp_de, DE_Theta])
        temp_de = np.vstack([temp_de, DE_alpha])
        temp_de = np.vstack([temp_de, DE_beta])
        temp_de = np.vstack([temp_de, DE_gamma])

    temp_trail_de = temp_de.reshape(-1, 5, num_sample)
    print("trial_DE shape", DE_Characteristics.shape)
    temp_trail_de = temp_trail_de.transpose([2, 0, 1])
    DE_Characteristics = np.vstack([temp_trail_de])

    for i in range(0, np.size(DE_Characteristics, 0)):
        temp = DE_Characteristics[i]
        DE_Characteristics[i] = temp * np.log(temp)

    return DE_Characteristics
