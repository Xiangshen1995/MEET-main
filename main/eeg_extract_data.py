from re import sub
from mne import Epochs, pick_types, events_from_annotations
from mne.io import concatenate_raws
from mne.io import read_raw_edf
from mne.datasets import eegbci
import mne
import numpy as np
import pandas as pd
import glob 
import numpy as np
import os
from scipy import signal, fft
import matplotlib.pyplot as plt
import scipy.io as sio

#患者发病发病起止时间表
path_time = r"C:\Users\xiang.shen\Desktop\MEET-main\dataset\CHB-MIT/time.csv"
file_dir = r"C:\Users\xiang.shen\Desktop\MEET-main\dataset\CHB-MIT\rawData/"
path_save = r"C:\Users\xiang.shen\Desktop\MEET-main\dataset\CHB-MIT\preprocessData/"
# 选择患者共有的通道
ch = ['FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1', 'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1', 'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2', 
  'FP2-F8', 'F8-T8', 'T8-P8-0', 'P8-O2', 'FZ-CZ', 'CZ-PZ', 'P7-T7', 'T7-FT9', 'FT9-FT10', 'FT10-T8']
sbj = ['chb01', 'chb02', 'chb03', 'chb04', 'chb05', 'chb06', 'chb07', 'chb08', 'chb09', 'chb10']
time = pd.read_csv(path_time, index_col="chb")
sampling = 256

for sbj_num in range(10):
    subject = sbj[sbj_num]
    files = sorted(os.listdir(file_dir + subject))

    all_seizure_data = []
    raws_d = []
    for file in files:
        if os.path.splitext(file)[1] == '.edf':   # 返回最后一个扩展名
            f = os.path.splitext(file)[0]
            f_str = str(os.path.splitext(os.path.splitext(file)[0])[0])
            if f_str in time.index:
                # 提取对应文件的raw数据
                raw = mne.io.read_raw_edf(file_dir + subject + "/" + file, preload=True,verbose=False)
                raw.pick_channels(ch)
                raw_d, raw_t = raw[:,:]

                # 截取出 发作期间数据
                seizure_num = time.loc[f_str].size / 2
                if int(seizure_num) > 1:
                    for num in range(int(seizure_num)-1, -1, -1):   # [2,-1)
                        start = time.loc[f_str]['start'][num] * sampling
                        end = time.loc[f_str]['end'][num] * sampling
                        seizure_data = raw_d[:,start:end]
                        if len(all_seizure_data) == 0:
                            all_seizure_data = seizure_data
                        else:
                            all_seizure_data = np.concatenate((all_seizure_data, seizure_data), axis=1)
                        raw_d = np.delete(raw_d, np.s_[start:end], axis=1)  # axis=1 按列删除
                else:
                    start = time.loc[f_str]['start'] * sampling
                    end = time.loc[f_str]['end'] * sampling
                    seizure_data = raw_d[:,start:end]
                    if len(all_seizure_data) == 0:
                            all_seizure_data = seizure_data
                    else:
                        all_seizure_data = np.concatenate((all_seizure_data, seizure_data), axis=1)
                    raw_d = np.delete(raw_d, np.s_[start:end], axis=1)  # axis=1 按列删除
                
                if len(raws_d) == 0:
                    raws_d = raw_d
                else:
                    raws_d = np.concatenate((raws_d, raw_d), axis=1)
                # raw.filter(0.1,50.,method='iir')
                # raws = concatenate_raws([raws, raw])
                # raws_d, raw_t = raws[:,:]
    # d, t = raws[:,:]
    normal_data = raws_d*1e6
    all_seizure_data = all_seizure_data * 1e6
    Normal_data = np.array(normal_data)
    All_seizure_data = np.array(all_seizure_data)
    # np.save(path_save+"/"+file_dir+".npy",normal_data)
    sio.savemat(path_save + subject + '.mat', {'no_seizure': Normal_data, 'seizure': All_seizure_data})
