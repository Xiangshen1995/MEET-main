from readEdf import *
from convert2image import *
from testcopy import *
import numpy as np
import os
import pickle
from mne.io import concatenate_raws, read_raw_edf
import matplotlib.pyplot as plt
import argparse
import scipy.io as sio
import mne
import pandas as pd

path_time = r"C:\Users\xiang.shen\Desktop\MEET-main\dataset\CHB-MIT/time.csv"
filepath = r"C:/Users/xiang.shen/Desktop/MEET-main/dataset/CHB-MIT/Rawdata/chb01/"
filename = "chb01_29.edf"
#filename = "chb01_10.edf"
ch = ['FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1', 'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1', 'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2',
      'FP2-F8', 'F8-T8', 'T8-P8-0', 'P8-O2', 'FZ-CZ', 'CZ-PZ', 'P7-T7', 'T7-FT9', 'FT9-FT10', 'FT10-T8']

time = pd.read_csv(path_time, index_col="chb")
sampling = 256
SeizureFlag = 0
f_str = str(os.path.splitext(os.path.splitext(filename)[0])[0])
if f_str in time.index:  # 返回最后一个扩展名
    #[start, end] = list(input('Input start and end').split(','))
    start = time.loc[f_str]['start'] * sampling
    end = time.loc[f_str]['end'] * sampling
    raw = read_raw_edf(filepath + filename, preload=False)
    SeizureFlag = 1
    print('---------seizure and no-seizure data------------')
else:
    raw = read_raw_edf(filepath + filename, preload=False)
    print('----------only no-seizure data------------------')

print(raw)
print(raw.info)
raw.pick_channels(ch)

raw_d, raw_t = raw[:, :]
DEData = dict()
raw_d = raw_d * 1e6
if SeizureFlag == 1:
    seizure_data = raw_d[:, start:end]
    no_seizure_data = np.delete(raw_d, np.s_[start:end], axis=1)
    DE_seizure = decompose(seizure_data)
    DE_no_seizure = decompose(no_seizure_data)
    DEData['seizure'] = DE_seizure
    DEData['no_seizure'] = DE_no_seizure
else:
    no_seizure_data = raw_d
    DE_no_seizure = np.empty([0, 22, 5])
    DE_no_seizure = decompose(no_seizure_data)
    DEData['no_seizure'] = DE_no_seizure

#raw.plot_psd(fmax=50)
#raw.plot(duration=5, n_channels=22)
#plt.show()

# x = sio.loadmat(r"C:\Users\xiang.shen\Desktop\MEET-main\dataset\CHB-MIT\preprocessData/chb01.mat")
#data = x['no_seizure']
#data = data[:, 1:200000]

path_save = r"C:\Users\xiang.shen\Desktop\MEET-main\dataset\CHB-MIT\extractFeature/"
# sio.savemat(path_save + 'test.mat', DE_Characteristics)
with open(path_save + 'DE_data_' + os.path.splitext(filename)[0]+'.pkl', 'wb') as fp:
    pickle.dump(DEData, fp)
    print('dictionary saved successfully to file')

convertEEG(DEData, SeizureFlag)
print('successfully convert to images')

parser = argparse.ArgumentParser()
# parser.add_argument('--num_classes', type=int, default=3)
parser.add_argument('--epochs', type=int, default=1)
parser.add_argument('--batch-size', type=int, default=1000)
parser.add_argument('--lr', type=float, default=0.001)  # 原：0.001
parser.add_argument('--lrf', type=float, default=0.01)

# 执行任务类型
# parser.add_argument('--task', default="WM_S1-15_1time_1000epochs_12depth_16heads_16patch_hhh")
parser.add_argument('--task', default="CHB_S1")
parser.add_argument('--data-path', type=str,
                    default=r"E:\Data\Datasets\CV\flower_photos")
parser.add_argument('--model-name', default='', help='create model name')
parser.add_argument('--weights', type=str, default= r'C:\Users\xiang.shen\Desktop\MEET-main/weights/CHB/CHB_S1/best_model.pth',
                    help='initial weights path')
parser.add_argument('--freeze-layers', type=bool, default=False)  # 原：冻结=True
parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

opt = parser.parse_args()
prediction = CHB_MIT_test_main(opt)

if prediction[0] == 1:
    my_annot = mne.Annotations(onset=0, duration=4, description='no seizure')
else:
    my_annot = mne.Annotations(onset=0, duration=4, description='seizure')

for i in range(1, len(prediction)):
    if prediction[i] == 1:
        new_annot = mne.Annotations(onset=i * 4, duration=4, description='no seizure' )
    else:
        new_annot = mne.Annotations(onset=i * 4, duration=4, description='seizure')
    my_annot = my_annot + new_annot
raw.set_annotations(my_annot)

raw.plot(start=1, duration=10, n_channels=22, scalings="auto", remove_dc=True)
plt.show()

