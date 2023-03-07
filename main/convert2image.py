import scipy.io as sio
from Utils_Bashivan import *
import numpy as np
import os


def convertEEG(data, FLag):
    time_len = 1

    #path = r"C:\Users\xiang.shen\Desktop\MEET-main\dataset\CHB-MIT\extractFeature/"

    for cnt in range(1, 2):
        ## Load signal
        if FLag == 1:
            seizure_data = data['seizure']
            no_seizure_data = data['no_seizure']
            length1 = seizure_data.shape[0]
            length2 = no_seizure_data.shape[0]
            feats = np.concatenate((seizure_data, no_seizure_data), axis=0)
            label = np.concatenate((np.zeros((length1,), dtype=int), np.ones((length2,), dtype=int)), axis=0)
            Flag = 0
        else:
            no_seizure_data = data['no_seizure']
            feats = no_seizure_data
            length2 = no_seizure_data.shape[0]
            label = np.ones((length2,), dtype=int)


        #data = np.load(path + 'TEST.npy')

        '''
        data = sio.loadmat(path + 'chb' + str(cnt) + '.mat')
        seizure_data = data['seizure']  # [110, 22, 5]
        no_seizure_data = data['no_seizure']  # [110, 22, 5]
        
        seizure_data = []
        seizure_data = np.array(seizure_data)
        no_seizure_data = data
        length1 = seizure_data.shape[0]
        length2 = no_seizure_data.shape[0]
        #feats = np.concatenate((seizure_data, no_seizure_data), axis=0)
        #label = np.concatenate((np.zeros((length1,), dtype=int), np.ones((length2,), dtype=int)), axis=0)
        feats = no_seizure_data
        label = np.zeros((length2,), dtype=int)
        if flag == 1:
            feats = seizure_data
            label = np.zeros((length,), dtype=int)
            flag = 0
        else:
            feats = np.concatenate((feats, seizure_data), axis=0)
            label = np.concatenate((label, np.zeros((length,), dtype=int)), axis=0
    
        feats = np.concatenate((feats, no_seizure_data), axis=0)
        label = np.concatenate((label, np.ones((length,), dtype=int)), axis=0)
        
        
        '''
        # Reshape
        label = np.expand_dims(label, axis=1)
        # feats:[220,22,5]; label:[220,1]
        feats = np.reshape(feats, (-1,110))      # [851,310]

        locs = sio.loadmat(r'C:\Users\xiang.shen\Desktop\MEET-main\dataset\CHB-MIT/3DLocs.mat')
        locs_3d = locs['locs']
        locs_2d = []

        # Convert to 2D
        for e in locs_3d:
            locs_2d.append(azim_proj(e))
        sio.savemat(r'C:\Users\xiang.shen\Desktop\MEET-main\dataset\CHB-MIT/2DLocs.mat',{"loc":locs_2d})

        images_timewin = np.array(gen_images(np.array(locs_2d), feats, 32, normalize=True))
        images_timewin = np.reshape(images_timewin, (-1,time_len,5,32,32))
        # print(1)
        #sio.savemat(r"C:\Users\xiang.shen\Desktop\MEET-main\dataset\CHB-MIT\convertImage/chb" + str(cnt) + "_1time-step_32@32.mat", {"img":images_timewin, 'label':label})
        sio.savemat(r"C:\Users\xiang.shen\Desktop\MEET-main\dataset\CHB-MIT\convertImage/chbtest.mat", {"img": images_timewin, 'label': label})

