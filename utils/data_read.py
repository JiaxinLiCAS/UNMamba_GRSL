import os
import numpy as np
import scipy.io as sio
import math


class data_get_func():
    def __init__(self, dataname, file_path):
        if dataname == 'Jasper Ridge':
            mat_hsi = sio.loadmat(os.path.join(file_path, 'jasperRidge2_R198.mat'))
            mat_gt = sio.loadmat(os.path.join(file_path, 'GroundTruth.mat'))
            self.data_hsi = (mat_hsi['Y'] - mat_hsi['Y'].min()) / (mat_hsi['Y'].max() - mat_hsi['Y'].min())
            self.data_aban = mat_gt['A']
            self.data_endm = mat_gt['M']
            self.num_cols = self.num_rows = int(math.sqrt(self.data_hsi.shape[1]))
            self.num_bands = self.data_hsi.shape[0]
            self.num_endm = self.data_endm.shape[1]

        elif dataname == 'Urban':
            mat_hsi = sio.loadmat(os.path.join(file_path, 'Urban_R162.mat'))
            mat_gt = sio.loadmat(os.path.join(file_path, 'end4_groundTruth.mat'))
            self.data_hsi = (mat_hsi['Y'] - mat_hsi['Y'].min()) / (mat_hsi['Y'].max() - mat_hsi['Y'].min())
            self.data_aban = mat_gt['A']
            self.data_endm = mat_gt['M']
            self.num_cols = self.num_rows = int(math.sqrt(self.data_hsi.shape[1]))
            self.num_bands = self.data_hsi.shape[0]
            self.num_endm = self.data_endm.shape[1]

        elif dataname == 'Apex':
            mat_hsi = sio.loadmat(os.path.join(file_path, 'apex_dataset.mat'))
            self.data_hsi = mat_hsi['Y']
            self.data_aban = mat_hsi['A']
            self.data_endm = mat_hsi['M']
            self.num_cols = self.num_rows = int(math.sqrt(self.data_hsi.shape[1]))
            self.num_bands = self.data_hsi.shape[0]
            self.num_endm = self.data_endm.shape[1]

    def get_hsi_mean(self):
        hsi_mean = np.mean(self.data_hsi, axis=1)
        if len(hsi_mean.shape) == 2:
            hsi_mean = np.mean(hsi_mean, axis=0)
        hsi_mean = np.repeat(hsi_mean, self.num_endm).reshape(-1, self.num_endm).T
        return hsi_mean


