import os
from net_framework import AttentionUNet2D
from torch.utils.data import DataLoader
import torch
from data_load import MatReader3D
import numpy as np
import scipy.io as sio
import nrrd
import nibabel as nib

ground_path = 'D:/MedicalProject/kindey_kits/kits19_data/label/val/'
predict_path_1 = 'D:/MedicalProject/kindey_kits/kits_exper/experment/experment_2/result_3d/'
predict_path_2 = 'D:/MedicalProject/kindey_kits/kits_exper/experment/experment_1/second/result_3d/'
ground_names = os.listdir(ground_path)

classes = 3
epsilon = 1.0e-6
dice_scores = np.zeros(classes, dtype=np.float32)

for name in ground_names:
    # ground_mat = sio.loadmat(ground_path + name)['label']
    # ground_mat[ground_mat > 0] = 1
    ground_mat = nib.load(ground_path + name).get_data()
    # ground_mat[ground_mat > 0] = 1
    predict_mat_1, temp = nrrd.read(predict_path_1 + name.split('_')[0] + '.nrrd')
    # predict_mat_2, temp = nrrd.read(predict_path_2 + name.split('_')[0] + '.nrrd')
    # predict_mat, temp_data = nrrd.read(predict_path_1 + name.split('_')[0] + '.nrrd')
    # predict_mat = predict_mat_1 + predict_mat_2

    for class_id in range(0, classes):
        label_id = np.array(ground_mat == class_id, dtype=np.float32).flatten()
        result_id = np.array(predict_mat_1 == class_id, dtype=np.float32).flatten()
        dice = 2.0 * np.sum(label_id * result_id) / (np.sum(label_id) + np.sum(result_id) + epsilon)

        dice_scores[class_id] = dice
    # print(f'{name}-{dice_scores[1]}-{dice_scores[2]}')
    print(f'{name}-{dice_scores[1]}-{dice_scores[2]}')

