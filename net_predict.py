from __future__ import print_function
import os
from net_framework import AttentionUNet2D
import torch
from torch.utils.data import DataLoader
from data_load import NrrdReader3D
import numpy as np
from os import listdir
import nrrd


def result_net(model, val_file_path, save_file_path):
    if torch.cuda.is_available():
        model = model.cuda()

    val_dataset = NrrdReader3D(val_file_path, test=True)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    file_name = listdir(val_file_path)
    file_name.sort()
    model.eval()

    print('Saving mat file...')
    print('Please wait')
    for batch_idx, sample_batched in enumerate(val_dataloader):
        if torch.cuda.is_available():
            data_var = sample_batched['data'].cuda()
        else:
            data_var = sample_batched['data']

        # compute output
        output = model(data_var)
        output = output.data.cpu().numpy()

        output = np.squeeze(output)
        (classes, k1, k2) = output.shape
        #(classes, k1, k2, k3) = output.shape

        result = np.zeros((k1, k2), np.float32)
        for i in range(0, k1):
            for j in range(0, k2):
                #for k in range(0, k3):
                result[i][j] = np.argmax(output[:, i, j])
        nrrd.write(os.path.join(save_file_path, file_name[batch_idx]), result)


if __name__ == '__main__':
    net = AttentionUNet2D(n_channels=1, n_classes=2)
    # , map_location={'cuda:1': 'cuda:0'}
    weight_path = 'E:/code/radiologist/45_2d.pkl'
    net.load_state_dict(torch.load(weight_path))
    result_net(net, val_file_path='D:/liver_ct_2/segment/data_2d/',
               save_file_path='D:/liver_ct_2/segment/result_2d')


