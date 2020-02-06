from torch.utils.data import Dataset
import numpy as np
import scipy.io as sio
import nrrd
import os


class NrrdReader3D(Dataset):
    def __init__(self, data_path, label_path=None, test=False, transform=None):
        self.data_path = data_path
        self.files = os.listdir(data_path)
        self.files.sort()
        self.transform = transform
        self.test = test
        if not self.test:
            self.label_path = label_path

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        file_name = self.files[index]
        data, _ = nrrd.read(self.data_path + file_name)
        data = data.astype(np.float32)
        data = data[np.newaxis, ...]
        if not self.test:
            label, _ = nrrd.read(self.label_path + file_name)
            sample = {'data': data, 'label': label}
        else:
            sample = {'data': data}
        return sample
