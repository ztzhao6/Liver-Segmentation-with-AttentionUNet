import scipy.io as sio
from skimage import transform
import os
import numpy as np
# if python2.7, listdir need sort


def meanstd_calc(file_path):
    files = os.listdir(file_path)
    sum = sio.loadmat(file_path + '/' + files[0])
    for i in range(1, len(files)):
        data = sio.loadmat(file_path + '/' + files[i])['data']
        sum = np.concatenate([sum, data])
    avg = np.mean(sum)
    std = np.std(sum)
    print(f'Average is: {avg}')
    print(f'Std is: {std}')


def meanstd_process(file_path, out_path, avg, std):
    files = os.listdir(file_path)
    for i in range(0, len(files)):
        mat = sio.loadmat(file_path + '/' + files[i])
        data = mat['data']
        label = mat['label']

        data = data.astype(np.float32)
        data = (data - avg) / std

        out_name = out_path + '/' + str(i) + '.mat'
        sio.savemat(out_name, {'data': data, 'label': label})


def basic_calc(file_path):
    files = os.listdir(file_path)
    for i in range(0, len(files)):
        mat = sio.loadmat(file_path + '/' + files[i])
        # data = mat['data']
        label = mat['label']
        # print(f"data's shape is: {data.shape}")
        print(f"label's shape is: {label.shape}")
        # print(f"data's range is: {np.min(data)} ~ {np.max(data)}")
        print(f"label's range is: {set(label.flatten())}")
        # print(f"data's Iou range is: {np.min(data[label > 0])} ~ {np.max(data[label > 0])}")


def resize_img(file_path, out_path, shape):
    ''' resize n * shape * shape
    resize can upsampling or downsampling'''

    files = os.listdir(file_path)
    for i in range(0, len(files)):
        mat = sio.loadmat(file_path + '/' + files[i])
        data = mat['data']
        label = mat['label']
        # resize前需将label*255，很奇怪data不用吗，问题待解决
        h = data.shape[0]
        data = transform.resize(data, (h, shape, shape), mode='constant')
        label = transform.resize(label, (h, shape, shape), mode='constant')
        label[label >= 0.5] = 1
        label[label < 0.5] = 0

        out_name = out_path + '/' + str(i) + '.mat'
        sio.savemat(out_name, {'data': data, 'label': label})


def rename(file_path, name_len):
    files = os.listdir(file_path)
    for file in files:
        length = len(file.split('.')[0])
        if length < name_len:
            os.rename(file_path + '/' + file, file_path + '/' + '0' * (name_len - length) + file)


def cut_to2D(file_path, out_path):
    files = os.listdir(file_path)

    for file in files:
        mat = sio.loadmat(file_path + file)
        data = mat['data']
        label = mat['label']
        num = data.shape[0]
        for idx in range(0, num):
            data_new = data[idx, :, :]
            label_new = label[idx, :, :]
            out_name = out_path + file.split('.')[0] + '_' + str(idx) + '.mat'
            sio.savemat(out_name, {'data': data_new, 'label': label_new})


def cut_the_first(file_path, out_path, height, stride):
    '''long cut to length; short padding 0 back'''

    files = os.listdir(file_path)
    # name_index = 0

    for file in files:
        name = file.split('.')[0]
        name_index = 0
        mat = sio.loadmat(file_path + '/' + file)
        data = mat['data']
        label = mat['label']

        # resize to 128*128
        # label = label.astype(np.float32)  # important
        #
        # h = data.shape[0]
        # data = transform.resize(data, (h, 128, 128), mode='constant')
        # label = transform.resize(label, (h, 128, 128), mode='constant')
        #
        # label[label >= 0.5] = 1
        # label[label < 0.5] = 0

        ori_height = data.shape[0]
        if ori_height >= height:
            start = 0
            while start + height <= ori_height:
                save_data = data[start:start + height, :, :]
                save_label = label[start:start + height, :, :]
                out_name = out_path + '/' + name + '-' + str(name_index) + '.mat'
                sio.savemat(out_name, {'data': save_data, 'label': save_label})
                name_index += 1
                start += stride
            if start - stride + height < ori_height:
                save_data = data[ori_height - height:ori_height, :, :]
                save_label = label[ori_height - height:ori_height, :, :]
                out_name = out_path + '/' + name + '-' + str(name_index) + '.mat'
                sio.savemat(out_name, {'data': save_data, 'label': save_label})
                name_index += 1
        else:
            add = np.zeros((height - ori_height, 144, 144), dtype=np.float32)
            save_data = np.concatenate([data, add])
            save_label = np.concatenate([label, add])
            out_name = out_path + '/' + name + '-' + str(name_index) + '.mat'
            sio.savemat(out_name, {'data': save_data, 'label': save_label})
            name_index += 1


def cut_three(file_path, out_path, height, square, stride):
    '''add if condition and then not save all pieces'''
    files = os.listdir(file_path)
    name_index = 0

    for i in range(len(files)):
        print(i, name_index)
        mat = sio.loadmat(file_path + '/' + files[i])
        data = mat['data']
        label = mat['label']

        ori_height = data.shape[0]
        ori_square = data.shape[1]
        # default ori_square >= square and square % stride == 0
        start_column = 0
        start_row = 0

        while start_row + square <= ori_square:
            while start_column + square <= ori_square:
                tmp_data = data[:, start_row:start_row + square, start_column:start_column + square]
                tmp_label = label[:, start_row:start_row + square, start_column:start_column + square]

                if ori_height >= height:
                    start_height = 0
                    while start_height + height <= ori_height:
                        save_data = tmp_data[start_height:start_height + height, :, :]
                        save_label = tmp_label[start_height:start_height + height, :, :]
                        if np.sum(save_label) > 0:
                            out_name = out_path + '/' + str(name_index) + '.mat'
                            sio.savemat(out_name, {'data': save_data, 'label': save_label})
                            name_index += 1
                        start_height += stride
                    if start_height - stride + height < ori_height:
                        save_data = tmp_data[ori_height - height:ori_height, :, :]
                        save_label = tmp_label[ori_height - height:ori_height, :, :]
                        if np.sum(save_label) > 0:
                            out_name = out_path + '/' + str(name_index) + '.mat'
                            sio.savemat(out_name, {'data': save_data, 'label': save_label})
                            name_index += 1
                else:
                    add = np.zeros((height - ori_height, square, square), dtype=np.float32)
                    save_data = np.concatenate([tmp_data, add])
                    save_label = np.concatenate([tmp_label, add])
                    if np.sum(save_label) > 0:
                        out_name = out_path + '/' + str(name_index) + '.mat'
                        sio.savemat(out_name, {'data': save_data, 'label': save_label})
                        name_index += 1

                start_column += stride

            start_row += stride
            start_column = 0


def livechallenge_process():
    '''can't use'''
    import nibabel as nib

    LIVER_RIGHT = np.arange(68, 83)
    SPINE_UP = np.r_[np.arange(0, 53), np.arange(68, 83)]

    for i in range(130, 131):
        data_path = 'E:/liver_data_all/LiverChallenge/data/volume-' + str(i) + '.nii'
        label_path = 'E:/liver_data_all/LiverChallenge/mask/segmentation-' + str(i) + '.nii'
        data = nib.load(data_path).get_fdata()
        label = nib.load(label_path).get_data()
        data = data.transpose(2, 0, 1)
        label = label.transpose(2, 0, 1)  # to be n*512*512

        if i in SPINE_UP:
            data = data[:, :, ::-1]
            label = label[:, :, ::-1]
        if i in LIVER_RIGHT:
            data = data[:, ::-1, :]
            label = label[:, ::-1, :]  # reversal

        data[data <= -200] = -200.0
        data[data >= 250] = 250.0   # threshold

        data = (data - (-200.0)) / (250.0 - (-200.0))  # to be [0,1]

        print(data.shape)

        out_name = 'E:/tumor_data/' + str(i) + '.mat'
        sio.savemat(out_name, {'data': data, 'label': label})
