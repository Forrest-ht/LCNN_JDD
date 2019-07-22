from __future__ import division
import os
import time
import scipy.signal
import numpy as np
import h5py
from .ops import *
from .utils import *


def prepare_data(dataset, patch_size, stride, aug_times=1):
    print('process training data')
    imglist = loadimglist(dataset)
    cnt = 0
    for p in imglist:
        img = cv2.imread(p)
        img_ori = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        aug_idx = list(range(8))
        random.shuffle(aug_idx)
        for aug in range(aug_times):
            if aug_times>1:
                img = data_augmentation(img_ori, aug_idx[aug])
            if cnt == 0:
                out = im2patch(img, stride=stride, win=patch_size)
            else:
                tmp = im2patch(img, stride=stride, win=patch_size)
                out = np.concatenate([out, tmp], 3)
            cnt += 1
    return np.transpose(np.float32(out), (3, 0, 1, 2))


def prepare_idx(dataset, patch_size, stride, aug_times=1):
    print('process training data')
    imglist = loadimglist(dataset)
    data_list = []
    cnt = 0
    error_list = []
    for p in imglist:
        img = cv2.imread(p)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        patchs = im2patch(img, stride=stride, win=patch_size)
        for i in range(patchs.shape[3]):
            patch = patchs[:,:,:,i]
            mean = np.mean(patch.reshape(-1,1))
            error = np.sum((patch - mean).reshape(-1, 1)**2)
            error_list.append(error)
    error_list.sort(reverse=False)
    num1 = int(0.05 * len(error_list))
    thr_d = error_list[num1]
    num2 = int(0.8 * len(error_list))
    thr_u = error_list[num2]

    for p in imglist:
        img = cv2.imread(p)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        patch_idx = im2patch_idx(img, stride=stride, win=patch_size)
        for p_idx in patch_idx:
            [i,j] = p_idx
            patch = img[i:i + patch_size, j:j + patch_size, :]
            mean = np.mean(patch.reshape(-1, 1))
            error = np.sum((patch - mean).reshape(-1, 1)**2)
            if error ==0:
                continue
            if error<=thr_d:
                augs = 1
            elif error >= thr_u:
                augs = aug_times+1
            else:
                augs = aug_times
            aug_idx = list(range(8))
            random.shuffle(aug_idx)
            for aug in range(augs):
                data_list.append([p, p_idx, aug_idx[aug]])
    return data_list


def idx2data(idxs, batch_dim):
    cnt = 0
    out = np.zeros((batch_dim, batch_dim, 3, len(idxs)))
    for idx in idxs:
        p, [i,j], aug = idx
        img = cv2.imread(p)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        patch = img[i:i + batch_dim, j:j + batch_dim, :]
        out[:, :, :, cnt] = data_augmentation(patch, aug)
        cnt += 1
    return np.transpose(np.float32(out), (3, 0, 1, 2))


def prepare_testdata(dataset, pad=10):
    print('process testing data')
    imglist = loadimglist(dataset)
    cnt = 0
    for p in imglist:
        img = cv2.imread(p)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if 'kodak' in dataset.lower() and img.shape[0] != 512:
            img = np.transpose(img, (1,0,2))
        img = np.pad(img, ((pad, pad), (pad, pad), (0, 0)), 'symmetric')
        # img = np.pad(img, ((pad, pad), (pad, pad), (0, 0)), 'constant', constant_values=0)
        if cnt == 0:
            out = np.expand_dims(img, axis=0)
        else:
            tmp = np.expand_dims(img, axis=0)
            out = np.concatenate([out, tmp], 0)
        cnt += 1
    return np.float32(out)

def init_rgbdata(data, pattern):
    """
    CFA data to init rgb data
    """
    out = np.zeros((data.shape[0], data.shape[1]*2, data.shape[2]*2, 3))
    if pattern == 'bayer_gbrg':
        r = data[:, 0::2, 1::2]
        out[:, 0::2, 0::2, 0] = data[:,:,:, 2]
        out[:, 0::2, 1::2, 0] = data[:,:,:, 2]
        out[:, 1::2, 0::2, 0] = data[:,:,:, 2]
        out[:, 1::2, 1::2, 0] = data[:, :, :, 2]
        g1 = data[:, 0::2, 0::2]
        g2 = data[:, 1::2, 1::2]
        out[:, 0::2, 0::2, 1] = data[:, :, :, 0]
        out[:, 0::2, 1::2, 1] = data[:, :, :, 0]
        out[:, 1::2, 0::2, 1] = data[:, :, :, 3]
        out[:, 1::2, 1::2, 1] = data[:, :, :, 3]
        b = data[:, 1::2, 0::2]
        out[:, 0::2, 0::2, 2] = data[:, :, :, 1]
        out[:, 0::2, 1::2, 2] = data[:, :, :, 1]
        out[:, 1::2, 0::2, 2] = data[:, :, :, 1]
        out[:, 1::2, 1::2, 2] = data[:, :, :, 1]
        return out


def compute_CFA(data, pattern):
    """
    data mode: [batch_size, h, w, 3]
    """
    out_3d = np.zeros((data.shape[0], data.shape[1]//2, data.shape[2]//2, 4))
    mask = np.zeros((data.shape[0], data.shape[1], data.shape[2], 3))
    if pattern == 'bayer_gbrg':
        mask[:, 0::2, 0::2, 1] = 1
        mask[:, 1::2, 0::2, 2] = 1
        mask[:, 0::2, 1::2, 0] = 1
        mask[:, 1::2, 1::2, 1] = 1
        out_2d = data * mask
        out_2d = np.sum(out_2d, axis=3)
        out_2d = out_2d[:, :, :, np.newaxis]

        out_3d[:, :, :, 0] = data[:, 0::2, 0::2, 1]
        out_3d[:, :, :, 1] = data[:, 1::2, 0::2, 2]
        out_3d[:, :, :, 2] = data[:, 0::2, 1::2, 0]
        out_3d[:, :, :, 3] = data[:, 1::2, 1::2, 1]
        return out_3d, out_2d
    else:
        raise NotImplementedError('Only bayer_gbrg is implemented')