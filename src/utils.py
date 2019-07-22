"""

"""
from __future__ import division
import math
import json
import random
import pprint
import scipy.misc
import numpy as np
import cv2
import glob
from time import gmtime, strftime

pp = pprint.PrettyPrinter()
get_stddev = lambda x, k_h, k_w: 1/math.sqrt(k_w*k_h*x.get_shape()[-1])


def im2patch(img, stride=1, win=64):
    [h, w, c] = img.shape
    patch = img[0:h-win+1:stride, 0: w-win+1:stride, :]
    TotalPatNum = patch.shape[0] * patch.shape[1]
    out = np.zeros([win, win, c, TotalPatNum], np.float32)
    cnt = 0
    for i in range(0, h+1-win, stride):
        for j in range(0, w+1-win, stride):
            out[:, :, :, cnt] = img[i:i+win, j:j+win, :]
            cnt += 1
    return out


def im2patch_idx(img, stride=1, win=64):
    [h, w, c] = img.shape
    out = []
    for i in range(0, h + 1 - win, stride):
        for j in range(0, w + 1 - win, stride):
            out.append([i,j])
    return out


def patch2img(patch, img_dims, stride=1):
    h, w = img_dims[0], img_dims[1]
    win = patch.shape[0]
    out = np.zeros([h, w, 3], np.float32)
    wei = np.zeros([h, w, 3], np.float32)
    cnt = 0
    for i in range(0, h + 1 - win, stride):
        for j in range(0, w + 1 - win, stride):
            out[i:i + win, j:j + win, :] += patch[:, :, :, cnt]
            wei[i:i + win, j:j + win, :] += 1
            cnt += 1
    out = out/wei
    return out


def data_augmentation(image, mode):
    # img mode: rgb
    out = image
    if mode == 0:
        # original
        out = out
    elif mode == 1:
        # flip up and down
        out = np.flipud(out)
    elif mode == 2:
        # rotate counterwise 90 degree
        out = np.rot90(out)
    elif mode == 3:
        # rotate 90 degree and flip up and down
        out = np.rot90(out)
        out = np.flipud(out)
    elif mode == 4:
        # rotate 180 degree
        out = np.rot90(out, k=2)
    elif mode == 5:
        # rotate 180 degree and flip
        out = np.rot90(out, k=2)
        out = np.flipud(out)
    elif mode == 6:
        # rotate 270 degree
        out = np.rot90(out, k=3)
    elif mode == 7:
        # rotate 270 degree and flip
        out = np.rot90(out, k=3)
        out = np.flipud(out)
    return out

def data_augmentation_inv(image, mode):
    # img mode: rgb
    out = image
    if mode == 0:
        # original
        out = out
    elif mode == 1:
        out = np.flipud(out)
    elif mode == 2:
        out = np.rot90(out, k=3)
    elif mode == 3:
        out = np.flipud(out)
        out = np.rot90(out, k=3)
    elif mode == 4:
        # inv of rotate 180 degree
        out = np.rot90(out, k=2)
    elif mode == 5:
        # rotate 180 degree and flip
        out = np.flipud(out)
        out = np.rot90(out, k=2)
    elif mode == 6:
        # rotate 270 degree
        out = np.rot90(out, k=1)
    elif mode == 7:
        # rotate 270 degree and flip
        out = np.flipud(out)
        out = np.rot90(out, k=1)
    return out


def loadimglist(path):
    if not path.endswith('/'):
        path += '/'
    suffix = ['*.jpg', '*.png', '*.tif', '*.jpeg', '*.bmp', '*.JPEG']
    imglist = []
    for s in suffix:
        imglist += glob.glob(path + s)

    return imglist


if __name__ == '__main__':
    # img = cv2.imread('../data/WED/00001.bmp')
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # img = img[:11, :11,:]
    # out = im2patch(img, stride=3, win=5)
    # img1 = patch2img(out, [11, 11], stride=3)
    # xx = img - img1
    # print(np.max(xx), np.min(xx))
    for i in range(8):
        x = np.random.rand(16,16)
        a = data_augmentation(x,i)
        b = data_augmentation_inv(a,i)
        xx = b - x
        print(np.max(xx), np.min(xx))