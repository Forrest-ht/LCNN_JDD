import tensorflow as tf
import numpy as np
import scipy.misc
import scipy.io
from  .utils import *
from PIL import Image


def _phase_shift(I, r):
    bsize, a, b, c = I.get_shape().as_list()
    bsize = tf.shape(I)[0]  # Handling Dimension(None) type for undefined batch dim
    X = tf.reshape(I, (bsize, a, b, r, r))
    X = tf.transpose(X, (0, 1, 2, 4, 3))  # bsize, a, b, 1, 1
    X = tf.split(X, a, axis=1)  # a, [bsize, b, r, r]
    X = tf.concat([tf.squeeze(x, axis=1) for x in X], 2)  # bsize, b, a*r, r
    X = tf.split(X, b, axis=1)  # b, [bsize, a*r, r]
    X = tf.concat([tf.squeeze(x, axis=1) for x in X], 2)  # bsize, a*r, b*r
    return tf.reshape(X, (bsize, a*r, b*r, 1))


def phase_shift(img, filter_depth, r, color=False):
    if color:
        img_split = tf.split(3, 3, img)
        img_out = tf.concat(3, [_phase_shift(im, r) for im in img_split])
    else:
        img_split = tf.split(img, filter_depth, axis=3)
        img_out = tf.concat([_phase_shift(im, r) for im in img_split], 3)
        # img_out = _phase_shift(img, r)
    return img_out


# compute multi_frame video
def phase_shift_3d(X, r, color=False):
    if color:
        Xc = tf.split(4, 3, X)
        X = tf.concat(4, [_phase_shift_3d(x, r) for x in Xc])
        X = tf.reduce_sum(X, axis=1)
    else:
        X = _phase_shift_3d(X, r)
        X = tf.reduce_sum(X, axis=1)
    return X


def _phase_shift_3d(I, r):
    bsize, f_d, a, b, c = I.get_shape().as_list()
    bsize = tf.shape(I)[0]  # Handling Dimension(None) type for undefined batch dim
    X = tf.reshape(I, (bsize, f_d, a, b, r, r))
    X = tf.transpose(X, (0, 1, 2, 3, 5, 4))  # bsize, f_d, a, b, 1, 1
    X = tf.split(2, a, X)  # shape is [bsize, f_d, b, r, r], length is a
    X = tf.concat(3, [tf.squeeze(x, axis=2) for x in X])  # bsize, f_d, b, a*r, r
    X = tf.split(2, b, X)  # b, [bsize, f_d, a*r, r]
    X = tf.concat(3, [tf.squeeze(x, axis=2) for x in X])  # bsize, f_d, a*r, b*r
    return tf.reshape(X, (bsize, f_d, a*r, b*r, 1))


if __name__ == "__main__":
    pass