"""
Author: Tao Huang
Code for Lightweight Deep Residue Learning for Joint Color Image Demosaicking and Denoising (ICPR2018 Oral)
"""

from __future__ import (absolute_import, division, print_function)
import os
import numpy as np
import tensorflow as tf

from src.utils import pp
from src.Resmodel_ICPR import LCNN

os.environ["CUDA_DEVICE_ORDER"]= "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

flags = tf.app.flags
flags.DEFINE_integer("is_train", 0, "1 for training, 0 for testing")
flags.DEFINE_string("checkpoint_dir","checkpoint", "directory name to save the checkpoint")
flags.DEFINE_string("model_dir","gbrg_patch50", "directory name to save the checkpoint")

### train parameter setting
flags.DEFINE_integer("epoch", 100, "Epoch to train")
flags.DEFINE_integer("batch_size", 256, "the size of batch images for training")
flags.DEFINE_integer("batch_HW", 50, "the dims of batch demosaic images for training")
flags.DEFINE_integer("train_size", 256, "the max size of train images")
flags.DEFINE_integer("train_aug_times", 3, "num for train augment")
flags.DEFINE_integer("stride_img", 70, "sride for each patch from image")
flags.DEFINE_float("learning_rate",0.001,"Learning  rate for Adam")
flags.DEFINE_boolean("with_BN",False,"True for using BN, False for Not using BN")
flags.DEFINE_string("train_dataset","./data/WED/", "train dataset directory ")

### test parameter setting
flags.DEFINE_string("test_dataset","./data/Kodak/", "test dataset directory")
flags.DEFINE_string("bayer_mask","bayer_gbrg", "mask of mosiack")
flags.DEFINE_integer("test_aug", 1, "num for test augment")
FLAGS = flags.FLAGS


def main(_):
    pp.pprint(flags.FLAGS.__flags)

    with tf.Session() as sess:
        model = LCNN(sess, FLAGS)
        if FLAGS.is_train:
            model.train(FLAGS)
        else:
            model.test(FLAGS)


if __name__ == '__main__':
    tf.app.run()
