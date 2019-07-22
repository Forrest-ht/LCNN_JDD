from __future__ import division
import os
import time
import scipy.signal
import numpy as np
import tensorflow as tf
from .ops import *
from .utils import *
from .data_loader import *
from datetime import timedelta
from skimage.measure import compare_ssim as ssim
from skimage.color import  rgb2ycbcr


TF_VERSION = float('.'.join(tf.__version__.split('.')[:2]))
class LCNN(object):
    def __init__(self, sess, config):

        self.sess = sess
        self.batch = config.batch_size
        self.batch_HW = config.batch_HW
        self.is_train = config.is_train
        self.checkpoint_dir = config.checkpoint_dir
        self.with_BN = config.with_BN
        self.train_dataset = config.train_dataset

        h,w = self.batch_HW, self.batch_HW
        if self.is_train:
            self.input = tf.placeholder(tf.float32, [None, h, w, 4], name='masaicked_image')
            self.label = tf.placeholder(tf.float32, [None, h*2, w*2, 3], name='demasaicking_image')
        else:
            self.input = tf.placeholder(tf.float32, [None, 425, 425, 4], name='masaicked_image')
            self.label = tf.placeholder(tf.float32, [None, 850, 850, 3], name='demasaicking_image')

        mean_x = 0
        input = self.input - mean_x
        label = self.label - mean_x

        # paras:226240
        with tf.variable_scope("conv2d_0"):
            output = self.conv2d(input, 12, kernel_size=3, with_add=True)
        xx = output
        with tf.variable_scope("dense_Block"):
            output = self.add_block(output, 32, 4, self.with_BN)
        with tf.variable_scope("conv2d_2"):
            output = self.conv2d(output, 32, kernel_size=1, with_add=True)
        for block in range(8):
            with tf.variable_scope("res_Block_%d" % block):
                output = self.block4(output, 32, self.with_BN, ratio= 2, num_subpath= 2)
        with tf.variable_scope("conv2d_final"):
            output = self.conv2d(output, 12, kernel_size=3)
        output = output + xx
        with tf.variable_scope("conv2d_final2"):
            output = self.conv2d(output, 12, kernel_size=3)
        output = self.recon_out(input, output, config.bayer_mask)

        ###############################
        output = phase_shift(output, 3, 2, False)
        self.net_output =  tf.clip_by_value(output + mean_x, 0.0, 255.0)
        self.loss = tf.reduce_mean(tf.square(label - output))
        self.loss_sum = tf.summary.scalar("Loss_value", self.loss)
        t_vars = tf.trainable_variables()
        self.g_vars = [var for var in t_vars]
        self.saver = tf.train.Saver(max_to_keep=5)
        self.saver_step = tf.train.Saver()
        self.writer = tf.summary.FileWriter("./logs", self.sess.graph)


    def train(self, config):
        """Train netmodel"""
        optim = tf.train.AdamOptimizer(config.learning_rate).minimize(self.loss, var_list=self.g_vars)
        tf.global_variables_initializer().run()

        if self.load(self.checkpoint_dir, config.model_dir):
            print("[*] load success...")
        else:
            print("[*] load failed....")

        print("[prepare loading train data...")
        batch_HW, batch_size = config.batch_HW, config.batch_size
        Labels_idxs = prepare_idx(self.train_dataset, batch_HW*2, config.stride_img, aug_times=config.train_aug_times)
        train_Labels_imax = prepare_data('./data/cfa_test/imax/', batch_HW*2, batch_HW*2, aug_times=1)
        train_Labels_kodak = prepare_data('./data/cfa_test/kodak/', batch_HW*2, batch_HW*2, aug_times=1)
        batch_input_imax, _ = compute_CFA(train_Labels_imax, config.bayer_mask)
        batch_input_kodak, _ = compute_CFA(train_Labels_kodak, config.bayer_mask)


        train_size = len(Labels_idxs)
        print("[INFO] the train dataset is: %s" % (str(train_size)))
        tmp = 0
        step = 1

        for epoch in range(config.epoch):
            start_time_total = time.time()
            if epoch == int(0.2*config.epoch):
                config.learning_rate = config.learning_rate / 10.0
            elif epoch == int(0.8*config.epoch):
                config.learning_rate = config.learning_rate /2.0

            random.shuffle(Labels_idxs)
            batch_total = max(train_size, config.train_size) // config.batch_size
            for idx in range(0, batch_total):

                start_time = time.time()
                batch_label = idx2data(Labels_idxs[idx * batch_size:(idx + 1) * batch_size], batch_HW*2)
                bsize, sz1, sz2 = int(batch_label.shape[0]), int(batch_label.shape[1] //2), int(batch_label.shape[2] //2)
                batch_input, _ = compute_CFA(batch_input, config.bayer_mask)

                _, train_loss = self.sess.run([optim, self.loss],
                                        feed_dict={self.input: batch_input, self.label: batch_label})

                step += 1
                if step % 200 == 0:

                    total_num = int((batch_input_kodak).shape[0])
                    tmp_num = int(total_num//2)

                    de_img1_1, val_loss1_1 = self.sess.run(
                        [self.net_output, self.loss],
                        feed_dict={
                            self.input: batch_input_kodak[0:tmp_num],
                            self.label: train_Labels_kodak[0:tmp_num]
                        })

                    de_img1_2, val_loss1_2 = self.sess.run(
                        [self.net_output, self.loss],
                        feed_dict={
                            self.input: batch_input_kodak[tmp_num:total_num],
                            self.label: train_Labels_kodak[tmp_num:total_num]
                        })

                    de_img2, val_loss2 = self.sess.run(
                        [self.net_output, self.loss],
                        feed_dict={
                            self.input: batch_input_imax,
                            self.label: train_Labels_imax
                        })

                    psnr1 = 20 * np.log10(255.0 / np.sqrt(val_loss1_1/2+val_loss1_2/2))
                    psnr2 = 20 * np.log10(255.0 / np.sqrt(val_loss2))

                    print(
                    "Epoch: [%2d] [%4d/%4d] time: %4.4f, train_loss: %3.8f, val_loss: %3.8f, Kodak PSNR: %3.8f, Imax PSNR: %3.8f, Kodak+Imax: %3.8f" %
                    (epoch + 1, idx + 1, batch_total, time.time() - start_time, train_loss, val_loss2, psnr1, psnr2,
                     psnr1 + psnr2))

                    if (psnr1+psnr2) > tmp:
                        self.save(config.checkpoint_dir, step, config.model_dir)
                        tmp = psnr1+psnr2

            time_per_epoch = time.time() - start_time_total
            seconds_left = int((config.epoch - epoch - 1) * time_per_epoch)
            print("Time per epoch: %s, Est. complete in: %s" % (
                str(timedelta(seconds=time_per_epoch)),
                str(timedelta(seconds=seconds_left))))

    def save(self, checkpoint_dir, step, model_dir):

        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        self.saver.save(self.sess, os.path.join(checkpoint_dir, "LCNN-JDD"), global_step=step)


    def recon_out(self, input, recon0, mask):
        bsize = tf.shape(input)[0]
        sz1 = int(input.shape[1])
        sz2 = int(input.shape[2])
        if mask == 'bayer_gbrg':
            with tf.variable_scope("recon_out"):
                x1 = tf.reshape(input[:, :, :, 0], (bsize, sz1, sz2, 1))
                x2 = tf.reshape(input[:, :, :, 1], (bsize, sz1, sz2, 1))
                x3 = tf.reshape(input[:, :, :, 2], (bsize, sz1, sz2, 1))
                x4 = tf.reshape(input[:, :, :, 3], (bsize, sz1, sz2, 1))

                input_12 = tf.concat([x3,x3,x3,x3,x1,x1,x4,x4,x2,x2,x2,x2],3)
                recon = input_12 + recon0
                return recon

    def block4(self, output, out_features, with_BN, ratio= 4, num_subpath= 4):

        shortcut = output
        Tmp = 0
        for block in range(num_subpath):
            with tf.variable_scope("block_%d" % block):
                tmp = self.block4_ratio(output, out_features, with_BN, ratio= ratio)
            Tmp =  Tmp + tmp

        return  0.1*Tmp + shortcut

    def block4_ratio(self, output, out_features, with_BN, ratio= 1):

        f_num = int(out_features // ratio)
        with tf.variable_scope("conv1"):
            output = self.conv2d(output, out_features=f_num, kernel_size=3, with_add=True)
        if with_BN:
            output = self.batch_norm(output)
        output = tf.nn.relu(output)
        with tf.variable_scope("conv2"):
            output = self.conv2d(output, out_features=out_features, kernel_size=3, with_add=True)
        if with_BN:
            output = self.batch_norm(output)

        return output

    def composite_function(self, _input, out_features, kernel_size=3, with_BN = False):
        """Function from paper H_l that performs:
        - batch normalization
        - ReLU nonlinearity
        - convolution with required kernel
        - dropout, if required
        """
        with tf.variable_scope("composite_function"):
            # BN
            #output = self.batch_norm(_input)
            # ReLU
            # convolution
            output = self.conv2d(_input, out_features=out_features, kernel_size=kernel_size)
            biases = self.bias_variable(shape=[out_features], name='bias')
            output = tf.nn.bias_add(output, biases)
            if with_BN:
                output = self.batch_norm(output)
            output = tf.nn.relu(output)
        return output

    def bias_variable(self, shape, name='bias'):
        initial = tf.constant(0.0, shape=shape)
        return tf.get_variable(name, initializer=initial)

    def conv2d(self, _input, out_features, kernel_size,  strides=[1, 1, 1, 1], padding='SAME', with_add= False):
        in_features = int(_input.get_shape()[-1])
        kernel = self.weight_variable_msra([kernel_size, kernel_size, in_features, out_features],name='kernel')
        output = tf.nn.conv2d(_input, kernel, strides, padding)
        if with_add:
            biases = self.bias_variable(shape=[out_features], name='bias')
            output = tf.nn.bias_add(output, biases)
        return output

    def weight_variable_msra(self, shape, name):
        return tf.get_variable(
            name=name,
            shape=shape,
            initializer=tf.contrib.layers.variance_scaling_initializer())

    def add_internal_layer(self, _input, growth_rate, with_BN = False, kernel_size=3):
        """Perform H_l composite function for the layer and after concatenate
        input with output from composite function.
        """
        # call composite function with 3x3 kernel

        comp_out = self.composite_function(_input, out_features=growth_rate, kernel_size=kernel_size, with_BN= with_BN )

        # concatenate _input with out from composite function
        if TF_VERSION >= 1.0:
            output = tf.concat(axis=3, values=(_input, comp_out))
        else:
            output = tf.concat(3, (_input, comp_out))
        return output

    def add_block(self, _input, growth_rate, layers_per_block, with_BN= False, kernel_size= 3):
        """Add N H_l internal layers"""
        output = _input
        for layer in range(layers_per_block):
            with tf.variable_scope("layer_%d" % layer):
                output = self.add_internal_layer(output, growth_rate, with_BN, kernel_size=kernel_size)
        return output

    def conv2d_relu(self, _input, out_features, kernel_size=3):
        """Function from paper H_l that performs:
        - batch normalization
        - ReLU nonlinearity
        - convolution with required kernel
        - dropout, if required
        """
        with tf.variable_scope("conv2d_relu"):
            # convolution
            output = self.conv2d(_input, out_features=out_features, kernel_size=kernel_size)
            biases = self.bias_variable(shape=[out_features], name='bias')
            output = tf.nn.bias_add(output, biases)
            # ReLU
            output = tf.nn.relu(output)
        return output

    def batch_norm(self, _input):
        output = tf.contrib.layers.batch_norm(
            _input, scale=True, is_training=self.is_train,
            updates_collections=None)
        return output

    def load(self, checkpoint_dir, model_dir):
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False

    def test(self, config):
        if self.load(self.checkpoint_dir, config.model_dir):
            print("[*] load success")
        else:
            print("[!] load failed.")

        test_label_ori = prepare_testdata(config.test_dataset, 10)
        # for batch processing 
        test_label = np.zeros((test_label_ori.shape[0], 850, 850, 3))
        pad = 20
        h, w = test_label_ori.shape[1], test_label_ori.shape[2]
        test_label[:,pad:h+pad, pad:pad+w,:] = test_label_ori

        print('kodak INFO [Test] starting test...')
        # start_time = time.clock()
        PSNR_R, PSNR_G, PSNR_B, PSNR_RGB, = [], [], [], []
        test_out_lst, SSIM_RGB = [], []

        for idx in range(test_label.shape[0]):
            label = test_label[idx]
            rec_img = np.zeros(list(label.shape))
            for mode in range(config.test_aug):
                tmp = data_augmentation(label, mode)
                tmp = tmp[np.newaxis,:,:,:]
                tmp_cfa, _ = compute_CFA(tmp, config.bayer_mask)

                de_img, loss = self.sess.run([self.net_output, self.loss],
                                                    feed_dict={self.input:tmp_cfa,
                                                                self.label:tmp})
                de_img = np.squeeze(de_img, axis=0)
                rec_img += data_augmentation_inv(de_img, mode)
            rec_img /= config.test_aug

            max = 255.0
            rec_img = rec_img[pad+10:pad+h-10, pad+10:pad+w-10, :]
            label = label[pad+10:pad+h-10, pad+10:pad+w-10, :]
            rec_img = np.clip(rec_img, 0.0, max)
            MSE_R = np.mean(np.square(rec_img[:, :, 0] - label[:, :, 0]))
            MSE_G = np.mean(np.square(rec_img[:, :, 1] - label[:, :, 1]))
            MSE_B = np.mean(np.square(rec_img[:, :, 2] - label[:, :, 2]))
            MSE_RGB = np.mean(np.square(rec_img - label))
            PSNR_R.append(20 * np.log10(max / np.sqrt(MSE_R)))
            PSNR_G.append(20 * np.log10(max / np.sqrt(MSE_G)))
            PSNR_B.append(20 * np.log10(max / np.sqrt(MSE_B)))
            PSNR_RGB.append(20 * np.log10(max / np.sqrt(MSE_RGB)))
            y1 = rgb2ycbcr(np.uint8(rec_img))[:, :, 0]
            y2 = rgb2ycbcr(np.uint8(label))[:, :, 0]
            SSIM_RGB.append(ssim(y1, y2, data_range=255, sigma=1.5, gaussian_weights= True, use_sample_covariance= False))

            test_out_lst.append(rec_img)

        print ('Result: %s=%s, %s=%s, %s=%s, %s=%s, %s=%s \n' % ('mean_PSNR_R', np.mean(PSNR_R), 'mean_PSNR_G', np.mean(PSNR_G),
                                  'mean_PSNR_B', np.mean(PSNR_B), 'mean_PSNR_RGB', np.mean(PSNR_RGB), 'SSIM_RGB', np.mean(SSIM_RGB)))