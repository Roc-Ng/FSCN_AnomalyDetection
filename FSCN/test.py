#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 28 20:37:32 2018

@author: xidian
"""
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import time
import glob
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

'''
params
'''
ROOT_PATH = '../data/hidden_features/u2/test'
SAVED_DIR = '../data/fcsd_loss/u2'
MODEL_DIR = '../storage/model/u2/fscd'
MODEL_NAME = 'fscd_model-50001'


FEAT_LENS = 512
MAX_ITS = 1950  # av 15219 # u2:1950 u1 7020 sh 40256
CODE_LENS = 1024
ALPHA_SP = 1e-3


def L2norm(arr):
   norm_arr = np.sqrt(np.sum(arr**2, axis=1))
   norm_arr = norm_arr.reshape([norm_arr.size, 1])
   arr_l2 = arr/(norm_arr+1e-20)
   return arr_l2

loss_all = []

test_features_list = glob.glob(os.path.join(ROOT_PATH, '*.npy'))
test_features_list = sorted(test_features_list)
print(len(test_features_list))
with tf.Graph().as_default():
    with tf.name_scope('data'):
        inputs = tf.placeholder(tf.float32, [None, FEAT_LENS], 'inputs')

    with tf.variable_scope('coefficient'):

        weights = tf.get_variable('weights', [FEAT_LENS, CODE_LENS],
                                  initializer=tf.truncated_normal_initializer(stddev=0.05))
        biases = tf.get_variable('biases', [CODE_LENS], initializer=tf.constant_initializer(0.1))
        fc1 = tf.matmul(inputs, weights) + biases
        fc1 = tf.tanh(fc1)
        gain_coff = tf.Variable(tf.eye(CODE_LENS), name='gain_diag')
        code = tf.matmul(fc1, gain_coff)
        bases = tf.get_variable('bases', [CODE_LENS, FEAT_LENS],
                                initializer=tf.truncated_normal_initializer(stddev=0.5))
        gen_rec = tf.matmul(code, bases)


    with tf.name_scope('loss'):

        dict_loss = tf.reduce_sum(tf.square(inputs-gen_rec),1)
        sparse_loss = ALPHA_SP*tf.reduce_sum(tf.abs(code),1)
        SP_LOSS = tf.reduce_sum(tf.abs(code), 1)  ## 2020-2-28
        total_loss = tf.add_n([dict_loss, sparse_loss])

    saver = tf.train.Saver()
    loss_set = []
    SP_LOSS_set = []
    start_t = time.time()
    with tf.Session() as sess:
        saver.restore(sess, os.path.join(MODEL_DIR, MODEL_NAME))

        for it in range(int(MAX_ITS)):
            features = np.load(test_features_list[it])
            datas_ = L2norm(features)
            sp_loss_, loss_, in_, out_ = sess.run([SP_LOSS, total_loss, inputs, gen_rec], feed_dict={inputs: datas_})
            loss_ = np.squeeze(loss_)
            loss_set.append(loss_)
            SP_LOSS_set.extend(sp_loss_)
        print('sparse value: {0:.4f}'.format(sum(SP_LOSS_set)/len(SP_LOSS_set)))
if not os.path.exists(SAVED_DIR):
    os.makedirs(SAVED_DIR)

np.save(os.path.join(SAVED_DIR, 'loss.npy'), loss_set)
print('total testing folders are done! Runing time:{:0.3f}s.'.format(time.time()-start_t))
## Then use loss_set to compute AUC