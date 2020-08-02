#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 24 10:00:01 2018

@author: xidian
"""
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import time
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

'''
params
'''

ROOT_PATH = 'train_features.npy'
SUMMARY_DIR = '../storage/summary/u2/fscd'
MODEL_DIR = '../storage/model/u2/fscd'
MODEL_NAME = 'fscd_model'

FEAT_LENS = 512
MAX_ITS = 5e4+1

BATCH_SIZE = 160
CODE_LENS = 1024

LR_CO = 1e-2
LR_SP = 5

ALPHA_REG = 1e-1
ALPHA_SP = 1e-2

def L2norm(arr):
   norm_arr = np.sqrt(np.sum(arr**2, axis=1))
   norm_arr = norm_arr.reshape([norm_arr.size, 1])
   arr_l2 = arr/(norm_arr+1e-20)
   return arr_l2


with tf.name_scope('data'):
    inputs = tf.placeholder(tf.float32, [None, FEAT_LENS], 'inputs')
    
with tf.variable_scope('sparse_vector'): # Each samples corresponds to a sparse vectors
    sparse_vec = tf.get_variable('sparse_vec', [BATCH_SIZE, CODE_LENS],
                                 initializer=tf.random_normal_initializer(stddev=0.1))
    
with tf.variable_scope('coefficient'):
    
    weights = tf.get_variable('weights', [FEAT_LENS, CODE_LENS], 
                              initializer=tf.truncated_normal_initializer(stddev=0.1))
    biases = tf.get_variable('biases', [CODE_LENS], initializer=tf.constant_initializer(.1))
    fc1 = tf.matmul(inputs, weights) + biases   
    fc1 = tf.tanh(fc1)

    diag_e = tf.random_normal([CODE_LENS])
    gain_coff = tf.Variable(tf.matrix_diag(diag_e), name='gain_diag')

    code = tf.matmul(fc1, gain_coff)

    bases = tf.get_variable('bases', [CODE_LENS, FEAT_LENS], 
                            initializer=tf.random_normal_initializer(stddev=0.1))    
    gen_rec = tf.matmul(sparse_vec, bases)


with tf.name_scope('loss'):
    
    sparse_loss = ALPHA_SP*tf.reduce_mean(tf.reduce_sum(tf.abs(sparse_vec),1)) 
    regress_loss = ALPHA_REG*tf.reduce_mean(tf.reduce_sum(tf.square(sparse_vec-code),1))
    dict_loss = tf.reduce_mean(tf.reduce_sum(tf.square(inputs-gen_rec),1))
    coeff_loss = tf.add_n([regress_loss, dict_loss, sparse_loss])

coeff_vars = tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES, scope='coefficient') ##
sparse_vars = tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES, scope='sparse_vector')##
coeff_step = tf.Variable(0, trainable=False)
sparse_step = tf.Variable(0, trainable=False)
###############################################################
var1 = tf.trainable_variables()[4:5]  ## dict learning
var2 = tf.trainable_variables()[1:4]  ## regressor learning
train_op1 = tf.train.GradientDescentOptimizer(LR_CO).minimize(coeff_loss, 
                                             var_list=var1, global_step=coeff_step) 
#different learning rate
train_op2 = tf.train.GradientDescentOptimizer(LR_CO*1e-2).minimize(coeff_loss,
                                             var_list=var2)
optimize_coeff = tf.group(train_op1, train_op2)
###############################################################
optimize_sparse = tf.train.GradientDescentOptimizer(LR_SP).minimize(coeff_loss, \
                                var_list= sparse_vars, global_step=sparse_step)

norms = tf.sqrt(tf.reduce_sum(tf.square(bases),1))
norms = tf.reshape(norms, [CODE_LENS, 1])
bases_norm = bases/(norms+1e-10)

update = tf.assign(bases, bases_norm)

tf.summary.scalar('sparse_loss', sparse_loss)
tf.summary.scalar('regress_loss', regress_loss)
tf.summary.scalar('dict_loss', dict_loss)
tf.summary.scalar('coeff_loss', coeff_loss)
tf.summary.histogram('base_norm', norms)
summary_op = tf.summary.merge_all()

sparse_vec_new = tf.random_normal(shape=[BATCH_SIZE, CODE_LENS], stddev=0.1)
update2 = tf.assign(sparse_vec, sparse_vec_new)

if not os.path.exists(SUMMARY_DIR):
    os.makedirs(SUMMARY_DIR)
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

saver = tf.train.Saver(max_to_keep=26)
start_t = time.time()

features = np.load(ROOT_PATH)
features_nums = len(features)


config = tf.ConfigProto()
config.gpu_options.allow_growth = True

sparse_loss_sets = []
regress_loss_sets = []
dict_loss_sets = []
with tf.Session(config=config) as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    print('initialization is done!\n')
    summary_writer = tf.summary.FileWriter(SUMMARY_DIR, sess.graph)
    for it in range(int(MAX_ITS)):
        idx = np.random.randint(features_nums - BATCH_SIZE)
        datas_ = features[idx:idx + BATCH_SIZE]
        datas_ = L2norm(datas_)
        #step1: optimize sparse vector
        old_sp_loss_ = float('inf')
        while 1:
            _, sp_loss_, sparse_vec_ = sess.run([optimize_sparse, sparse_loss, sparse_vec],
                                                feed_dict={inputs: datas_})
            if sp_loss_ < old_sp_loss_:
                old_sp_loss_ = sp_loss_
            else:
                break
        #step2: optimize bases and regressor   
        _, re_loss_, d_loss_ = sess.run([optimize_coeff, regress_loss, dict_loss],
                                        feed_dict={inputs: datas_})
        sparse_loss_sets.append(sp_loss_)
        regress_loss_sets.append(re_loss_)
        dict_loss_sets.append(d_loss_)
#        print(d_loss_)
        inputs_, gens_, summary_, code_, coeff_step_ = sess.run([inputs, gen_rec,
                                        summary_op, code, coeff_step], \
                                         feed_dict={inputs: datas_})
        #step3: re-scale the column of bases to unit norm
        if it < int(MAX_ITS)-1:  #?
            _, norms_, dict_loss_ = sess.run([update, norms, dict_loss],
                                             feed_dict={inputs: datas_})
        #step4: Reinitialize sparse vector
            _ = sess.run([update2], feed_dict={inputs: datas_})
        if coeff_step_ % 100 == 0:
            print('now iteration is:{}, running time:{:0.3f}s.'.format(it, time.time()-start_t))
            summary_writer.add_summary(summary_, global_step=coeff_step_)
        if it % 5000 == 0:
            base_, gain_coff_ = sess.run([bases, gain_coff], feed_dict={inputs: datas_})
            saver.save(sess, os.path.join(MODEL_DIR, MODEL_NAME), global_step=coeff_step_)
print('training is done! running time:{:0.3f}h.'.format((time.time()-start_t)/3600))
