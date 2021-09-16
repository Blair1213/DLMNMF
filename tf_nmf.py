# -*- coding: utf-8 -*-
# @Time    : 2021/5/19 下午3:21
# @Author  : xiaorui su
# @Email   :  suxiaorui19@mails.ucas.edu.cn
# @File    : tf_nmf.py
# @Software : PyCharm

import tensorflow as tf
import numpy as np
import os
from preprocess import Preprocess
from config import ModelConfig
from plot import  plot_scatter
from sklearn.decomposition import PCA
from config import RAW_DATA_DIR,PROCESSED_DATA_DIR

np.random.seed(2021)



def pca(X):

    p = PCA(n_components = 2)
    p.fit(X)

    return p.transform(X);



def WMF(config,dataset):

    rank = config.rank
    alpha = config.alpha
    belta = config.belta
    lamba1 = config.lamba1
    lambda2 = config.lambda2
    steps = config.steps
    learning_rate = config.learning_rate

    weight_matrix_path = os.path.join(PROCESSED_DATA_DIR,dataset+"_weight_matrix.npy")
    interaction_path = os.path.join(PROCESSED_DATA_DIR,dataset+"_interaction.npy")
    drugsim_path = os.path.join(PROCESSED_DATA_DIR,dataset+"_drugsim.npy")
    virussim_path = os.path.join(PROCESSED_DATA_DIR,dataset+"_virussim.npy")

    rating_matrix = config.weight_matrix = np.load(weight_matrix_path)
    interaction = config.interaction = np.load(interaction_path)
    drugsim = config.drugsim = np.load(drugsim_path)
    virussim = config.virussim = np.load(virussim_path)


    rating_matrix = np.array(rating_matrix, dtype=np.float32)
    drugsim = np.array(drugsim, dtype=np.float32)
    virussim = np.array(virussim, dtype=np.float32)

    # Boolean mask for computing cost only on non-missing value
    ##这个应该是interaction矩阵
    tf_mask = tf.Variable(interaction)

    shape = rating_matrix.shape
    rating_matrix = tf.constant(rating_matrix)

    ###hyperparameter
    A = [1 for i in range(0, shape[0])]
    B = [1 for i in range(0, shape[1])]
    A = np.array(np.array(A).reshape(1, shape[0]), dtype=np.float32)
    B = np.array(np.array(B).reshape(shape[1], 1), dtype=np.float32)


    H = tf.Variable(np.random.randn(rank, shape[1]).astype(np.float32))  ##rank*34
    W = tf.Variable(np.random.randn(shape[0], rank).astype(np.float32))  ##219*rank

    WH = tf.matmul(W, H)

    w2 = tf.matmul(W, tf.transpose(W))
    Sw = tf.reshape(tf.diag_part(w2), [-1, 1])  ##对角矩阵
    Sw1 = tf.matmul(Sw, A)
    Sw2 = tf.transpose(Sw1)

    h2 = tf.matmul(tf.transpose(H), H)
    Sh = tf.reshape(tf.diag_part(h2), [1, -1])  ##对角矩阵
    Sh1 = tf.matmul(B, Sh)
    Sh2 = tf.transpose(Sh1)

    ##定义损失函数
    cost = tf.reduce_sum(1 / 2.0 * tf.pow(rating_matrix - WH, 2)) \
           + tf.reduce_sum(alpha * drugsim * (Sw1 - 2 * w2 + Sw2)) \
           + tf.reduce_sum(belta * virussim * (Sh1 - 2 * h2 + Sh2)) \
           + tf.reduce_sum(lamba1 * tf.norm(W, 2)) \
           + tf.reduce_sum(lambda2 * tf.norm(H, 2))


    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
    init = tf.global_variables_initializer()

    # Clipping operation. This ensure that W and H learnt are non-negative
    clip_W = W.assign(tf.maximum(tf.zeros_like(W), W))
    clip_H = H.assign(tf.maximum(tf.zeros_like(H), H))
    clip = tf.group(clip_W, clip_H)
    loss_per_step = []
    with tf.Session() as sess:
        sess.run(init)
        for i in range(steps):
            sess.run(train_step)
            sess.run(clip)
            loss_per_step.append(sess.run(cost))
            if i % 100 == 0:
                print("Cost: ", sess.run(cost))
            if len(loss_per_step) > 2:
                if np.abs(loss_per_step[-1]-loss_per_step[-2]) < 1e-7:
                    break

        learnt_W = sess.run(W)
        learnt_H = sess.run(H)

    '''
    w2 = tf.matmul(learnt_W, tf.transpose(learnt_W))
    Sw = tf.reshape(tf.diag_part(w2), [-1, 1])  ##对角矩阵
    Sw1 = tf.matmul(Sw, A)
    Sw2 = tf.transpose(Sw1)
    print(sess.run(Sw1 + Sw2 - w2 - w2))'''

    return learnt_W,learnt_H;

