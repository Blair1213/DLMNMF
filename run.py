# -*- coding: utf-8 -*-
# @Time    : 2021/5/20 上午2:11
# @Author  : xiaorui su
# @Email   :  suxiaorui19@mails.ucas.edu.cn
# @File    : run.py
# @Software : PyCharm

from tf_nmf import WMF
from preprocess import Preprocess
from config import ModelConfig
import numpy as np
from representation import representation
import os

config = ModelConfig()

def train(dataset: str,rank: int, steps: int, learning_rate: float,alpha: float, belta: float, lamba1: float, lambda2: float):

    config.rank = rank
    config.steps = steps
    config.learning_rate = learning_rate
    config.alpha = alpha
    config.belta = belta
    config.lamba1 = lamba1
    config.lambda2 = lambda2

    ##read file
    Preprocess(config,dataset)  ##219*34, 219 drugs and 34 viruses
    ##WMF 获得隐向量
    W,H = WMF(config,dataset)
    feature = np.concatenate((W,H.T),axis=0)
    np.save("./data/" + dataset + "_drugfeature",W)
    np.save("./data/" + dataset + "_virusfeature",H.T)
    np.save("./data/" + dataset + "_feature",feature)
    print(W)
    print(H)

    representation(dataset,config.neighborsize,config.depth,config.rank)
    ##加一个加载结果的函数，直接出图的内种，我记得有，copy过来就行

    return ;

##参数：正则参数，迭代次数，
p1 = [0,0.001,0.002,0.004,0.008,0.016,0.032,0.064,0.128,0.256,0.512,1]
p2 = [0,0.001,0.002,0.004,0.008,0.016,0.032,0.064,0.128,0.256,0.512,1]
d = ['VDA1']

train('VDA1',16,50000,0.003,0.003,0.005,0.005,0.001)

