# -*- coding: utf-8 -*-
# @Time    : 2021/5/19 下午3:23
# @Author  : xiaorui su
# @Email   :  suxiaorui19@mails.ucas.edu.cn
# @File    : preprocess.py
# @Software : PyCharm

import numpy as np
import os
from config import ModelConfig
from config import RAW_DATA_DIR,PROCESSED_DATA_DIR


class Preprocess(object):

    def __init__(self, config : ModelConfig, dataset):
        self.config = config
        self.model = self.build(dataset)

    def build(self,dataset):
        RAW_DATA_DIR = "./raw_data"
        PROCESSED_DATA_DIR = "./data"
        interaction_path = os.path.join(RAW_DATA_DIR,dataset,'virusdrug.csv')
        virus_sim = os.path.join(RAW_DATA_DIR,dataset,'virussim.csv')
        drug_sim = os.path.join(RAW_DATA_DIR,dataset,'drugsim.csv')

        interaction = self.file_read(interaction_path)
        drugsim = self.file_read_value(drug_sim)
        virussim = self.file_read_value(virus_sim)
        ##可以当作两个通道
        DI = np.dot(drugsim, interaction)
        VI = np.dot(interaction, virussim)
        weight_matrix = np.dot(np.dot(drugsim, interaction), virussim)

        ##对weight matrix做归一化 weight matrix == rating matrxi
        weight_matrix = self.normalize(weight_matrix)

        np.save(os.path.join(PROCESSED_DATA_DIR,dataset+"_interaction"), interaction)
        np.save(os.path.join(PROCESSED_DATA_DIR,dataset+"_drugsim"), drugsim)
        np.save(os.path.join(PROCESSED_DATA_DIR,dataset+"_virussim"), virussim)
        np.save(os.path.join(PROCESSED_DATA_DIR,dataset+"_weight_matrix"),weight_matrix)

        return ;

    def file_read(self,file_path):

        ##input: file_path
        ##output: a matrix

        interactions = []  # drug id, virus id
        f = open(file_path, 'r')
        flag = 1
        for line in f.readlines():
            if flag == 1:
                ##读到的是virus名称
                flag = 0
                continue
            else:
                inter = line.strip().split(",")[1:]
                inter = [np.int(j) for j in inter]
                interactions.append(inter)
        interactions = np.array(interactions)

        return interactions;

    def file_read_value(self,file_path):
        ##input: file_path
        ##output: a matrix

        interactions = []  # drug id, virus id
        f = open(file_path, 'r')
        flag = 1
        for line in f.readlines():
            if flag == 1:
                ##读到的是virus名称
                flag = 0
                continue
            else:
                inter = line.strip().split(",")[1:]
                inter = [np.float(j) for j in inter]
                interactions.append(inter)
        interactions = np.array(interactions)

        return interactions;

    def distance_calculation(self,X):

        # determin dimensions of raw_data matrix
        m, n = X.shape
        # compute Gram matrix
        G = np.dot(X, X.T)
        # initialize squared EDM D
        D = np.zeros([m, m])
        # iterate over upper triangle of D
        for i in range(m):
            for j in range(i + 1, m):
                d = X[i] - X[j]
                D[i, j] = G[i, i] - 2 * G[i, j] + G[j, j]
                D[j, i] = D[i, j]

        return D;

    def distance_matrix(self, X):
        m, n = X.shape
        ab = np.dot(X, X.T)
        Sa = np.diag(ab)  # 219*1
        print("Sa")
        print(Sa.shape)
        Sa = Sa.reshape(219, 1)
        number = [1 for i in range(0, m)]
        number = np.array(number).reshape(1, 219)
        a1 = np.dot(Sa, number)
        a2 = a1.transpose()

        return a1 + a2 - ab - ab;

    def normalize(self, weight_matrix):

        m, n = weight_matrix.shape
        D = np.zeros([m, n])

        for i in range(m):
            min = np.min(weight_matrix[i])
            max = np.max(weight_matrix[i])
            dis = max - min
            D[i] = (weight_matrix[i] - min) / dis

        return D;

