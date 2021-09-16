# -*- coding: utf-8 -*-
# @Time    : 2021/5/19 下午3:31
# @Author  : xiaorui su
# @Email   :  suxiaorui19@mails.ucas.edu.cn
# @File    : config.py
# @Software : PyCharm
import os

RAW_DATA_DIR = os.getcwd()+'/raw_data'
PROCESSED_DATA_DIR = os.getcwd()+'/data'
LOG_DIR = os.getcwd()+'/log'
MODEL_SAVED_DIR = os.getcwd()+'/ckpt'

KG_FILE = {'HDVD':os.path.join(RAW_DATA_DIR,'HDVD','train2id.txt'),
           'VDA1':os.path.join(RAW_DATA_DIR,'VDA1','train2id.txt'),
           'VDA2':os.path.join(RAW_DATA_DIR,'VDA2','train2id.txt'),}
ENTITY2ID_FILE = {'HDVD':os.path.join(RAW_DATA_DIR,'HDVD','entity2id.txt'),
                  'VDA1':os.path.join(RAW_DATA_DIR,'VDA1','entity2id.txt'),
                  'VDA2':os.path.join(RAW_DATA_DIR,'VDA2','entity2id.txt'),}
EXAMPLE_FILE = {'HDVD':os.path.join(RAW_DATA_DIR,'HDVD','approved_example.txt'),
                'VDA1':os.path.join(RAW_DATA_DIR,'VDA1','approved_example.txt'),
                'VDA2':os.path.join(RAW_DATA_DIR,'VDA2','approved_example.txt')}
SEPARATOR = {'HDVD':' ','VDA1':' ','VDA2':' '}
THRESHOLD = {'HDVD':4,'VDA1':4, 'VDA2':4} #添加drug修改
NEIGHBOR_SIZE = {'HDVD':5,'VDA1':5, 'VDA2':5}

#
DRUG_VOCAB_TEMPLATE = '{dataset}_drug_vocab.pkl'
ENTITY_VOCAB_TEMPLATE = '{dataset}_entity_vocab.pkl'
RELATION_VOCAB_TEMPLATE = '{dataset}_relation_vocab.pkl'
ADJ_ENTITY_TEMPLATE = '{dataset}_adj_entity.npy'
ADJ_RELATION_TEMPLATE = '{dataset}_adj_relation.npy'
ENTITY_HASH = "entity_hash.npy"
TRAIN_DATA_TEMPLATE = '{dataset}_train.npy'
DEV_DATA_TEMPLATE = '{dataset}_dev.npy'
TEST_DATA_TEMPLATE = '{dataset}_test.npy'
#RESULT_LOG='result.txt'
RESULT_LOG={'HDVD':'hdvd_result.txt','VDA1':'vda1_result.txt','VDA2':'vda2_result.txt'}
PERFORMANCE_LOG = 'parameter_performance.log'
DRUG_EXAMPLE='{dataset}_examples.npy'
##similarity file
ADJ_D_V = '{dataset}_interaction.npy'
DRUG_SIM = '{dataset}_drugsim.npy'
VIRUS_SIM = '{dataset}_virussim.npy'

FEATURE_TEMPLATE = '{dataset}_feature.npy'

class ModelConfig(object):

    def __init__(self):
        # checkpoint configuration 设置检查点
        self.checkpoint_dir = MODEL_SAVED_DIR
        self.checkpoint_monitor = 'val_auc'
        self.checkpoint_save_best_only = True
        self.checkpoint_save_weights_only = True
        self.checkpoint_save_weights_mode = 'max'
        self.checkpoint_verbose = 1

        # early_stoping configuration
        self.early_stopping_monitor = 'val_auc'
        self.early_stopping_mode = 'max'
        self.early_stopping_patience = 5
        self.early_stopping_verbose = 1
        self.dataset = 'drug'
        self.K_Fold = 1
        self.callbacks_to_add = None
        self.optimizer = 'adam'

        # config for learning rating scheduler and ensembler
        self.swa_start = 3

        self.drugsim = None
        self.virussim = None
        self.weight_matrix = None
        self.interaction = None
        self.drugfeature = None
        self.virusfeature = None

        self.rank = 16
        self.steps = 2000
        self.learning_rate = 0.001
        self.alpha = 0.001
        self.belta = 0.001
        self.lamba1 = 0.001
        self.lambda2 = 0.001

        self.neighborsize = 3
        self.depth = 3

        self.l2_weight = 1e-7
