# -*- coding:utf-8 -*-
__author__ = 'Fei Liu'


class Config(object):
    def __init__(self):
        self.TRAININGSET_DIR = '../dataset/train_sxw_window.json'
        self.TRAININGFORTESTSET_DIR = '../dataset/train_test_sxw_window.json'
        self.VALIDATIONSET_DIR = '../dataset/validation_sxw_window.json'
        self.NEG_SAMPLES = '../dataset/neg_sample.pickle'
        self.MODEL_DIR = 'runs/'
        self.device = 'cuda'
        self.clip = 10
        self.epochs = 30
        self.batch_size = 128
        self.seq_len = 48
        self.learning_rate = 0.01  # Initial Learning Rate
        self.log_interval = 1  # num of batches between two logging
        self.basket_pool_type = 'max'  # ['avg', 'max']
        self.rnn_type = 'LSTM'  # ['RNN_TANH', 'RNN_RELU', 'LSTM', 'GRU']
        self.rnn_layer_num = 2
        self.dropout = 0.5
        self.dropout_2 = 0.3
        self.num_product = 10535
        self.embedding_dim = 32  # 商品表征维数， 用于定义 Embedding Layer
        self.neg_num = 500  # 负采样个数
        self.top_k = 10  # Top K 取值
        self.num_position = 6+1+1 # 食物可能出现的位置，加1表示在多个位置出现的食物
        self.num_frequency = 4+1 # 食物出现的频次，最大记为20次
        self.dynamic_dim = 8
        self.hidden_dim = 32*2 + 8*2
        self.node = '../dataset/node_file_nutrition.dat' # kg-node
        self.link = '../dataset/link_file_nutrition.dat' # kg-link
        self.cuda = 0
        self.n_hid = 32 # hidden size hgt 默认是50 我们这里需要和embedding_dim保持一致
        self.n_heads = 4 # transformer的头数 8
        self.n_layers = 3 # transformer的层数 3
        self.sample_depth = 4 # 采样深度 8
        self.sample_width = 200 # 采样宽度 200
        self.attributed = 'True' # 是否有初始化表示
        self.loss_alpha = 0.9 # loss加权比
        self.gnn = 'True' # 是否采用gnn
        self.lstm = 'False' # 是否采用lstm
