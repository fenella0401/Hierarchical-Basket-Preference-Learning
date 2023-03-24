# -*- coding:utf-8 -*-
__author__ = 'Fei Liu'

import torch
from torch.autograd import Variable
from utils import data_helpers as dh
import math
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import glorot
from torch_geometric.utils import softmax
from transformer import *


class HGTConv(MessagePassing):
    def __init__(self, in_dim, out_dim, num_types, num_relations, n_heads, dropout = 0.2, **kwargs):
        super(HGTConv, self).__init__(aggr='add', **kwargs)

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_types = num_types
        self.num_relations = num_relations
        self.total_rel = num_types * num_relations * num_types
        self.n_heads = n_heads
        self.d_k = out_dim // n_heads
        self.sqrt_dk = math.sqrt(self.d_k)
        self.att = None
        
        self.k_linears = nn.ModuleList()
        self.q_linears = nn.ModuleList()
        self.v_linears = nn.ModuleList()
        self.a_linears = nn.ModuleList()
        
        for t in range(num_types):
            self.k_linears.append(nn.Linear(in_dim, out_dim))
            self.q_linears.append(nn.Linear(in_dim, out_dim))
            self.v_linears.append(nn.Linear(in_dim, out_dim))
            self.a_linears.append(nn.Linear(out_dim, out_dim))
            
        self.relation_pri = nn.Parameter(torch.ones(num_types, num_relations, num_types, self.n_heads))
        self.relation_att = nn.Parameter(torch.Tensor(num_relations, n_heads, self.d_k, self.d_k))
        self.relation_msg = nn.Parameter(torch.Tensor(num_relations, n_heads, self.d_k, self.d_k))
        self.skip = nn.Parameter(torch.ones(num_types))
        self.drop = nn.Dropout(dropout)
        self.emb = RelTemporalEncoding(in_dim)
        
        glorot(self.relation_att)
        glorot(self.relation_msg)
        
    def forward(self, node_inp, node_type, edge_index, edge_type, edge_time):
        return self.propagate(edge_index, node_inp=node_inp, node_type=node_type.reshape(-1,1), edge_type=edge_type, edge_time=edge_time)

    def message(self, edge_index_i, node_inp_i, node_inp_j, node_type_i, node_type_j, edge_type, edge_time):
        data_size = edge_index_i.size(0)
        res_att = torch.zeros(data_size, self.n_heads).to(node_inp_i.device)
        res_msg = torch.zeros(data_size, self.n_heads, self.d_k).to(node_inp_i.device)
        
        for source_type in range(self.num_types):
            sb = (node_type_j==int(source_type))
            k_linear = self.k_linears[source_type]
            v_linear = self.v_linears[source_type] 
            for target_type in range(self.num_types):
                tb = (node_type_i==int(target_type)) & sb
                q_linear = self.q_linears[target_type]
                for relation_type in range(self.num_relations):
                    idx = (edge_type==int(relation_type)) & tb.reshape(-1)
                    if idx.sum() == 0:
                        continue
                    target_node_vec = node_inp_i[idx]
                    source_node_vec = self.emb(node_inp_j[idx], edge_time[idx])

                    q_mat = q_linear(target_node_vec).view(-1, self.n_heads, self.d_k)
                    k_mat = k_linear(source_node_vec).view(-1, self.n_heads, self.d_k)
                    k_mat = torch.bmm(k_mat.transpose(1,0), self.relation_att[relation_type]).transpose(1,0)
                    res_att[idx] = (q_mat * k_mat).sum(dim=-1) * \
                        self.relation_pri[target_type][relation_type][source_type] / self.sqrt_dk
                    v_mat = v_linear(source_node_vec).view(-1, self.n_heads, self.d_k)
                    res_msg[idx] = torch.bmm(v_mat.transpose(1,0), self.relation_msg[relation_type]).transpose(1,0)   
        self.att = softmax(res_att, edge_index_i)
        res = res_msg * self.att.view(-1, self.n_heads, 1)
        del res_att, res_msg
        return res.view(-1, self.out_dim)

    def update(self, aggr_out, node_inp, node_type):
        aggr_out = F.gelu(aggr_out)
        res = torch.zeros(aggr_out.size(0), self.out_dim).to(node_inp.device)
        for target_type in range(self.num_types):
            idx = (node_type==int(target_type)).reshape(-1)
            if idx.sum() == 0:
                continue
            alpha = F.sigmoid(self.skip[target_type])
            res[idx] = self.a_linears[target_type](aggr_out[idx]) * alpha + node_inp[idx] * (1 - alpha)
        return self.drop(res)


class RelTemporalEncoding(nn.Module):
    def __init__(self, n_hid, max_len = 240, dropout = 0.2):
        super(RelTemporalEncoding, self).__init__()
        self.drop = nn.Dropout(dropout)
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = 1 / (10000 ** (torch.arange(0., n_hid * 2, 2.)) / n_hid / 2)
        self.emb = nn.Embedding(max_len, n_hid * 2)
        self.emb.weight.data[:, 0::2] = torch.sin(position * div_term) / math.sqrt(n_hid)
        self.emb.weight.data[:, 1::2] = torch.cos(position * div_term) / math.sqrt(n_hid)
        self.emb.requires_grad = False
        self.lin = nn.Linear(n_hid * 2, n_hid)
        
    def forward(self, x, t):
        return x + self.lin(self.drop(self.emb(t))) 


class GNN(nn.Module):
    def __init__(self, in_dim, n_hid, num_types, num_relations, n_heads, n_layers, dropout):
        super(GNN, self).__init__()
        self.gcs = nn.ModuleList()
        self.num_types = num_types
        self.in_dim = in_dim
        self.n_hid = n_hid
        self.adapt_ws = nn.ModuleList()
        self.drop = nn.Dropout(dropout)
        for t in range(num_types):
            self.adapt_ws.append(nn.Linear(in_dim, n_hid))
        for l in range(n_layers):
            self.gcs.append(HGTConv(n_hid, n_hid, num_types, num_relations, n_heads, dropout))
            
    def forward(self, node_feature, node_type, edge_time, edge_type, edge_index):
        res = torch.zeros(node_feature.size(0), self.n_hid).to(node_feature.device)
        for t_id in range(self.num_types):
            idx = (node_type == int(t_id))
            if idx.sum() == 0:
                continue
            res[idx] = torch.tanh(self.adapt_ws[t_id](node_feature[idx]))
        meta_xs = self.drop(res)
        del res
        for gc in self.gcs:
            meta_xs = gc(meta_xs, node_type, edge_index, edge_type, edge_time)
        return meta_xs


class HBP(torch.nn.Module):
    """
    Input Data: b_1, ... b_i ..., b_t
                b_i stands for user u's ith basket
                b_i = [p_1,..p_j...,p_n]
                p_j stands for the  jth product in user u's ith basket
    """

    def __init__(self, in_dim, num_types, num_relations, config):
        super(HBP, self).__init__()

        # Model configuration
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.hgt = GNN(in_dim, config.n_hid, num_types, num_relations, config.n_heads, config.n_layers, config.dropout_2)

        self.position_embeddings = torch.nn.Embedding(num_embeddings=config.num_position,
                                                      embedding_dim=config.dynamic_dim,
                                                      padding_idx=config.num_position-1)
        self.frequency_embeddings = torch.nn.Embedding(num_embeddings=config.num_frequency,
                                                       embedding_dim=config.dynamic_dim,
                                                       padding_idx=config.num_frequency-1)
        
        self.self_attention = Encoder(config.hidden_dim, 4, config.dropout, config.hidden_dim, 3)
        self.pool = {'avg': dh.pool_avg, 'max': dh.pool_max}[config.basket_pool_type]  # Pooling of basket

        # RNN type specify
        if config.rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(torch.nn, config.rnn_type)(input_size=config.hidden_dim,
                                                          hidden_size=config.hidden_dim,
                                                          num_layers=config.rnn_layer_num,
                                                          batch_first=True,
                                                          dropout=config.dropout,
                                                          bidirectional=False)
        else:
            nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[config.rnn_type]
            self.rnn = torch.nn.RNN(input_size=config.hidden_dim,
                                    hidden_size=config.hidden_dim,
                                    num_layers=config.rnn_layer_num,
                                    nonlinearity=nonlinearity,
                                    batch_first=True,
                                    dropout=config.dropout,
                                    bidirectional=False)

    def forward(self, x, lengths, positions, frequencies, hidden, node_feature, node_type, edge_time, edge_type, edge_index, reindex):
        # Basket Encoding
        # users' basket sequence
        if self.config.gnn == 'True':
            node_rep = self.hgt(node_feature, node_type, edge_time, edge_type, edge_index)
            node_rep = F.relu(node_rep)

        ub_seqs = torch.Tensor(self.config.batch_size, self.config.seq_len, self.config.hidden_dim).to(self.device)
        for (i, user) in enumerate(x):  # shape of x: [batch_size, seq_len, indices of product]
            embed_baskets = torch.Tensor(self.config.seq_len, self.config.hidden_dim).to(self.device)

            for (j, basket) in enumerate(user):  # shape of user: [seq_len, indices of product]
                emb_basket = torch.Tensor(1, len(basket), self.config.embedding_dim * 2).to(self.device)
                for (k, item) in enumerate(basket):
                    if self.config.attributed == 'True':
                        if self.config.gnn == 'True':
                            if(item != 10535):
                                ori_emb = node_feature[reindex[str(item)]]
                                gnn_emb = node_rep[reindex[str(item)]]
                                emb_basket[0][k] = torch.cat((ori_emb, gnn_emb),0)
                            else:
                                emb_basket[0][k] = torch.zeros(self.config.embedding_dim * 2).to(self.device)
                        else:
                            if(item != 10535):
                                emb_basket[0][k] = node_feature[reindex[str(item)]]
                            else:
                                emb_basket[0][k] = torch.zeros(self.config.embedding_dim).to(self.device)
                    else:
                        if(item != 10535):
                            emb_basket[0][k] = node_rep[reindex[str(item)]]
                        else:
                            emb_basket[0][k] = torch.zeros(self.config.embedding_dim).to(self.device)
                position = []
                frequency = []
                for (k, item) in enumerate(basket):
                    if(item != 10535):
                        if item in positions[i][j][0]:
                            position.append(positions[i][j][1][positions[i][j][0].index(item)])
                            frequency.append(frequencies[i][j][1][frequencies[i][j][0].index(item)])
                        else:
                            position.append(0)
                            frequency.append(0)
                    else:
                        position.append(self.config.num_position-1)
                        frequency.append(self.config.num_frequency-1)
                position = torch.LongTensor(position).resize_(1, len(position)).to(self.device)
                position = self.position_embeddings(torch.autograd.Variable(position))
                frequency = torch.LongTensor(frequency).resize_(1, len(frequency)).to(self.device)
                frequency = self.frequency_embeddings(torch.autograd.Variable(frequency))
                #new_basket = emb_basket + position + frequency
                new_basket = torch.cat((emb_basket, position, frequency), 2)
                
                new_basket = self.self_attention(new_basket.squeeze(0))
                new_basket = new_basket.unsqueeze(0)
                new_basket = self.pool(new_basket, dim=1)
                new_basket = new_basket.reshape(self.config.hidden_dim)
                embed_baskets[j] = new_basket  # shape:  [seq_len, 1, embedding_dim]
            # Concat current user's all baskets and append it to users' basket sequence
            ub_seqs[i] = embed_baskets  # shape: [batch_size, seq_len, embedding_dim]

        if self.config.lstm == 'True':
            # Packed sequence as required by pytorch
            packed_ub_seqs = torch.nn.utils.rnn.pack_padded_sequence(ub_seqs, lengths, batch_first=True)

            # RNN
            output, h_u = self.rnn(packed_ub_seqs, hidden)

            # shape: [batch_size, true_len(before padding), embedding_dim]
            dynamic_user, _ = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        else:
            dynamic_user = self.pool(ub_seqs, dim=1)
            return node_rep, dynamic_user

        if self.config.gnn == 'True':
            return node_rep, dynamic_user, h_u
        else:
            return dynamic_user, h_u

    def init_weight(self):
        # Init item embedding
        initrange = 0.1
        self.position_embeddings.weight.data.uniform_(-initrange, initrange)
        self.frequency_embeddings.weight.data.uniform_(-initrange, initrange)

    def init_hidden(self, batch_size):
        # Init hidden states for rnn
        weight = next(self.parameters()).data
        if self.config.rnn_type == 'LSTM':
            return (Variable(weight.new(self.config.rnn_layer_num, batch_size, self.config.hidden_dim).zero_()),
                    Variable(weight.new(self.config.rnn_layer_num, batch_size, self.config.hidden_dim).zero_()))
        else:
            return Variable(torch.zeros(self.config.rnn_layer_num, batch_size, self.config.hidden_dim)).to(self.device)
