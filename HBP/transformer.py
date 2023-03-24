# -*- coding:utf-8 -*-
__author__ = 'Fei Liu'

'''Attention Is All You Need'''
'''refers to https://arxiv.org/pdf/1706.03762.pdf'''


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import copy

class Encoder(nn.Module):
    """
    BERT encoder exploits 12 or 24 transformer layers to extract features.
    """
    def __init__(self, emb_size, heads_num, dropout, feedforward_size, layers_num):
        super(Encoder, self).__init__()
        self.layers_num = layers_num
        self.transformer = nn.ModuleList([
            TransformerLayer(emb_size, heads_num, dropout, feedforward_size) for _ in range(self.layers_num)
        ])
        
    def forward(self, sen):
        """
        Args:
            emb: [batch_size x seq_length x emb_size]

        Returns:
            hidden: [batch_size x seq_length x hidden_size]
        """

        for i in range(self.layers_num):
            hidden = self.transformer[i](sen, sen)
        return hidden


class TransformerLayer(nn.Module):
    """
    Transformer layer mainly consists of two parts:
    multi-headed self-attention and feed forward layer.
    """
    def __init__(self, emb_size, heads_num, dropout, feedforward_size):
        super(TransformerLayer, self).__init__()

        # Multi-headed self-attention.
        self.self_attn = MultiHeadedAttention(
            emb_size, heads_num, dropout
        )
        self.dropout_1 = nn.Dropout(dropout)
        self.layer_norm_1 = LayerNorm(emb_size)
        # Feed forward layer.
        self.feed_forward = PositionwiseFeedForward(
            emb_size, feedforward_size
        )
        self.dropout_2 = nn.Dropout(dropout)
        self.layer_norm_2 = LayerNorm(emb_size)

    def forward(self, hidden1, hidden2):
        """
        Args:
            hidden: [batch_size x seq_length x emb_size]
            mask: [batch_size x 1 x seq_length x seq_length]

        Returns:
            output: [batch_size x seq_length x hidden_size]
        """
        inter = self.dropout_1(self.self_attn(hidden2, hidden2, hidden1))
        inter = self.layer_norm_1(inter + hidden1)
        output = self.dropout_2(self.feed_forward(inter))
        output = self.layer_norm_2(output + inter)  
        return output


class MultiHeadedAttention(nn.Module):
    """
    Each head is a self-attention operation.
    """
    def __init__(self, hidden_size, heads_num, dropout):
        super(MultiHeadedAttention, self).__init__()
        self.hidden_size = hidden_size
        self.heads_num = heads_num
        self.per_head_size = hidden_size // heads_num

        self.linear_layers = nn.ModuleList([
                nn.Linear(hidden_size, hidden_size) for _ in range(3)
            ])
        self.dropout = nn.Dropout(dropout)
        self.final_linear = nn.Linear(hidden_size, hidden_size)

    def forward(self, key, value, query):
        """
        Args:
            key: [batch_size x seq_length x hidden_size]
            value: [batch_size x seq_length x hidden_size]
            query: [batch_size x seq_length x hidden_size]
            mask: [batch_size x 1 x seq_length x seq_length]

        Returns:
            output: [batch_size x seq_length x hidden_size]
        """
        batch_size, hidden_size = key.size()
        heads_num = self.heads_num
        per_head_size = self.per_head_size

        def shape(x):
            return x. \
                   contiguous(). \
                   view(batch_size, heads_num, per_head_size). \
                   transpose(1, 2)

        def unshape(x):
            return x. \
                   transpose(1, 2). \
                   contiguous(). \
                   view(batch_size, hidden_size)

        query, key, value = [l(x). \
                             view(batch_size, -1, heads_num, per_head_size). \
                             transpose(1, 2) \
                             for l, x in zip(self.linear_layers, (query, key, value))
                            ]

        scores = torch.matmul(query, key.transpose(-2, -1))
        scores = scores / math.sqrt(float(per_head_size)) 
        probs = nn.Softmax(dim=-1)(scores)
        probs = self.dropout(probs)
        output = unshape(torch.matmul(probs, value))
        output = self.final_linear(output)
        
        return output


class PositionwiseFeedForward(nn.Module):
    """ Feed Forward Layer """
    def __init__(self, hidden_size, feedforward_size):
        super(PositionwiseFeedForward, self).__init__()
        self.linear_1 = nn.Linear(hidden_size, feedforward_size)
        self.linear_2 = nn.Linear(feedforward_size, hidden_size)
   
    def forward(self, x):
        inter = self.linear_1(x)
        inter = inter * 0.5 * (1.0 + torch.erf(inter / math.sqrt(2.0)))
        output = self.linear_2(inter)
        return output


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(hidden_size))
        self.beta = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x-mean) / (std+self.eps) + self.beta