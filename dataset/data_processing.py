# -*- coding:utf-8 -*-
__author__ = 'Fei Liu'

import numpy as np
import pandas as pd
import datetime
import time
import pickle as pkl
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import json
import re
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import heapq
import statistics
import math

with open('mfp_cleaned_dun.pkl', 'rb') as f:
    data = pkl.load(f)

user_10 = [117, 291, 741, 833, 985, 995, 1179, 1278, 1415, 1493, 1501, 1520, 1583, 1690, 1914, 2916, 3009, 3294, 3618, 3989, 4257, 4332, 4456, 4519, 4670, 4696, 4744, 5009, 5662, 5667, 5758, 5908, 5958, 6036, 6116, 6743, 7050, 7080, 7150, 7217, 7556, 7740, 8137, 8381, 8761, 9105, 9236, 9462]
with open('item_key.pickle', 'rb') as f:
    item_key = pkl.load(f)
train = []
train_test = []
val = []
test = []
for user in data.keys():
    if user not in user_10 and len(data[user].keys()) > 18:
        each_user = []
        each_user_position = []
        each_user_frequency = []
        window_start = 1
        item_pos = {}
        item_freq = {}
        position = {}
        frequency = {}
        position_list = [[], []]
        frequency_list = [[], []]
        window_del = 0
        window_del_pos = []
        for baskets in data[user].keys():
            if len(data[user][baskets].keys()) >= 2:
                basket = []
                for item in data[user][baskets].keys():
                    basket.append(item_key[item])
                if len(basket) > 20:
                    basket = basket[:20]
                each_user.append(basket)
                for each in position:
                    position_list[0].append(each)
                    position_list[1].append(position[each])
                each_user_position.append(position_list)
                for each in frequency:
                    frequency_list[0].append(each)
                    frequency_list[1].append(frequency[each])
                each_user_frequency.append(frequency_list)
                position = {}
                frequency = {}
                position_list = [[], []]
                frequency_list = [[], []]

                for item in data[user][baskets].keys():
                    if item_key[item] in basket:
                        if item_key[item] not in item_pos.keys():
                            item_pos[item_key[item]] = [data[user][baskets][item]]
                        else:
                            item_pos[item_key[item]].append(data[user][baskets][item])
                        if item_key[item] not in item_freq.keys():
                            item_freq[item_key[item]] = 1
                        else:
                            item_freq[item_key[item]] += 1
                window_start += 1
                
                if window_start > 14:
                    for item in window_del_pos[window_del].keys():
                        item_freq[item] -= 1
                        if item_freq[item] == 0:
                            del item_freq[item]

                        item_pos[item].remove(window_del_pos[window_del][item])
                        if len(item_pos[item]) == 0:
                            del item_pos[item]
                    window_del += 1
                
                tmp = {}
                for item in data[user][baskets].keys():
                    if item_key[item] in basket:
                        tmp[item_key[item]] = data[user][baskets][item]
                window_del_pos.append(tmp)

                for i in item_pos.keys():
                    position[i] = max(item_pos[i],key=item_pos[i].count)
                
                for j in item_freq.keys():
                    if item_freq[j] >=7:
                        frequency[j] = 3
                    elif item_freq[j] >= 3:
                        frequency[j] = 2
                    else:
                        frequency[j] = 1
        if len(each_user) > 18:
            if len(each_user) > 63:
                each_user = each_user[15:63]
                each_user_position = each_user_position[15:63]
                each_user_frequency = each_user_frequency[15:63]
            else:
                each_user = each_user[15:]
                each_user_position = each_user_position[15:]
                each_user_frequency = each_user_frequency[15:]
            train.append([user, each_user[:-2], len(each_user[:-2]), each_user_position[:-2], each_user_frequency[:-2]])
            train_test.append([user, each_user[:-1], len(each_user[:-1]), each_user_position[:-1], each_user_frequency[:-1]])
            val.append([user, each_user[-2], each_user_position[-2], each_user_frequency[-2]])
            test.append([user, each_user[-1], each_user_position[-1], each_user_frequency[-1]])

train_set = pd.DataFrame(train, columns=['userID', 'baskets', 'num_baskets', 'positions', 'frequencies'])
train_test_set = pd.DataFrame(train_test, columns=['userID', 'baskets', 'num_baskets', 'positions', 'frequencies'])
valid_set = pd.DataFrame(val, columns=['userID', 'baskets', 'positions', 'frequencies'])
test_set = pd.DataFrame(test, columns=['userID', 'baskets', 'positions', 'frequencies'])

train_set.to_json('train_sxw_window.json', orient='records', lines=True)
train_test_set.to_json('train_test_sxw_window.json', orient='records', lines=True)
test_set.to_json('test_sxw_window.json', orient='records', lines=True)
valid_set.to_json('validation_sxw_window.json', orient='records', lines=True)