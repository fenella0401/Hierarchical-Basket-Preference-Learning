# -*- coding:utf-8 -*-
__author__ = 'Fei Liu'

import os
import math
import random
import time
import logging
import pickle
import torch
import numpy as np
from math import ceil
from utils import data_helpers as dh
from config import Config
from rnn_model import HBP
from warnings import filterwarnings
filterwarnings("ignore")
from hgt_data import *
from hgt_utils import *
import multiprocessing as mp # hgt 采样

logging.info("✔︎ HBP Model Training...")
logger = dh.logger_fn("torch-log", "logs/training-{0}.log".format(time.strftime("%a, %d %b %Y %H:%M:%S +0000: ", time.localtime())))

dilim = '-' * 120
logger.info(dilim)
for attr in sorted(Config().__dict__):
    logger.info('{:>50}|{:<50}'.format(attr.upper(), Config().__dict__[attr]))
logger.info(dilim)

def sample_batch(seed, graph):
    
    np.random.seed(seed)
    
    seed_nodes, ntypes = {}, graph.get_types()
    for ntype in ntypes:
        type_nnodes = len(graph.node_feature[ntype])
        batch_sampled = np.random.choice(np.arange(type_nnodes), min(type_nnodes, Config().batch_size//len(ntypes)), replace=False)
        seed_nodes[ntype] = np.vstack([batch_sampled, np.full(len(batch_sampled),0)]).T

    feature, times, edge_list, node_dict, seed_nodes = sample_subgraph(graph, {0:True}, Config().sample_depth, Config().sample_width, seed_nodes)
    node_feature, node_type, edge_time, edge_type, edge_index = to_torch(graph, edge_list, feature, times)
    posi, nega = posi_nega(edge_list, node_dict)
    reindex = realign(graph, seed_nodes, node_dict)

    return node_feature, node_type, edge_time, edge_type, edge_index, posi, nega, reindex

def train():

    np.random.seed(7)
    torch.manual_seed(7)
    torch.cuda.manual_seed_all(7)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样

    logger.info("✔︎ Load negative sample...")
    with open(Config().NEG_SAMPLES, 'rb') as handle:
        neg_samples = pickle.load(handle)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 构造图
    logger.info("✔︎ Build graph...")
    graph, in_size = preprocess(Config().node, Config().link, Config().n_hid, Config().attributed)

    seed_nodes = {}
    type_nnodes = len(graph.node_feature['0'])
    batch_sampled = np.random.choice(np.arange(type_nnodes), type_nnodes, replace=False)
    seed_nodes['0'] = np.vstack([batch_sampled, np.full(len(batch_sampled),0)]).T
    feature, times, edge_list, node_dict, seed_nodes = sample_subgraph(graph, {0:True}, Config().sample_depth, Config().sample_width, seed_nodes)
    node_feature, node_type, edge_time, edge_type, edge_index = to_torch(graph, edge_list, feature, times)
    node_feature = node_feature.to(device)
    node_type = node_type.to(device)
    edge_time = edge_time.to(device)
    edge_type = edge_type.to(device)
    edge_index = edge_index.to(device)
    posi, nega = posi_nega(edge_list, node_dict)
    reindex = realign(graph, seed_nodes, node_dict)
    print("采样后的节点数：", len(node_feature))

    # Model config
    
    print('initail model')
    model = HBP(in_size, len(graph.get_types()), len(graph.get_meta_graph())+1, Config()).to(device)
    global train_step
    train_step = 1500

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=Config().learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 1000, eta_min=1e-6) # hgt加的

    def bpr_loss(uids, baskets, dynamic_user, item_embs, positions, frequencies):
        basket_dynamic_position = torch.Tensor(Config().num_product, Config().dynamic_dim).to(device)
        basket_dynamic_frequency = torch.Tensor(Config().num_product, Config().dynamic_dim).to(device)
        
        loss = 0
        user = 0
        for uid, bks, du in zip(uids, baskets, dynamic_user):
            loss_u = []  # loss for user
            if Config().lstm == 'True':
                for t, basket_t in enumerate(bks):
                    if basket_t[0] != 10535 and t != 0:
                        basket_dynamic_position[:][:] = model.position_embeddings.weight[0][0]
                        basket_dynamic_frequency[:][:] = model.frequency_embeddings.weight[0][0]
                        for k in range(len(positions[user][t][0])):
                            basket_dynamic_position[positions[user][t][0][k]][:] = model.position_embeddings.weight[0][positions[user][t][1][k]]
                            basket_dynamic_frequency[frequencies[user][t][0][k]][:] = model.frequency_embeddings.weight[0][frequencies[user][t][1][k]]
                        dynamic_item_embs = torch.cat((item_embs, basket_dynamic_position, basket_dynamic_frequency), 1)
                        
                        du_p_product = torch.mm(du, dynamic_item_embs.t())  # shape: [pad_len, num_item]
                        
                        pos_idx = torch.LongTensor(basket_t)

                        # Sample negative products
                        neg = random.sample(list(neg_samples[uid]), len(basket_t))
                        neg_idx = torch.LongTensor(neg)

                        # Score p(u, t, v > v')
                        score = du_p_product[t - 1][pos_idx] - du_p_product[t - 1][neg_idx]

                        # Average Negative log likelihood for basket_t
                        loss_u.append(torch.mean(-torch.nn.LogSigmoid()(score)))
            else:
                t = len(bks) - 1
                basket_t = bks[t]
                if basket_t[0] != 10535 and t != 0:
                    basket_dynamic_position[:][:] = model.position_embeddings.weight[0][0]
                    basket_dynamic_frequency[:][:] = model.frequency_embeddings.weight[0][0]
                    for k in range(len(positions[user][t][0])):
                        basket_dynamic_position[positions[user][t][0][k]][:] = model.position_embeddings.weight[0][positions[user][t][1][k]]
                        basket_dynamic_frequency[frequencies[user][t][0][k]][:] = model.frequency_embeddings.weight[0][frequencies[user][t][1][k]]
                    dynamic_item_embs = torch.cat((item_embs, basket_dynamic_position, basket_dynamic_frequency), 1)
                    
                    du_p_product = torch.mm(du, dynamic_item_embs.t())  # shape: [pad_len, num_item]
                    
                    pos_idx = torch.LongTensor(basket_t)

                    # Sample negative products
                    neg = random.sample(list(neg_samples[uid]), len(basket_t))
                    neg_idx = torch.LongTensor(neg)

                    # Score p(u, t, v > v')
                    score = du_p_product[t - 1][pos_idx] - du_p_product[t - 1][neg_idx]
                
                    loss_u = [torch.mean(-torch.nn.LogSigmoid()(score))]
            for i in loss_u:
                loss = loss + i / len(loss_u) # why i/len because all basket a loss
            user += 1
        avg_loss = torch.div(loss, len(baskets))
        return avg_loss
    
    # hgt的loss
    criterion = torch.nn.BCEWithLogitsLoss()

    def score(criterion, node_rep, posi, nega):
        
        edges = np.vstack([posi, nega])
        labels = torch.from_numpy(np.concatenate([np.ones(len(posi)), np.zeros(len(nega))]).astype(np.float32)).to(device)
        inner = torch.bmm(node_rep[edges[:,0]][:,None,:], node_rep[edges[:,1]][:,:,None]).squeeze()
        loss = criterion(inner, labels)
        
        return loss

    def train_model():
        global train_step
        model.train()  # turn on training mode for dropout
        dr_hidden = model.init_hidden(Config().batch_size)
        train_loss = 0
        torch.cuda.empty_cache() # hgt加的
        start_time = time.clock()
        num_batches = ceil(len(train_test_data) / Config().batch_size)

        for i, x in enumerate(dh.batch_iter(train_test_data, Config().batch_size, Config().seq_len, device=device, shuffle=True, iftrain=True)):
            uids, baskets, lens, positions, frequencies = x
            #node_feature, node_type, edge_time, edge_type, edge_index, posi, nega, reindex = sample_batch(i, graph)

            model.zero_grad()  # 如果不置零，Variable 的梯度在每次 backward 的时候都会累加
            if Config().lstm == 'True':
                if Config().gnn == 'True':
                    node_rep, dynamic_user, _ = model(baskets, lens, positions, frequencies, dr_hidden, node_feature, node_type, edge_time, edge_type, edge_index, reindex)
                else:
                    dynamic_user, _ = model(baskets, lens, positions, frequencies, dr_hidden, node_feature, node_type, edge_time, edge_type, edge_index, reindex)
            else:
               node_rep, dynamic_user = model(baskets, lens, positions, frequencies, dr_hidden, node_feature, node_type, edge_time, edge_type, edge_index, reindex)

            item_embs = torch.Tensor(Config().num_product, Config().embedding_dim * 2).to(device)
            for ori_idx, batch_idx in reindex.items():
                if int(ori_idx) < Config().num_product:
                    if Config().attributed == 'True':
                        if Config().gnn == 'True':
                            item_embs[int(ori_idx)] = torch.cat((node_feature[batch_idx].to(device), node_rep[batch_idx]), 0)
                        else:
                            item_embs[int(ori_idx)] = node_feature[batch_idx].to(device)
                    else:
                        item_embs[int(ori_idx)] = node_rep[batch_idx]

            if Config().gnn == 'True':
                loss1 = bpr_loss(uids, baskets, dynamic_user, item_embs, positions, frequencies)
                loss2 = score(criterion, node_rep, posi, nega)
                loss = Config().loss_alpha * loss1 + (1.0-Config().loss_alpha) * loss2
            else:
                loss = bpr_loss(uids, baskets, dynamic_user, item_embs, positions, frequencies)
            optimizer.zero_grad()
            torch.cuda.empty_cache()
            loss.backward()
            
            # Clip to avoid gradient exploding
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config().clip)

            # Parameter updating
            optimizer.step()
            train_step += 1
            scheduler.step(train_step)
            train_loss += loss.data

            # Logging
            if i % Config().log_interval == 0 and i > 0:
                elapsed = (time.clock() - start_time) / Config().log_interval
                cur_loss = train_loss.item() / Config().log_interval  # turn tensor into float
                train_loss = 0
                start_time = time.clock()
                logger.info('[Training]| Epochs {:3d} | Batch {:5d} / {:5d} | ms/batch {:02.2f} | Loss {:05.4f} |'
                            .format(epoch, i, num_batches, elapsed, cur_loss))

    def validate_model():
        model.eval()
        dr_hidden = model.init_hidden(Config().batch_size)
        
        basket_dynamic_position = torch.Tensor(Config().num_product, Config().dynamic_dim).to(device)
        basket_dynamic_frequency = torch.Tensor(Config().num_product, Config().dynamic_dim).to(device)
        
        hitratio_numer = 0
        hitratio_denom = 0
        ndcg = 0.0

        for i, x in enumerate(dh.batch_iter(train_test_data, Config().batch_size, Config().seq_len, device=device, shuffle=False, iftrain=True)):
            uids, baskets, lens, positions, frequencies = x
            #node_feature, node_type, edge_time, edge_type, edge_index, posi, nega, reindex = sample_batch(i, graph)
            if Config().lstm == 'True':
                if Config().gnn == 'True':
                    node_rep, dynamic_user, _ = model(baskets, lens, positions, frequencies, dr_hidden, node_feature, node_type, edge_time, edge_type, edge_index, reindex)
                else:
                    dynamic_user, _ = model(baskets, lens, positions, frequencies, dr_hidden, node_feature, node_type, edge_time, edge_type, edge_index, reindex)
            else:
               node_rep, dynamic_user = model(baskets, lens, positions, frequencies, dr_hidden, node_feature, node_type, edge_time, edge_type, edge_index, reindex)

            item_embs = torch.Tensor(Config().num_product, Config().embedding_dim * 2).to(device)
            for ori_idx, batch_idx in reindex.items():
                if int(ori_idx) < Config().num_product:
                    if Config().attributed == 'True':
                        if Config().gnn == 'True':
                            item_embs[int(ori_idx)] = torch.cat((node_feature[batch_idx].to(device), node_rep[batch_idx]), 0)
                        else:
                            item_embs[int(ori_idx)] = node_feature[batch_idx].to(device)
                    else:
                        item_embs[int(ori_idx)] = node_rep[batch_idx]

            for uid, l, du in zip(uids, lens, dynamic_user):
                basket_dynamic_position[:][:] = model.position_embeddings.weight[0][0]
                basket_dynamic_frequency[:][:] = model.frequency_embeddings.weight[0][0]
                for k in range(len(validation_data[validation_data['userID'] == uid].positions.values[0][0])):
                    basket_dynamic_position[validation_data[validation_data['userID'] == uid].positions.values[0][0][k]][:] = model.position_embeddings.weight[0][validation_data[validation_data['userID'] == uid].positions.values[0][1][k]]
                    basket_dynamic_frequency[validation_data[validation_data['userID'] == uid].frequencies.values[0][0][k]][:] = model.frequency_embeddings.weight[0][validation_data[validation_data['userID'] == uid].frequencies.values[0][1][k]]
                dynamic_item_embs = torch.cat((item_embs, basket_dynamic_position, basket_dynamic_frequency), 1)
                
                scores = []
                if Config().lstm == 'True':
                    du_latest = du[l - 1].unsqueeze(0)
                else:
                    du_latest = du.unsqueeze(0)

                # calculating <u,p> score for all test items <u,p> pair
                positives = validation_data[validation_data['userID'] == uid].baskets.values[0]  # list dim 1
                p_length = len(positives)
                positives = torch.LongTensor(positives)

                # Deal with positives samples
                scores_pos = list(torch.mm(du_latest, dynamic_item_embs[positives].t()).data.cpu().numpy()[0])
                for s in scores_pos:
                    scores.append(s)

                # Deal with negative samples
                negtives = random.sample(list(neg_samples[uid]), Config().neg_num)
                negtives = torch.LongTensor(negtives)
                scores_neg = list(torch.mm(du_latest, dynamic_item_embs[negtives].t()).data.cpu().numpy()[0])
                for s in scores_neg:
                    scores.append(s)

                # Calculate hit-ratio # see here!!
                index_k = []
                for k in range(Config().top_k):
                    index = scores.index(max(scores))
                    index_k.append(index)
                    scores[index] = -9999
                hitratio_numer += len((set(np.arange(0, p_length)) & set(index_k)))
                hitratio_denom += p_length

                # Calculate NDCG
                u_dcg = 0
                u_idcg = 0
                for k in range(Config().top_k):
                    if index_k[k] < p_length:  # 长度 p_length 内的为正样本
                        u_dcg += 1 / math.log(k + 1 + 1, 2)
                    u_idcg += 1 / math.log(k + 1 + 1, 2)
                ndcg += u_dcg / u_idcg

        hit_ratio = hitratio_numer / hitratio_denom
        ndcg = ndcg / len(train_test_data)
        logger.info('[Validate]| Epochs {:3d} | Hit ratio {:02.4f} | NDCG {:05.4f} |'
                    .format(epoch, hit_ratio, ndcg))
        return hit_ratio, ndcg

    def test_model():
        model.eval()
        dr_hidden = model.init_hidden(Config().batch_size)

        basket_dynamic_position = torch.Tensor(Config().num_product, Config().dynamic_dim).to(device)
        basket_dynamic_frequency = torch.Tensor(Config().num_product, Config().dynamic_dim).to(device)

        hitratio_numer = 0
        hitratio_denom = 0
        ndcg = 0.0
        all_novelty = 0.0

        hitratio_numer_5 = 0
        hitratio_denom_5 = 0
        ndcg_5 = 0.0
        all_novelty_5 = 0.0

        hitratio_numer_15 = 0
        hitratio_denom_15 = 0
        ndcg_15 = 0.0
        all_novelty_15 = 0.0

        for i, x in enumerate(dh.batch_iter(train_test_data, Config().batch_size, Config().seq_len, device=device, shuffle=False, iftrain=False)):
            uids, baskets, lens, positions, frequencies = x
            #node_feature, node_type, edge_time, edge_type, edge_index, posi, nega, reindex = sample_batch(i, graph)
            if Config().lstm == 'True':
                if Config().gnn == 'True':
                    node_rep, dynamic_user, _ = model(baskets, lens, positions, frequencies, dr_hidden, node_feature, node_type, edge_time, edge_type, edge_index, reindex)
                else:
                    dynamic_user, _ = model(baskets, lens, positions, frequencies, dr_hidden, node_feature, node_type, edge_time, edge_type, edge_index, reindex)
            else:
               node_rep, dynamic_user = model(baskets, lens, positions, frequencies, dr_hidden, node_feature, node_type, edge_time, edge_type, edge_index, reindex)
            
            item_embs = torch.Tensor(Config().num_product, Config().embedding_dim * 2).to(device)
            for ori_idx, batch_idx in reindex.items():
                if int(ori_idx) < Config().num_product:
                    if Config().attributed == 'True':
                        if Config().gnn == 'True':
                            item_embs[int(ori_idx)] = torch.cat((node_feature[batch_idx].to(device), node_rep[batch_idx]), 0)
                        else:
                            item_embs[int(ori_idx)] = node_feature[batch_idx].to(device)
                    else:
                        item_embs[int(ori_idx)] = node_rep[batch_idx]

            for uid, l, du, basket in zip(uids, lens, dynamic_user, baskets):
                basket_dynamic_position[:][:] = model.position_embeddings.weight[0][0]
                basket_dynamic_frequency[:][:] = model.frequency_embeddings.weight[0][0]
                for k in range(len(test_data[test_data['userID'] == uid].positions.values[0][0])):
                    basket_dynamic_position[test_data[test_data['userID'] == uid].positions.values[0][0][k]][:] = model.position_embeddings.weight[0][test_data[test_data['userID'] == uid].positions.values[0][1][k]]
                    basket_dynamic_frequency[test_data[test_data['userID'] == uid].frequencies.values[0][0][k]][:] = model.frequency_embeddings.weight[0][test_data[test_data['userID'] == uid].frequencies.values[0][1][k]]
                dynamic_item_embs = torch.cat((item_embs, basket_dynamic_position, basket_dynamic_frequency), 1)
                
                scores = []
                if Config().lstm == 'True':
                    du_latest = du[l - 1].unsqueeze(0)
                else:
                    du_latest = du.unsqueeze(0)

                # calculating <u,p> score for all test items <u,p> pair
                positives = test_data[test_data['userID'] == uid].baskets.values[0]  # list dim 1
                p_length = len(positives)
                positives_t = torch.LongTensor(positives)

                # Deal with positives samples
                scores_pos = list(torch.mm(du_latest, dynamic_item_embs[positives_t].t()).data.cpu().numpy()[0])
                for s in scores_pos:
                    scores.append(s)

                # Deal with negative samples
                negtives = random.sample(list(neg_samples[uid]), Config().neg_num)
                negtives_t = torch.LongTensor(negtives)
                scores_neg = list(torch.mm(du_latest, dynamic_item_embs[negtives_t].t()).data.cpu().numpy()[0])
                for s in scores_neg:
                    scores.append(s)

                index_k = []
                item_id = []
                for k in range(15):
                    index = scores.index(max(scores))
                    index_k.append(index)
                    if index < p_length:
                        item_id.append(positives[index])
                    else:
                        item_id.append(negtives[index-p_length])
                    scores[index] = -9999
                
                # Calculate hit-ratio # see here!!
                hitratio_numer += len((set(np.arange(0, p_length)) & set(index_k[:Config().top_k])))
                hitratio_denom += p_length

                # Calculate NDCG
                u_dcg = 0
                u_idcg = 0
                for k in range(Config().top_k):
                    if index_k[k] < p_length:  # 长度 p_length 内的为正样本
                        u_dcg += 1 / math.log(k + 1 + 1, 2)
                    u_idcg += 1 / math.log(k + 1 + 1, 2)
                ndcg += u_dcg / u_idcg

                # calculate novelty
                all_basket_item = [each for e in basket for each in e]
                novelty = len((set(item_id[:Config().top_k]) & set(all_basket_item)))
                all_novelty += 1 - novelty / len(set(item_id[:Config().top_k]))

                # @5
                # Calculate hit-ratio # see here!!
                hitratio_numer_5 += len((set(np.arange(0, p_length)) & set(index_k[:5])))
                hitratio_denom_5 += p_length

                # Calculate NDCG
                u_dcg = 0
                u_idcg = 0
                for k in range(5):
                    if index_k[k] < p_length:  # 长度 p_length 内的为正样本
                        u_dcg += 1 / math.log(k + 1 + 1, 2)
                    u_idcg += 1 / math.log(k + 1 + 1, 2)
                ndcg_5 += u_dcg / u_idcg

                # calculate novelty
                novelty = len((set(item_id[:5]) & set(all_basket_item)))
                all_novelty_5 += 1 - novelty / len(set(item_id[:5]))

                # @15
                # Calculate hit-ratio # see here!!
                hitratio_numer_15 += len((set(np.arange(0, p_length)) & set(index_k)))
                hitratio_denom_15 += p_length

                # Calculate NDCG
                u_dcg = 0
                u_idcg = 0
                for k in range(15):
                    if index_k[k] < p_length:  # 长度 p_length 内的为正样本
                        u_dcg += 1 / math.log(k + 1 + 1, 2)
                    u_idcg += 1 / math.log(k + 1 + 1, 2)
                ndcg_15 += u_dcg / u_idcg

                # calculate novelty
                novelty = len((set(item_id) & set(all_basket_item)))
                all_novelty_15 += 1 - novelty / len(set(item_id))

        hit_ratio = hitratio_numer / hitratio_denom
        ndcg = ndcg / len(train_test_data)
        all_novelty = all_novelty / len(train_test_data)
        hit_ratio_5 = hitratio_numer_5 / hitratio_denom_5
        ndcg_5 = ndcg_5 / len(train_test_data)
        hit_ratio_15 = hitratio_numer_15 / hitratio_denom_15
        ndcg_15 = ndcg_15 / len(train_test_data)
        all_novelty_5 = all_novelty_5 / len(train_test_data)
        all_novelty_15 = all_novelty_15 / len(train_test_data)
        logger.info('[Test]| Epochs {:3d} | Hit ratio {:02.4f} | NDCG {:05.4f} | Novelty {:.4f}'
                    .format(epoch, hit_ratio, ndcg, all_novelty))
        logger.info('[Test]| Epochs {:3d} | Hit ratio @5 {:02.4f} | NDCG @5 {:05.4f} | Novelty @5 {:.4f}'
                    .format(epoch, hit_ratio_5, ndcg_5, all_novelty_5))
        logger.info('[Test]| Epochs {:3d} | Hit ratio @15 {:02.4f} | NDCG @15 {:05.4f} | Novelty @15 {:.4f}'
                    .format(epoch, hit_ratio_15, ndcg_15, all_novelty_15))

    timestamp = str(int(time.time()))
    out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    logger.info('Save into {0}'.format(out_dir))
    checkpoint_dir = out_dir + '/model-{epoch:02d}-{hitratio:.4f}-{ndcg:.4f}.model'

    best_hit_ratio = None

    logger.info("✔︎ Validation data processing...")
    validation_data = dh.load_data(Config().VALIDATIONSET_DIR)
    logger.info("✔︎ Trainingfortest data processing...")
    train_test_data = dh.load_data(Config().TRAININGFORTESTSET_DIR)
    logger.info("✔︎ Test data processing...")
    test_data = dh.load_data(Config().TESTSET_DIR)

    try:
        # Training
        for epoch in range(Config().epochs):
            
            train_model()
            logger.info('-' * 89)

            hit_ratio, ndcg = validate_model()
            logger.info('-' * 89)

            # Checkpoint
            if not best_hit_ratio or hit_ratio > best_hit_ratio:
                test_model()
                logger.info('-' * 89)
                with open(checkpoint_dir.format(epoch=epoch, hitratio=hit_ratio, ndcg=ndcg), 'wb') as f:
                    torch.save(model, f)
                best_hit_ratio = hit_ratio

    except KeyboardInterrupt:
        logger.info('*' * 89)
        logger.info('Early Stopping!')


if __name__ == '__main__':
    train()
