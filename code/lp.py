import random

import torch.optim.optimizer
import torch.nn as nn

from load_data import load_data
from model import aggregate

import numpy as np
import evaluation
from evaluation import node_classfication

import scipy.sparse as sp
device=torch.device('cuda')

dataset='jung'
train_file='./train_0.7'
test_file='./test_0.3'
epoch=40
batch_size=10000
emb_dim=128
lr=0.002
neg_nei=2
edge_batch_size=10000
node_batch_size=5000




for x in range(1):
    graph, node_id, pos_egs = load_data(train_file)
    feature_dim = len(node_id)
    features = sp.eye(len(node_id))
    degree_ave = (len(pos_egs) / len(node_id))
    cuda = True


    source_nei = []
    target_nei = []
    for i in range(len(node_id)):
        if node_id[i] in graph[0]:
            target_nei.append(set(graph[0].get(node_id[i])))
        else:
            target_nei.append(set())
        if node_id[i] in graph[1]:
            source_nei.append(set(graph[1].get(node_id[i])))
        else:
            source_nei.append(set())
    source_nei2 = []
    target_nei2 = []
    for i in range(len(node_id)):
        if node_id[i] in graph[1]:
            target_nei2.append(set())
        else:
            target_nei2.append(set(graph[0].get(node_id[i])))
        if node_id[i] in graph[0]:
            source_nei2.append(set())
        else:
            source_nei2.append(set(graph[1].get(node_id[i])))

    model = aggregate(graph, node_id, pos_egs, features, feature_dim, emb_dim, edge_batch_size,
                      node_batch_size)
    if cuda:
        model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_auc = [0, 0, 0]
    best_micro = 0
    best_macro = 0
    best_auc_epoch = 0
    best_micro_epoch = 0
    best_macro_epoch = 0
    best_auc_embedding = [[], [], []]
    best_micro_embedding = []
    best_macro_embedding = []
    eval = evaluation.LinkPrediction(test_file)

    for i in range(epoch):
        print(i)

        random.shuffle(pos_egs)
        for j in range(int(len(pos_egs) / batch_size)):

            optimizer.zero_grad()
            source_emb, target_emb = model.forward(source_nei, target_nei,source_nei2,target_nei2)
            loss = model.loss(source_emb, target_emb, j, pos_egs, batch_size)
            loss.backward()
            optimizer.step()

            source_emb_eval, target_emb_eval = model.forward(source_nei, target_nei,source_nei2,target_nei2)
            emb_matrix = []
            source_emb_eval = source_emb_eval.detach().cpu().numpy()
            target_emb_eval = target_emb_eval.detach().cpu().numpy()
            emb_matrix.append(source_emb_eval)
            emb_matrix.append(target_emb_eval)
            auc = eval.evaluate(emb_matrix)

            print('loss=', loss.item(), 'auc=', auc)


            if auc[0] > best_auc[0]:
                best_auc[0] = auc[0]
            if auc[1] > best_auc[1]:
                best_auc[1] = auc[1]
            if auc[2] > best_auc[2]:
                best_auc[2] = auc[2]
    print(best_auc)

