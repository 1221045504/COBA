import random

import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
from layer import mean_agg

from sklearn.cross_decomposition import CCA

class aggregate(nn.Module):
    def __init__(self, graph, node, pos_egs, features, fea_dim, emb_dim, edge_batch_size, node_batch_size,
                 cuda=True):
        super(aggregate, self).__init__()
        self.graph = graph
        self.node = node
        self.features = features
        self.pos_egs = pos_egs
        self.fea_dim = fea_dim
        self.emb_dim = emb_dim
        self.edge_batch_size = edge_batch_size
        self.node_batch_size = node_batch_size
        self.cuda = cuda

        self.agg1 = mean_agg(self.fea_dim, self.emb_dim)
        'self.agg2 = mean_agg(self.emb_dim, self.emb_dim)'
        self.agg3 = mean_agg(self.fea_dim, self.emb_dim)
        'self.agg4 = mean_agg(self.emb_dim, self.emb_dim)'
        'self.agg5 = mean_agg(self.emb_dim, self.emb_dim)'
        'self.agg6 = mean_agg(self.emb_dim, self.emb_dim)'
        self.device = torch.device('cuda')
        if self.cuda:
            self.agg1 = self.agg1.to(self.device)
            'self.agg2 = self.agg2.to(self.device)'
            self.agg3 = self.agg3.to(self.device)
            'self.agg4 = self.agg4.to(self.device)'
            'self.agg5 = self.agg5.to(self.device)'
            'self.agg6 = self.agg6.to(self.device)'

    def sparse_mx_to_torch_sparse_tensor(self,sparse_mx):
        """Convert a scipy sparse matrix to a torch sparse tensor."""
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)

    def forward(self, source_nei, target_nei,source_nei2,target_nei2):

        features = self.features
        features=self.sparse_mx_to_torch_sparse_tensor(features)
        if self.cuda:
            features = features.to(self.device)
        source_emb = self.agg1.forward_sp(features,features, source_nei,target_nei2)
        target_emb = self.agg3.forward_sp(features, features, target_nei, source_nei2)

        'source_emb = self.agg2.forward_den(source_emb,target_emb, source_nei,target_nei2)'
        'target_emb = self.agg4.forward_den(target_emb,source_emb, target_nei, source_nei2)'

        'source_emb=self.agg5.forward_den(source_emb,target_emb, source_nei,target_nei2)'

        'target_emb=self.agg6.forward_den(target_emb,source_emb, target_nei,source_nei2)'


        return source_emb, target_emb

    def loss(self, source_emb, target_emb, j, pos_egs, batch_size):

        sample_node = []
        sample_pos_egs = pos_egs[j * batch_size:(j + 1) * batch_size]
        for i in range(len(sample_pos_egs)):
            sample_node.append(sample_pos_egs[i][0])
            sample_node.append(sample_pos_egs[i][1])
        neg_egs = []
        for i in range(len(sample_node)):
            for j in range(2):
                neg_source = np.random.randint(0, len(self.node))
                neg_target = np.random.randint(0, len(self.node))
                if (self.node[neg_source] not in self.graph[0]):
                    neg_egs.append([self.node[neg_source], sample_node[i]])
                elif (sample_node[i] not in self.graph[0].get(self.node[neg_source])):
                    neg_egs.append([self.node[neg_source], sample_node[i]])
                if (self.node[neg_target] not in self.graph[1]):
                    neg_egs.append([sample_node[i], self.node[neg_target]])
                elif (sample_node[i] not in self.graph[1].get(self.node[neg_target])):
                    neg_egs.append([sample_node[i], self.node[neg_target]])



        if self.cuda:
            pos_pre = torch.zeros(len(sample_pos_egs)).cuda()
            neg_pre = torch.zeros(len(neg_egs)).cuda()
            pos_label = torch.ones(len(sample_pos_egs)).cuda()
            neg_label = torch.zeros(len(neg_egs)).cuda()
        else:
            pos_pre = torch.zeros(len(sample_pos_egs))
            neg_pre = torch.zeros(len(neg_egs))
            pos_label = torch.ones(len(sample_pos_egs))
            neg_label = torch.zeros(len(neg_egs))
        for i in range(len(sample_pos_egs)):
            pos_pre[i] = torch.sigmoid(torch.dot(source_emb[sample_pos_egs[i][0]], target_emb[sample_pos_egs[i][1]]))
        for i in range(len(neg_egs)):
            neg_pre[i] = torch.sigmoid(torch.dot(source_emb[neg_egs[i][0]], target_emb[neg_egs[i][1]]))
        if self.cuda:
            loss_fun = nn.BCELoss().cuda()
        else:
            loss_fun = nn.BCELoss()
        pos_loss = loss_fun(pos_pre, pos_label)
        neg_loss = loss_fun(neg_pre, neg_label)

        return pos_loss + neg_loss