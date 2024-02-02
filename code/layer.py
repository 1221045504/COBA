import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
class mean_agg(nn.Module):
    def __init__(self,input_dim,output_dim,cuda=True):
        super(mean_agg,self).__init__()
        self.input_dim=input_dim
        self.output_dim=output_dim
        self.cuda = cuda
        self.device=torch.device('cuda')

        self.weight = nn.Parameter(torch.FloatTensor(4 * self.input_dim, output_dim))
        init.xavier_uniform_(self.weight)



    def forward_sp(self,node_fea1,node_fea2,neigh_list1,neigh_list2):
        node_fea=torch.cat((node_fea1,node_fea2),dim=1)
        samp_neighs = neigh_list1
        unique_nodes_list = list(set.union(*samp_neighs))
        unique_nodes = {n: i for i, n in enumerate(unique_nodes_list)}
        mask = Variable(torch.zeros(len(samp_neighs), len(unique_nodes)))
        column_indices = [unique_nodes[n] for samp_neigh in samp_neighs for n in samp_neigh]
        row_indices = [i for i in range(len(samp_neighs)) for j in range(len(samp_neighs[i]))]
        mask[row_indices, column_indices] = 1
        if self.cuda:
            mask = mask.cuda()
        num_neigh = mask.sum(1, keepdim=True)
        for i in range(mask.shape[0]):
            if num_neigh[i]!=0:
                mask[i] = mask[i].div(num_neigh[i])
        embed_matrix=torch.zeros(len(unique_nodes_list),len(node_fea1[0]))
        for i in range(len(unique_nodes_list)):
            embed_matrix[i][unique_nodes_list[i]] = 1

        if self.cuda:
            embed_matrix=embed_matrix.to(self.device)

        nei_fea = mask.mm(embed_matrix)
        idx=torch.nonzero(nei_fea).T
        data=nei_fea[idx[0],idx[1]]
        nei_fea=torch.sparse_coo_tensor(idx,data,nei_fea.shape)


        cat_fea=torch.cat((node_fea,nei_fea),dim=1)

        'agg_fea=torch.tanh(cat_fea.mm(self.weight))'

        samp_neighs = neigh_list2
        unique_nodes_list = list(set.union(*samp_neighs))
        unique_nodes = {n: i for i, n in enumerate(unique_nodes_list)}
        mask = Variable(torch.zeros(len(samp_neighs), len(unique_nodes)))
        column_indices = [unique_nodes[n] for samp_neigh in samp_neighs for n in samp_neigh]
        row_indices = [i for i in range(len(samp_neighs)) for j in range(len(samp_neighs[i]))]
        mask[row_indices, column_indices] = 1
        if self.cuda:
            mask = mask.cuda()
        num_neigh = mask.sum(1, keepdim=True)
        for i in range(mask.shape[0]):
            if num_neigh[i] != 0:
                mask[i] = mask[i].div(num_neigh[i])
        embed_matrix = torch.zeros(len(unique_nodes_list), len(node_fea1[0]))

        for i in range(len(unique_nodes_list)):
            embed_matrix[i][unique_nodes_list[i]]=1

        if self.cuda:
            embed_matrix = embed_matrix.to(self.device)


        nei_fea = mask.mm(embed_matrix)
        idx = torch.nonzero(nei_fea).T
        data = nei_fea[idx[0], idx[1]]
        nei_fea = torch.sparse_coo_tensor(idx, data, nei_fea.shape)
        cat_fea = torch.cat((cat_fea, nei_fea), dim=1)
        agg_fea = torch.tanh(torch.spmm(cat_fea,self.weight))






        return agg_fea
    def forward_den(self,node_fea1,node_fea2,neigh_list1,neigh_list2):
        node_fea=torch.cat((node_fea1,node_fea2),dim=1)
        samp_neighs = neigh_list1
        unique_nodes_list = list(set.union(*samp_neighs))
        unique_nodes = {n: i for i, n in enumerate(unique_nodes_list)}
        mask = Variable(torch.zeros(len(samp_neighs), len(unique_nodes)))
        column_indices = [unique_nodes[n] for samp_neigh in samp_neighs for n in samp_neigh]
        row_indices = [i for i in range(len(samp_neighs)) for j in range(len(samp_neighs[i]))]
        mask[row_indices, column_indices] = 1
        if self.cuda:
            mask = mask.cuda()
        num_neigh = mask.sum(1, keepdim=True)
        for i in range(mask.shape[0]):
            if num_neigh[i] != 0:
                mask[i] = mask[i].div(num_neigh[i])
        embed_matrix = torch.zeros(len(unique_nodes_list), len(node_fea1[0]))
        for i in range(len(unique_nodes_list)):
            embed_matrix[i]=node_fea1[unique_nodes_list[i]]

        if self.cuda:
            embed_matrix = embed_matrix.to(self.device)

        nei_fea = mask.mm(embed_matrix)


        cat_fea = torch.cat((node_fea, nei_fea), dim=1)


        samp_neighs = neigh_list2
        unique_nodes_list = list(set.union(*samp_neighs))
        unique_nodes = {n: i for i, n in enumerate(unique_nodes_list)}
        mask = Variable(torch.zeros(len(samp_neighs), len(unique_nodes)))
        column_indices = [unique_nodes[n] for samp_neigh in samp_neighs for n in samp_neigh]
        row_indices = [i for i in range(len(samp_neighs)) for j in range(len(samp_neighs[i]))]
        mask[row_indices, column_indices] = 1
        if self.cuda:
            mask = mask.cuda()
        num_neigh = mask.sum(1, keepdim=True)
        for i in range(mask.shape[0]):
            if num_neigh[i] != 0:
                mask[i] = mask[i].div(num_neigh[i])
        embed_matrix = torch.zeros(len(unique_nodes_list), len(node_fea1[0]))

        for i in range(len(unique_nodes_list)):
            embed_matrix[i]=node_fea1[unique_nodes_list[i]]

        if self.cuda:
            embed_matrix = embed_matrix.to(self.device)

        nei_fea = mask.mm(embed_matrix)

        cat_fea = torch.cat((cat_fea, nei_fea), dim=1)
        agg_fea = torch.tanh(torch.spmm(cat_fea, self.weight))

        return agg_fea
