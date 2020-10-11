import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import GATConv,NNConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp, global_sort_pool as gsp
from torch_geometric.utils import (add_self_loops, sort_edge_index,
                                   remove_self_loops)
from torch_sparse import spspmm

from net.MyTopK import TopKPooling
from net.MySAG import SAGPooling

import numpy as np



########################### PRGNN MICCAI Li et al. 2020  ###################################################
class NNGAT_Net(torch.nn.Module):
    def __init__(self, ratio, indim, poolmethod = 'topk'):
        super(NNGAT_Net, self).__init__()

        self.dim1 = 32
        self.dim2 = 32
        self.dim3 = 8
        self.indim = indim
        self.poolmethod = poolmethod

        self.conv1 = GATConv( self.indim, self.dim1)
        self.bn1 = torch.nn.BatchNorm1d(self.dim1)
        if self.poolmethod == 'topk':
            self.pool1 = TopKPooling(self.dim1, ratio=ratio, multiplier=1, nonlinearity=torch.sigmoid)
        elif self.poolmethod == 'sag':
            self.pool1 = SAGPooling(self.dim1, ratio=ratio, GNN=GATConv,nonlinearity=torch.sigmoid) #0.4 data1 10 fold

        self.conv2 = GATConv(self.dim1, self.dim2)
        self.bn2 = torch.nn.BatchNorm1d(self.dim2)
        if self.poolmethod == 'topk':
            self.pool2 = TopKPooling(self.dim2, ratio=ratio, multiplier=1, nonlinearity=torch.sigmoid)
        elif self.poolmethod == 'sag':
            self.pool2 = SAGPooling(self.dim2, ratio=ratio, GNN=GATConv,nonlinearity=torch.sigmoid)

        self.fc1 = torch.nn.Linear((self.dim1+self.dim2)*2, self.dim2)
        self.bn4 = torch.nn.BatchNorm1d(self.dim2)
        self.fc2 = torch.nn.Linear(self.dim2, self.dim3)
        self.bn5 = torch.nn.BatchNorm1d(self.dim3)
        self.fc3 = torch.nn.Linear(self.dim3, 2)

    def forward(self, x, edge_index, batch, edge_attr):
        # edge_attr = edge_attr.squeeze()
        # edge_index, edge_attr = self.augment_adj(edge_index, edge_attr, x.size(0))
        x = self.conv1(x, edge_index)
        if x.norm(p=2, dim=-1).min() == 0:
            print('x is zeros')
        x, edge_index, edge_attr, batch, perm, score1 = self.pool1(x, edge_index, edge_attr, batch)
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        edge_attr = edge_attr.squeeze()
        edge_index, edge_attr = self.augment_adj(edge_index, edge_attr, x.size(0))

        x = self.conv2(x, edge_index)
        x, edge_index, edge_attr, batch, perm, score2  = self.pool2(x, edge_index, edge_attr, batch)
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = torch.cat([x1,x2], dim=1) #concate

        x = self.bn4(F.relu(self.fc1(x)))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.bn5(F.relu(self.fc2(x)))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.log_softmax(self.fc3(x), dim=-1)

        return x, score1, score2

    def augment_adj(self, edge_index, edge_weight, num_nodes):
        edge_index, edge_weight = add_self_loops(edge_index, edge_weight,
                                                 num_nodes=num_nodes)
        edge_index, edge_weight = sort_edge_index(edge_index, edge_weight,
                                                  num_nodes)
        edge_index, edge_weight = spspmm(edge_index, edge_weight, edge_index,
                                         edge_weight, num_nodes, num_nodes,
                                         num_nodes)
        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)

        return edge_index, edge_weight


###########################  MICCAI Li et al. 2019  ###################################################
class LI_Net(torch.nn.Module):
    def __init__(self, indim):
        super(LI_Net, self).__init__()

        self.dim1 = 32
        self.dim2 = 32
        self.dim3 = 8
        self.indim = indim
        n1 = nn.Sequential(nn.Linear(1, 1), nn.ReLU(), nn.Linear(1, self.dim1 * self.indim))
        self.conv1 = NNConv(self.indim, self.dim1, n1, aggr="add")
        self.bn1 = torch.nn.BatchNorm1d(self.dim1)
        self.pool1 = TopKPooling(self.dim1, ratio=0.5)
        n2 = nn.Sequential(nn.Linear(1, 1), nn.ReLU(), nn.Linear(1, self.dim1 * self.dim2))
        self.conv2 = NNConv(self.dim1, self.dim2, n2, aggr="add")
        self.bn2 = torch.nn.BatchNorm1d(self.dim2)
        self.pool2 = TopKPooling(self.dim2, ratio=0.5)

        self.fc1 = torch.nn.Linear((self.dim1 + self.dim2) * 2, self.dim2)
        self.bn4 = torch.nn.BatchNorm1d(self.dim2)
        self.fc2 = torch.nn.Linear(self.dim2, self.dim3)
        self.bn5 = torch.nn.BatchNorm1d(self.dim3)
        self.fc3 = torch.nn.Linear(self.dim3, 2)

    def forward(self, x, edge_index, batch, edge_attr):
        edge_attr = edge_attr.squeeze()
        edge_index, edge_attr = self.augment_adj(edge_index, edge_attr, x.size(0))
        x = self.conv1(x, edge_index, edge_attr)
        x, edge_index, edge_attr, batch, perm, score1 = self.pool1(x, edge_index, edge_attr, batch)
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        edge_attr = edge_attr.squeeze()
        edge_index, edge_attr = self.augment_adj(edge_index, edge_attr, x.size(0))
        x = self.conv2(x, edge_index, edge_attr)
        x, edge_index, edge_attr, batch, perm, score2 = self.pool2(x, edge_index, edge_attr, batch)
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = torch.cat([x1, x2], dim=1)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.bn5(self.fc2(x)))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.log_softmax(self.fc3(x), dim=-1)

        return x,score1, score2

    def augment_adj(self, edge_index, edge_weight, num_nodes):
        edge_index, edge_weight = add_self_loops(edge_index, edge_weight,
                                                 num_nodes=num_nodes)
        edge_index, edge_weight = sort_edge_index(edge_index, edge_weight,
                                                  num_nodes)
        edge_index, edge_weight = spspmm(edge_index, edge_weight, edge_index,
                                         edge_weight, num_nodes, num_nodes,
                                         num_nodes)
        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)

        return edge_index, edge_weight
