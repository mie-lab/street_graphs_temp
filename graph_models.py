import os.path as osp

import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, ChebConv  # noqa
from torch import nn

import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import SplineConv

import torch
import torch.nn.functional as F
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
from torch_geometric.nn import GraphConv, TopKPooling, knn_interpolate
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from torch_geometric.nn import GATConv

from torch_geometric.nn import GraphUNet, SAGEConv
from torch_geometric.utils import dropout_adj
import torch_geometric.transforms as T
from torch_geometric.nn import ARMAConv
from torch_geometric.nn import GINConv, global_add_pool
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
import numpy as np
from sklearn.neighbors import NearestNeighbors


class Kipfblock(torch.nn.Module):
    def __init__(self, n_input, n_hidden=64, K=8, p=0.5, bn=False):
        super(Kipfblock, self).__init__()
        self.conv1 = ChebConv(n_input, n_hidden, K=K)
        self.p = p
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.do_bn = bn
        if bn:
            self.bn = torch.nn.BatchNorm1d(n_hidden)

    def forward(self, x, edge_index):
        if self.do_bn:
            x = F.relu(self.bn(self.conv1(x, edge_index)))
        else:
            x = F.relu(self.conv1(x, edge_index))

        x = F.dropout(x, training=self.training, p=self.p)

        return x


class KipfNet(torch.nn.Module):
    def __init__(self, num_features, num_classes, nh1=64, K=8, K_mix=2,
                     cached=True, inout_skipconn=False):
        super(KipfNet, self).__init__()
        self.inout_skipconn = inout_skipconn
        self.Kipfblock1 = Kipfblock(n_input=num_features, n_hidden=nh1, K=K)

        if inout_skipconn:
            self.conv_mix = ChebConv(nh1+num_features, num_classes, K=K_mix)
        else: 
            self.conv_mix = ChebConv(nh1, num_classes, K=K_mix)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.Kipfblock1(x,edge_index)

        if self.inout_skipconn:
            x = torch.cat((x, data.x),1)
            x = self.conv_mix(x, edge_index)
        else:
            x = self.conv_mix(x, edge_index)
        
        return x

class KipfNetd2(torch.nn.Module):
    def __init__(self, num_features, num_classes, nh1=2, nh2=2, K=2, K_mix=1,
                     cached=True, inout_skipconn=True):
        super(KipfNetd2, self).__init__()
        self.inout_skipconn = inout_skipconn
        self.Kipfblock1 = Kipfblock(n_input=num_features, n_hidden=nh1, K=K)
        self.Kipfblock2 = Kipfblock(n_input=nh1, n_hidden=nh2, K=K)

        if inout_skipconn:
            self.conv_mix = ChebConv(nh2+num_features, num_classes, K=K_mix)
        else: 
            self.conv_mix = ChebConv(nh2, num_classes, K=K_mix)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.Kipfblock1(x,edge_index)
        x = self.Kipfblock2(x,edge_index)

        if self.inout_skipconn:
            x = torch.cat((x, data.x),1)
            x = self.conv_mix(x, edge_index)
        else:
            x = self.conv_mix(x, edge_index)
        
        return x


class KipfNet_resd2(torch.nn.Module):
    def __init__(self, num_features, num_classes, nh1=64, nh2=32, K=8, K_mix=2,
                     cached=True, inout_skipconn=True):
        super(KipfNet_resd2, self).__init__()
        self.inout_skipconn = inout_skipconn

        self.Kipfblock1 = Kipfblock(n_input=num_features, n_hidden=nh1, K=K)
        self.Kipfblock2 = Kipfblock(n_input=nh1, n_hidden=nh2, K=K)

        self.skip_project1 = ChebConv(in_channels=self.Kipfblock1.n_input, 
                                    out_channels=self.Kipfblock1.n_hidden, K=1)

        self.skip_project2 = ChebConv(in_channels=self.Kipfblock2.n_input, 
                                    out_channels=self.Kipfblock2.n_hidden, K=1)


        if inout_skipconn:
            self.conv_mix = ChebConv(nh2+num_features, num_classes, K=K_mix)
        else: 
            self.conv_mix = ChebConv(nh2, num_classes, K=K_mix)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.Kipfblock1(x,edge_index) + self.skip_project1(x,edge_index)
        x = self.Kipfblock2(x,edge_index) + self.skip_project2(x,edge_index)

        if self.inout_skipconn:
            x = torch.cat((x, data.x),1)
            x = self.conv_mix(x, edge_index)
        else:
            x = self.conv_mix(x, edge_index)
        
        return x



class Graph_resnet(torch.nn.Module):
    def __init__(self, num_features, num_classes, nh=38, K=6, K_mix=2,
                    inout_skipconn=True, depth=3, p=0.5, bn=False):
        super(Graph_resnet, self).__init__()
        self.inout_skipconn = inout_skipconn
        self.depth = depth

        self.Kipfblock_list = nn.ModuleList()
        self.skipproject_list = nn.ModuleList()

        if isinstance(nh, list): 
            # if you give every layer a differnt number of channels
            # you need one number of channels for every layer!
            assert len(nh) == depth

        else:
            channels = nh
            nh = []
            for i in range(depth):
                nh.append(channels)

        for i in range(depth):
            if i == 0:
                self.Kipfblock_list.append(Kipfblock(n_input=num_features, 
                    n_hidden=nh[0], K=K, p=p, bn=bn))
                self.skipproject_list.append(ChebConv(num_features, nh[0], K=1))
            else:
                self.Kipfblock_list.append(Kipfblock(n_input=nh[i-1],
                                             n_hidden=nh[i], K=K, p=p, bn=bn))
                self.skipproject_list.append(ChebConv(nh[i-1], nh[i], K=1))

        if inout_skipconn:
            self.conv_mix = ChebConv(nh[-1]+num_features, num_classes, K=K_mix)
        else: 
            self.conv_mix = ChebConv(nh[-1], num_classes, K=K_mix)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        for i in range(self.depth):
            x = self.Kipfblock_list[i](x,edge_index) + \
                self.skipproject_list[i](x,edge_index)

        if self.inout_skipconn:
            x = torch.cat((x, data.x),1)
            x = self.conv_mix(x, edge_index)
        else:
            x = self.conv_mix(x, edge_index)
        
        return x

class KipfNet_old(torch.nn.Module):
    def __init__(self, num_features, num_classes, nh1=64, K=8):
        super(KipfNet_old, self).__init__()
#        self.conv1 = GCNConv(n_features, 60, cached=True)
#        self.conv2 = GCNConv(60, n_classes, cached=True)
        self.conv1 = ChebConv(num_features, nh1, K=K)
        self.conv2 = ChebConv(nh1, num_classes, K=K)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return x

# class KipfNet_resd2_old(torch.nn.Module):
#     def __init__(self, num_features, num_classes, nh1=64, nh2=32, K=8, K_mix=2,
#                      cached=True, inout_skipconn=True, intermediate_skipconn=False):
#         super(KipfNet_resd2, self).__init__()
#         self.inout_skipconn = inout_skipconn
#         self.intermediate_skipconn = intermediate_skipconn

#         self.Kipfblock1 = Kipfblock(n_input=num_features, n_hidden=nh1, K=K)

#         if intermediate_skipconn:
#             self.Kipfblock2 = Kipfblock(n_input=nh1+num_features, n_hidden=nh2, K=K)
#         else:
#             self.Kipfblock2 = Kipfblock(n_input=nh1, n_hidden=nh2, K=K)


#         if inout_skipconn:
#             self.conv_mix = ChebConv(nh1+nh2+num_features, num_classes, K=K_mix)
#         else: 
#             self.conv_mix = ChebConv(nh1+nh2, num_classes, K=K_mix)

#     def forward(self, data):
#         x, edge_index = data.x, data.edge_index

#         x1 = self.Kipfblock1(x,edge_index)

#         if self.intermediate_skipconn:
#             x1_cat = torch.cat((x1, x),1)
#             x2 = self.Kipfblock2(x1_cat,edge_index)
#         else:
#             x2 = self.Kipfblock2(x1,edge_index)

#         if self.inout_skipconn:
#             x3 = torch.cat((x1, x2, x),1)
#             x3 = self.conv_mix(x3, edge_index)
#         else:
#             x3 = torch.cat((x1, x2),1)
#             x3 = self.conv_mix(x3, edge_index)
        
#         return x3


# class KipfNet_resd3(torch.nn.Module):
#     def __init__(self, num_features, num_classes, nh1=64, nh2=32, K=8, K_mix=2,
#                      cached=True, inout_skipconn=True, intermediate_skipconn=True):
#         super(KipfNet_resd3, self).__init__()
#         self.inout_skipconn = inout_skipconn
#         self.intermediate_skipconn = intermediate_skipconn

#         self.Kipfblock1 = Kipfblock(n_input=num_features, n_hidden=nh1, K=K)

#         if intermediate_skipconn:
#             self.Kipfblock2 = Kipfblock(n_input=nh1+num_features, n_hidden=nh2, K=K)
#         else:
#             self.Kipfblock2 = Kipfblock(n_input=nh1, n_hidden=nh2, K=K)


#         if inout_skipconn:
#             self.conv_mix = ChebConv(nh1+nh2+num_features, num_classes, K=K_mix)
#         else: 
#             self.conv_mix = ChebConv(nh1+nh2, num_classes, K=K_mix)

#     def forward(self, data):
#         x, edge_index = data.x, data.edge_index

#         x1 = self.Kipfblock1(x,edge_index)

#         if intermediate_skipconn:
#             x1_cat = torch.cat((x1, x),1)
#             x2 = self.Kipfblock2(x1_cat,edge_index)
#         else:
#             x2 = self.Kipfblock2(x1,edge_index)

#         if self.inout_skipconn:
#             x3 = torch.cat((x1, x2, x),1)
#             x3 = self.conv_mix(x3, edge_index)
#         else:
#             x3 = torch.cat((x1, x2),1)
#             x3 = self.conv_mix(x3, edge_index)
        
#         return x3