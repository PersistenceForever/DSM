import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import BaseRGCN
from losses import ConLoss
from util import Dataset
from torch.utils.data import DataLoader
import torch.utils.data as Data
import numpy as np
import sys

class RGCN_GCL(nn.Module):
    def __init__(self, num_layers, num_mlp_layers, input_dim, hidden_dim, neighbor_pooling_type, device):
        super(RGCN_GCL, self).__init__()
        self.rgcn = BaseRGCN(num_layers, num_mlp_layers, input_dim, hidden_dim, neighbor_pooling_type, device)
        self.criterion = ConLoss(device)       

    def forward(self, seq1, adj, train_nodeSet, pathDict, bsz):
        h_1 = F.normalize(self.rgcn(seq1, adj), dim =1) #gin's input is adj and a feature 
        trainSubGraphEmd = []
        for index in train_nodeSet:               
            trainSubGraphEmd.append(torch.mean(h_1[index], dim = 0))
        # obtain each graph's embedding by h_1        
        trainSubGraphEmd = torch.stack(trainSubGraphEmd)
        
        viewsList = []
        for idx in range(len(train_nodeSet)):
            views_id = pathDict[idx][0]
            views_id.append(idx)            
            viewsList.append(trainSubGraphEmd[views_id])            
        viewsList = torch.stack(viewsList)
        
        #feats is a batchsize*n_views*d, n_views = k +1
        dataset = Dataset(viewsList)      
        feats_loader = DataLoader(dataset= dataset, batch_size = bsz, shuffle = True)
        
        loss = 0
        for step, feats in enumerate(feats_loader):
            loss_tmp = self.criterion(feats)
            loss += loss_tmp
            # print("loss:", loss_tmp.item())
        return loss/(step+1)

    def get_emb(self, seq1, adj):
        h_1 = self.rgcn(seq1, adj)
        return h_1.detach().cpu().numpy()
