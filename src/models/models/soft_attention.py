import torch
import torch.nn as nn
import torch.nn.functional as f 
import random
import numpy as np 
import torchvision
import imp
import time


class LinearProjection(nn.Module):
    def __init__(self,device,dmodel,projection_layers ,dropout = 0.1):
        super(LinearProjection, self).__init__()
        self.device = device
        self.dmodel = dmodel
        self.projection_layers = projection_layers
        self.projection_weight = []
        self.projection_weight.append(nn.Linear(self.dmodel*2,self.projection_layers[0]))

        self.projection_weight.append(nn.ReLU())

        for i in range(1,len(self.projection_layers)):
            self.projection_weight.append(nn.Linear(self.projection_layers[i-1], self.projection_layers[i]))
            self.projection_weight.append(nn.ReLU())

        self.projection_weight.append(nn.Linear(self.projection_layers[-1], 1))

        self.projection_weight = nn.Sequential(*self.projection_weight)

    def forward(self,q,k,v,mask = None): # B,N,dmodel
        
        _,Nq,_ = q.size()
        _,Nk,_ = k.size()

        q = q.unsqueeze(2).repeat(1,1,Nk,1) # B,Nq,Nk,dmodel 

        k = k.unsqueeze(1).repeat(1,Nq,1,1) # B,Nq,Nk,dmodel 

        comp_q_v = torch.cat([q,k],dim = 3) # B,Nq,Nk,2dmodel 
        comp_q_v = self.projection_weight(comp_q_v).squeeze(3)  # B,Nq,Nk  

        min_inf = float('-inf')       
        # mask
        if mask is not None:
            comp_q_v = comp_q_v.masked_fill(mask,min_inf)
        
        # softmax
        weights = f.softmax(comp_q_v,dim = 2 ) # B,Nq,Nk

        # matmul
        att = torch.bmm(weights,v) # B,Nq,Nk * B,Nk,dmodel -> B,Nq,dmodel
        return att

    def get_mask(self,points_mask,max_batch):
        # compute mask put one where the dot product conv_features*conv_features.T is 0.
        # and the sum over a given row of the resulting matrice is gt 1
        # Nmax = q.size()[1]
        # dot = torch.bmm(q, torch.transpose(k,2,1))
        # mask = (dot == 0) & (torch.sum(dot,dim = 1) > 0.).unsqueeze(2).repeat(1,1,Nmax)
        # mask = mask.to(self.device)
        # return mask
        if max_batch == 1:
            points_mask = np.expand_dims(points_mask, axis = 1)

        sample_sum = (np.sum(points_mask.reshape(points_mask.shape[0],points_mask.shape[1],-1), axis = 2) > 0).astype(int)
        a = np.repeat(np.expand_dims(sample_sum,axis = 2),max_batch,axis = -1)
        b = np.transpose(a,axes=(0,2,1))
        mha_mask = np.logical_and(np.logical_xor(a,b),a).astype(int)
        # eyes = np.expand_dims(np.eye(mha_mask.shape[-1]),0)
        # eyes = eyes.repeat(mha_mask.shape[0],0)

        # mha_mask = np.logical_or(mha_mask,eyes).astype(int)
        return torch.ByteTensor(mha_mask).detach()
        # return torch.ByteTensor(mha_mask).to(self.device)


class SoftAttention(nn.Module):
    # dk = dv = dmodel/h
    def __init__(self,device,dmodel,projection_layers ,dropout = 0.1):
        super(SoftAttention,self).__init__()
        self.device = device
        self.projection_layers = projection_layers
        
        self.mlp_attention = LinearProjection(device,dmodel,self.projection_layers,dropout = dropout)

    def forward(self,q,k,v,points_mask = None):
        if points_mask is not None:
            mask = self.mlp_attention.get_mask(points_mask,q.size()[1]).to(self.device)
            att = self.mlp_attention(q,k,v,mask)
        else:
            att = self.mlp_attention(q,k,v)


        return att #B,Nmax,dv