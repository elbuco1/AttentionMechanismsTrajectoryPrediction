import torch
import torch.nn as nn
import torch.nn.functional as f 
import random
import numpy as np 
import torchvision
import imp
import time
import copy


class MultiHeadAttention(nn.Module):
    # dk = dv = dmodel/h
    def __init__(self,device,dmodel,h,mha_dropout):
        super(MultiHeadAttention,self).__init__()
        self.device = device
        self.mha = nn.MultiheadAttention(dmodel,h,mha_dropout)
        self.h = h

    def forward(self,q,k,v,points_mask = None):
        if points_mask is not None:
            mask = self.get_mask(points_mask).to(self.device)
        else:
            mask = None

        
        q = q.permute(1,0,2)
        k = k.permute(1,0,2)
        v = v.permute(1,0,2)

        att,wgts = self.mha(q,k,v,key_padding_mask =  mask)      
        att = att.permute(1,0,2)
        return att #B,Nmax,dv

    def get_mask(self,points_mask):
        if len(points_mask.shape) < 4:
            print("please input size must be [b,n,s,i]")
            points_mask = np.expand_dims(points_mask, axis = 1)

        mha_mask = (np.sum(points_mask.reshape(points_mask.shape[0],points_mask.shape[1],-1), axis = 2) == 0).astype(int)
        return torch.ByteTensor(mha_mask).detach()

class EncoderLayer(nn.Module):
    def __init__(self,device,dmodel,h,mha_dropout, fforward_dim):
        super(EncoderLayer,self).__init__()

        self.mha = MultiHeadAttention(device,dmodel,h,mha_dropout)
        self.linear1 = nn.Linear(dmodel, fforward_dim)
        self.dropout = nn.Dropout(mha_dropout)
        self.linear2 = nn.Linear(fforward_dim, dmodel)

        self.norm1 = nn.LayerNorm(dmodel)
        self.norm2 = nn.LayerNorm(dmodel)
        self.dropout1 = nn.Dropout(mha_dropout)
        self.dropout2 = nn.Dropout(mha_dropout)

        self.device = device
    def forward(self,q,k,v,points_mask = None):
        x = self.mha(q,k,v,points_mask)
        x = q + self.dropout1(x)
        x = self.norm1(x)
        x1 = f.relu(self.linear1(x))
        x1 = self.dropout(x1)
        x1 = f.relu(self.linear2(x1))
        x = self.dropout2(x1) + x
        x = self.norm2(x)
        return x

class Encoder(nn.Module):
    def __init__(self,encoder_layer, num_layers):
        super(Encoder,self).__init__()
        self.layers = self._get_clones(encoder_layer,num_layers)
        self.num_layers = num_layers
    
    def forward(self, q, k, v, points_mask = None):
        
        output = self.layers[0](q,k,v,points_mask)

        for i in range(1,self.num_layers):
            output = self.layers[i](output, output, output)
        return output

    
    def _get_clones(self,module, N):
        #https://pytorch.org/docs/master/_modules/torch/nn/modules/transformer.html
        return nn.ModuleList([copy.deepcopy(module) for i in range(N)])





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
  
        # mask
        if mask is not None:
            comp_q_v = comp_q_v.masked_fill(mask.unsqueeze(1), float('-inf') )
        
        # softmax
        weights = f.softmax(comp_q_v,dim = 2 ) # B,Nq,Nk

        # matmul
        att = torch.bmm(weights,v) # B,Nq,Nk * B,Nk,dmodel -> B,Nq,dmodel
        return att
    

class SoftAttention(nn.Module):
    # dk = dv = dmodel/h
    def __init__(self,device,dmodel,projection_layers ,dropout = 0.1):
        super(SoftAttention,self).__init__()
        self.device = device
        self.projection_layers = projection_layers
        
        self.mlp_attention = LinearProjection(device,dmodel,self.projection_layers,dropout = dropout)

    def forward(self,q,k,v,points_mask = None):
        if points_mask is not None:
            mask = self.get_mask(points_mask).to(self.device)

        else:
            mask = None

        att = self.mlp_attention(q,k,v,mask)
        return att #B,Nmax,dv
    

    def get_mask(self,points_mask):
        if len(points_mask.shape) < 4:
            print("please input size must be [b,n,s,i]")
            points_mask = np.expand_dims(points_mask, axis = 1)

        mha_mask = (np.sum(points_mask.reshape(points_mask.shape[0],points_mask.shape[1],-1), axis = 2) == 0).astype(int)
        return torch.ByteTensor(mha_mask).detach()








# mask = self.get_mask(points_mask,q.size()[1]).to(self.device)

# def get_mask(self,points_mask,max_batch,multiquery = 0):
    #     # on met des 1 pour le poids entre un agent actif en ligne et un agent inactif en colonne
    #     # pour le cas de l'agent inactif en ligne, peu importe il ne sera pas utilisé pour
    #     # la rétropropagation

    #     if len(points_mask.shape) < 4:
    #         print("please input size must be [b,n,s,i]")
    #         points_mask = np.expand_dims(points_mask, axis = 1)

    #     sample_sum = (np.sum(points_mask.reshape(points_mask.shape[0],points_mask.shape[1],-1), axis = 2) > 0).astype(int)
    #     a = np.repeat(np.expand_dims(sample_sum,axis = 2),max_batch,axis = -1)
    #     b = np.transpose(a,axes=(0,2,1))
    #     mha_mask = np.logical_and(np.logical_xor(a,b),a).astype(int)

    #     if not multiquery:
    #         mha_mask = np.expand_dims(mha_mask[:,0],1)
    #     return torch.ByteTensor(mha_mask).detach()