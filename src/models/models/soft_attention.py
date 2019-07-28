import torch
import torch.nn as nn
import torch.nn.functional as f 
import random
import numpy as np 
import torchvision
import imp
import time


class MultiHeadAttention(nn.Module):
    # dk = dv = dmodel/h
    def __init__(self,device,dmodel,h,mha_dropout):
        super(MultiHeadAttention,self).__init__()
        self.device = device
        self.mha = nn.MultiheadAttention(dmodel,h,mha_dropout)
        self.mha = MultiheadAttentionSrc(dmodel,h,mha_dropout)


        MultiheadAttentionSrc

    def forward(self,q,k,v,points_mask = None, multiquery = 0):
        if points_mask is not None:
            mask = self.get_mask(points_mask,q.size()[1]).to(self.device)
        else:
            mask = None

        if not multiquery:
            q = q[:,0].unsqueeze(1)
        q = q.permute(1,0,2)
        k = k.permute(1,0,2)
        v = v.permute(1,0,2)

        # att = self.mha(q,k,v,attn_mask =  mask)
        att,wgts = self.mha(q,k,v)
       
        att = att.permute(1,0,2)

        return att #B,Nmax,dv

    def get_mask(self,points_mask,max_batch,multiquery = 0):
        # on met des 1 pour le poids entre un agent actif en ligne et un agent inactif en colonne
        # pour le cas de l'agent inactif en ligne, peu importe il ne sera pas utilisé pour
        # la rétropropagation
        if max_batch == 1:
            points_mask = np.expand_dims(points_mask, axis = 1)

        sample_sum = (np.sum(points_mask.reshape(points_mask.shape[0],points_mask.shape[1],-1), axis = 2) > 0).astype(int)
        a = np.repeat(np.expand_dims(sample_sum,axis = 2),max_batch,axis = -1)
        b = np.transpose(a,axes=(0,2,1))
        mha_mask = np.logical_and(np.logical_xor(a,b),a).astype(int)
        # mha_mask = np.logical_xor(a,b).astype(int)

        # eyes = np.expand_dims(np.eye(mha_mask.shape[-1]),0)
        # eyes = eyes.repeat(mha_mask.shape[0],0)

        # mha_mask = np.logical_or(mha_mask,eyes).astype(int)
        if not multiquery:
            mha_mask = np.expand_dims(mha_mask[:,0],1)
        return torch.ByteTensor(mha_mask).detach()

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
    
    

    
        # return torch.ByteTensor(mha_mask).to(self.device)


class SoftAttention(nn.Module):
    # dk = dv = dmodel/h
    def __init__(self,device,dmodel,projection_layers ,dropout = 0.1):
        super(SoftAttention,self).__init__()
        self.device = device
        self.projection_layers = projection_layers
        
        self.mlp_attention = LinearProjection(device,dmodel,self.projection_layers,dropout = dropout)

    def forward(self,q,k,v,points_mask = None, multiquery = 0):
        if points_mask is not None:
            mask = self.get_mask(points_mask,q.size()[1]).to(self.device)
        else:
            mask = None

        if not multiquery:
            q = q[:,0].unsqueeze(1)
        att = self.mlp_attention(q,k,v,mask)


        return att #B,Nmax,dv

    def get_mask(self,points_mask,max_batch,multiquery = 0):
        # on met des 1 pour le poids entre un agent actif en ligne et un agent inactif en colonne
        # pour le cas de l'agent inactif en ligne, peu importe il ne sera pas utilisé pour
        # la rétropropagation
        if max_batch == 1:
            points_mask = np.expand_dims(points_mask, axis = 1)

        sample_sum = (np.sum(points_mask.reshape(points_mask.shape[0],points_mask.shape[1],-1), axis = 2) > 0).astype(int)
        a = np.repeat(np.expand_dims(sample_sum,axis = 2),max_batch,axis = -1)
        b = np.transpose(a,axes=(0,2,1))
        mha_mask = np.logical_and(np.logical_xor(a,b),a).astype(int)
        # mha_mask = np.logical_xor(a,b).astype(int)

        # eyes = np.expand_dims(np.eye(mha_mask.shape[-1]),0)
        # eyes = eyes.repeat(mha_mask.shape[0],0)

        # mha_mask = np.logical_or(mha_mask,eyes).astype(int)
        if not multiquery:
            mha_mask = np.expand_dims(mha_mask[:,0],1)
        return torch.ByteTensor(mha_mask).detach()













####################################################################"

from torch.nn.init import xavier_uniform_
from torch.nn.init import constant_
from torch.nn.init import xavier_normal_
from torch.nn.parameter import Parameter
from torch.nn import functional as F

class MultiheadAttentionSrc(nn.Module):

    def __init__(self, embed_dim, num_heads, dropout=0., bias=True, add_bias_kv=False, add_zero_attn=False):
        super(MultiheadAttentionSrc, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5

        self.in_proj_weight = Parameter(torch.empty(3 * embed_dim, embed_dim))
        if bias:
            self.in_proj_bias = Parameter(torch.empty(3 * embed_dim))
        else:
            self.register_parameter('in_proj_bias', None)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        if add_bias_kv:
            self.bias_k = Parameter(torch.empty(1, 1, embed_dim))
            self.bias_v = Parameter(torch.empty(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self._reset_parameters()

    def _reset_parameters(self):
        xavier_uniform_(self.in_proj_weight[:self.embed_dim, :])
        xavier_uniform_(self.in_proj_weight[self.embed_dim:(self.embed_dim * 2), :])
        xavier_uniform_(self.in_proj_weight[(self.embed_dim * 2):, :])

        xavier_uniform_(self.out_proj.weight)
        if self.in_proj_bias is not None:
            constant_(self.in_proj_bias, 0.)
            constant_(self.out_proj.bias, 0.)
        if self.bias_k is not None:
            xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            xavier_normal_(self.bias_v)

    def forward(self, query, key, value, key_padding_mask=None, incremental_state=None,
                need_weights=True, static_kv=False, attn_mask=None):
        """
        Inputs of forward function
            query: [target length, batch size, embed dim]
            key: [sequence length, batch size, embed dim]
            value: [sequence length, batch size, embed dim]
            key_padding_mask: if True, mask padding based on batch size
            incremental_state: if provided, previous time steps are cashed
            need_weights: output attn_output_weights
            static_kv: key and value are static

        Outputs of forward function
            attn_output: [target length, batch size, embed dim]
            attn_output_weights: [batch size, target length, sequence length]
        """
        qkv_same = query.data_ptr() == key.data_ptr() == value.data_ptr()
        kv_same = key.data_ptr() == value.data_ptr()

        tgt_len, bsz, embed_dim = query.size()
        assert embed_dim == self.embed_dim
        assert list(query.size()) == [tgt_len, bsz, embed_dim]
        assert key.size() == value.size()

        if incremental_state is not None:
            saved_state = self._get_input_buffer(incremental_state)
            if 'prev_key' in saved_state:
                # previous time steps are cached - no need to recompute
                # key and value if they are static
                if static_kv:
                    assert kv_same and not qkv_same
                    key = value = None
        else:
            saved_state = None

        if qkv_same:
            # self-attention
            q, k, v = self._in_proj_qkv(query)
        elif kv_same:
            # encoder-decoder attention
            q = self._in_proj_q(query)
            if key is None:
                assert value is None
                k = v = None
            else:
                k, v = self._in_proj_kv(key)
        else:
            q = self._in_proj_q(query)
            k = self._in_proj_k(key)
            v = self._in_proj_v(value)
        q *= self.scaling

        if self.bias_k is not None:
            assert self.bias_v is not None
            k = torch.cat([k, self.bias_k.repeat(1, bsz, 1)])
            v = torch.cat([v, self.bias_v.repeat(1, bsz, 1)])
            if attn_mask is not None:
                attn_mask = torch.cat([attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1)
            if key_padding_mask is not None:
                key_padding_mask = torch.cat(
                    [key_padding_mask, key_padding_mask.new_zeros(key_padding_mask.size(0), 1)], dim=1)

        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if k is not None:
            k = k.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if v is not None:
            v = v.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        if saved_state is not None:
            # saved states are stored with shape (bsz, num_heads, seq_len, head_dim)
            if 'prev_key' in saved_state:
                prev_key = saved_state['prev_key'].view(bsz * self.num_heads, -1, self.head_dim)
                if static_kv:
                    k = prev_key
                else:
                    k = torch.cat((prev_key, k), dim=1)
            if 'prev_value' in saved_state:
                prev_value = saved_state['prev_value'].view(bsz * self.num_heads, -1, self.head_dim)
                if static_kv:
                    v = prev_value
                else:
                    v = torch.cat((prev_value, v), dim=1)
            saved_state['prev_key'] = k.view(bsz, self.num_heads, -1, self.head_dim)
            saved_state['prev_value'] = v.view(bsz, self.num_heads, -1, self.head_dim)

            self._set_input_buffer(incremental_state, saved_state)

        src_len = k.size(1)

        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len

        if self.add_zero_attn:
            src_len += 1
            k = torch.cat([k, k.new_zeros((k.size(0), 1) + k.size()[2:])], dim=1)
            v = torch.cat([v, v.new_zeros((v.size(0), 1) + v.size()[2:])], dim=1)
            if attn_mask is not None:
                attn_mask = torch.cat([attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1)
            if key_padding_mask is not None:
                key_padding_mask = torch.cat(
                    [key_padding_mask, torch.zeros(key_padding_mask.size(0), 1).type_as(key_padding_mask)], dim=1)

        attn_output_weights = torch.bmm(q, k.transpose(1, 2))
        assert list(attn_output_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0)
            attn_output_weights += attn_mask

        if key_padding_mask is not None:
            attn_output_weights = attn_output_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_output_weights = attn_output_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float('-inf'),
            )
            attn_output_weights = attn_output_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_output_weights = F.softmax(
            attn_output_weights.float(), dim=-1,
            dtype=torch.float32 if attn_output_weights.dtype == torch.float16 else attn_output_weights.dtype)
        attn_output_weights = F.dropout(attn_output_weights, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_output_weights, v)
        assert list(attn_output.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]
        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn_output = self.out_proj(attn_output)

        if need_weights:
            # average attention weights over heads
            attn_output_weights = attn_output_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_output_weights = attn_output_weights.sum(dim=1) / self.num_heads
        else:
            attn_output_weights = None

        return attn_output, attn_output_weights


    def _in_proj_qkv(self, query):
        return self._in_proj(query).chunk(3, dim=-1)

    def _in_proj_kv(self, key):
        return self._in_proj(key, start=self.embed_dim).chunk(2, dim=-1)

    def _in_proj_q(self, query):
        return self._in_proj(query, end=self.embed_dim)

    def _in_proj_k(self, key):
        return self._in_proj(key, start=self.embed_dim, end=2 * self.embed_dim)

    def _in_proj_v(self, value):
        return self._in_proj(value, start=2 * self.embed_dim)

    def _in_proj(self, input, start=0, end=None):
        weight = self.in_proj_weight
        bias = self.in_proj_bias
        weight = weight[start:end, :]
        if bias is not None:
            bias = bias[start:end]
        return F.linear(input, weight, bias)