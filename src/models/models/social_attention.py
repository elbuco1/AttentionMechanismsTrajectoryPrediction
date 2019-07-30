import torch
import torch.nn as nn
import torch.nn.functional as f 
import random
import numpy as np 
import torchvision
import time
from models.cnn import CNN


# from classes.transformer import Transformer,MultiHeadAttention
# from classes.tcn import TemporalConvNet

# from models.soft_attention import SoftAttention
# from models.soft_attention import MultiHeadAttention

import models.soft_attention as soft_attention



class SocialAttention(nn.Module):
    def __init__(self,args):
        super(SocialAttention,self).__init__()

        print("Social attention")

        self.args = args

        # general parameters
        self.device = args["device"]
        self.input_dim =  args["input_dim"]
        self.input_length =  args["input_length"]
        self.output_length =  args["output_length"]
        self.pred_dim =  args["pred_dim"]


        self.dmodel =  args["dmodel"]
        
        self.predictor_layers =  args["predictor_layers"]
        # cnn parameters
        self.nb_conv = args["nb_conv"] # depth
        self.nb_kernel = args["nb_kernel"] # nb kernel per layer
        self.cnn_feat_size = args["cnn_feat_size"] # projection size of output
        self.kernel_size =  args["kernel_size"] 


        #prediction layers
        self.projection_layers = args["projection_layers"]
        self.tfr_feed_forward_dim = args["tfr_feed_forward_dim"]
        self.tfr_num_layers = args["tfr_num_layers"]


        self.use_mha = args["use_mha"]
        self.h = args["h"]
        self.mha_dropout = args["mha_dropout"]
        self.joint_optimisation = args["joint_optimisation"]

        self.condition_on_trajectory = args["condition_on_trajectory"]
        

        self.cnn = CNN(num_inputs = self.input_dim,nb_kernel = self.nb_kernel,cnn_feat_size = self.cnn_feat_size,obs_len = self.input_length ,kernel_size = self.kernel_size,nb_conv = self.nb_conv)
        self.conv2att = nn.Linear(self.cnn_feat_size,self.dmodel)

        if self.use_mha == 1:
            print("----Multihead attention")
            self.soft = soft_attention.MultiHeadAttention(self.device,self.dmodel,self.h,self.mha_dropout)
            
        elif self.use_mha == 2:
            print("----Transformer encoder")
            encoder_layer = soft_attention.EncoderLayer(self.device,self.dmodel,self.h,self.mha_dropout, self.tfr_feed_forward_dim)
            self.soft = soft_attention.Encoder(encoder_layer,self.tfr_num_layers)

        else:
            print("----Soft attention")
            self.soft = soft_attention.SoftAttention(self.device,self.dmodel,self.projection_layers,self.mha_dropout)

############# Predictor #########################################

        self.predictor = []

        if self.condition_on_trajectory:
            print("----Conditioning attention result with encoded input trajectory")
            self.conv2pred = nn.Linear(self.cnn_feat_size,self.dmodel)
            self.predictor.append(nn.Linear(self.dmodel*2,self.predictor_layers[0]))
        else:
            print("----Not conditioning attention result with encoded input trajectory")
            self.predictor.append(nn.Linear(self.dmodel,self.predictor_layers[0]))




        self.predictor.append(nn.ReLU())

        for i in range(1,len(self.predictor_layers)):
            self.predictor.append(nn.Linear(self.predictor_layers[i-1], self.predictor_layers[i]))
            self.predictor.append(nn.ReLU())

        self.predictor.append(nn.Linear(self.predictor_layers[-1], self.pred_dim))

        self.predictor = nn.Sequential(*self.predictor)





    def forward(self,x):

        active_agents = x[2]
        points_mask = x[3][1]
        x = x[0]

        # permute channels and sequence length
        B,Nmax,Tobs,Nfeat = x.size()
        x = x.permute(0,1,3,2)  # B,Nmax,Nfeat,Tobs # à vérifier
        x = x.view(-1,x.size()[2],x.size()[3]) # [B*Nmax],Nfeat,Tobs


        # get ids for real agents
        # generate vector of zeros which size is the same as net output size
        # send only in the net the active agents
        # set the output values of the active agents to zeros tensor
      
        y = torch.zeros(B*Nmax,self.cnn_feat_size).to(self.device) # [B*Nmax],Nfeat,

        y[active_agents] = self.cnn(x[active_agents]) # [B*Nmax],Nfeat,Tobs

        conv_features = y.view(B,Nmax,y.size()[1]).contiguous() # B,Nmax,Nfeat


        x = self.conv2att(conv_features) # B,Nmax,dmodel    
        x = f.relu(x)

        if not self.joint_optimisation:
            q = x[:,0].clone().unsqueeze(1)
            conv_features = conv_features[:,0].unsqueeze(1)
        else:
            q = x
        
        att_feat = self.soft(q,x,x,points_mask)# B,Nmax,dmodel


        if self.condition_on_trajectory:
            conv_features = self.conv2pred(conv_features)
            conv_features = f.relu(conv_features)
            y = torch.cat([att_feat,conv_features],dim = 2 ) # B,Nmax,2*dmodel
        else:
            y = att_feat

        y = self.predictor(y)  

        if self.joint_optimisation:
            y = y.view(B,Nmax,self.output_length,self.input_dim) #B,Nmax,Tpred,Nfeat
        else:
            y = y.view(B,1,self.output_length,self.input_dim) #B,Nmax,Tpred,Nfeat
        return y


    def __get_nb_blocks(self,receptieve_field,kernel_size):
        nb_blocks = receptieve_field -1
        nb_blocks /= 2.0*(kernel_size - 1.0)
        nb_blocks += 1.0
        nb_blocks = np.log2(nb_blocks)
        nb_blocks = np.ceil(nb_blocks)

        return int(nb_blocks)

    def __get_active_ids(self,x):
        nb_active = torch.sum( (torch.sum(torch.sum(x,dim = 3),dim = 2) > 0.), dim = 1).to(self.device)
        active_agents = [torch.arange(start = 0, end = n) for n in nb_active]

        return active_agents



