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

        self.args = args

        # general parameters
        self.device = args["device"]
        self.input_dim =  args["input_dim"]
        self.input_length =  args["input_length"]
        self.output_length =  args["output_length"]
        self.pred_dim =  args["pred_dim"]


        self.dmodel =  args["dmodel"]
        
        self.predictor_layers =  args["predictor_layers"]
        # self.dropout_tfr =  args["dropout_tfr"]


        # self.convnet_nb_layers =  args["convnet_nb_layers"]

        # cnn parameters
        self.nb_conv = args["nb_conv"] # depth
        self.nb_kernel = args["nb_kernel"] # nb kernel per layer
        self.cnn_feat_size = args["cnn_feat_size"] # projection size of output
        self.kernel_size =  args["kernel_size"] 


        #prediction layers
        self.projection_layers = args["projection_layers"]

        self.use_mha = args["use_mha"]
        self.h = args["h"]
        self.mha_dropout = args["mha_dropout"]
        self.joint_optimisation = args["joint_optimisation"]
        



############# x/y embedding ###############################
        # self.coord_embedding = nn.Linear(self.input_dim,self.coordinates_embedding_size)
############# cnn #########################################
        # compute nb temporal blocks

       
        # self.nb_temporal_blocks = self.__get_nb_blocks(self.input_length,self.kernel_size)        
        # self.num_channels = [self.convnet_embedding for _ in range(self.nb_temporal_blocks)]

        # init network
        # self.tcn = TemporalConvNet(self.device, self.coordinates_embedding, self.num_channels, self.kernel_size, self.dropout_tcn)

        self.cnn = CNN(num_inputs = self.input_dim,nb_kernel = self.nb_kernel,cnn_feat_size = self.cnn_feat_size,obs_len = self.input_length ,kernel_size = self.kernel_size,nb_conv = self.nb_conv)



        # project conv features to dmodel
        self.conv2att = nn.Linear(self.cnn_feat_size,self.dmodel)

        # self.conv2pred = nn.Linear(self.input_length*self.convnet_embedding,self.dmodel)
        self.conv2pred = nn.Linear(self.cnn_feat_size,self.dmodel)

        if self.use_mha:
            self.soft = soft_attention.MultiHeadAttention(self.device,self.dmodel,self.h,self.mha_dropout)
        else:
            self.soft = soft_attention.SoftAttention(self.device,self.dmodel,self.projection_layers,self.mha_dropout)

############# Predictor #########################################

        self.predictor = []
        self.predictor.append(nn.Linear(self.dmodel*2,self.predictor_layers[0]))



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


        att_feat = self.soft(x.clone(),x.clone(),x.clone(),points_mask, self.joint_optimisation)# B,Nmax,dmodel
        if not self.joint_optimisation:
            conv_features = conv_features[:,0].unsqueeze(1)



        # conv_features = self.conv2pred(conv_features)
        conv_features = self.conv2pred(conv_features)

        conv_features = f.relu(conv_features)

        y = torch.cat([att_feat,conv_features],dim = 2 ) # B,Nmax,2*dmodel



   
        y = self.predictor(y)

   

        t_pred = int(self.pred_dim/float(self.input_dim))
        # print(t_pred,self.pred_dim,self.input_dim)

        if self.joint_optimisation:
            y = y.view(B,Nmax,t_pred,self.input_dim) #B,Nmax,Tpred,Nfeat
        else:
            y = y.view(B,1,t_pred,self.input_dim) #B,Nmax,Tpred,Nfeat


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



