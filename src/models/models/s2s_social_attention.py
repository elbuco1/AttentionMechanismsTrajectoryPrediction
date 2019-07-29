import torch
import torch.nn as nn
import torch.nn.functional as f
import random
import numpy as np
import torchvision
import imp
import time
# from models.soft_attention import SoftAttention
import models.soft_attention as soft_attention


class decoderLSTM(nn.Module):
    # def __init__(self,input_size,hidden_size,num_layers,output_size,batch_size,seq_len):

    def __init__(self,device,input_size,dec_hidden_size,num_layers):
        super(decoderLSTM,self).__init__()

        self.device = device

        self.input_size = input_size

        self.dec_hidden_size = dec_hidden_size
        self.num_layers = num_layers



        self.lstm = nn.LSTM(input_size = input_size,hidden_size = dec_hidden_size,num_layers = num_layers,batch_first = True)
        
    def forward(self,x,hidden):
        output,self.hidden = self.lstm(x,hidden)
        return output, hidden


class encoderLSTM(nn.Module):
    # def __init__(self,input_size,hidden_size,num_layers,output_size,batch_size,seq_len):

    def __init__(self,device,input_size,hidden_size,num_layers):
        super(encoderLSTM,self).__init__()

        self.device = device
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers


        self.lstm = nn.LSTM(input_size = self.input_size,hidden_size = self.hidden_size,num_layers = self.num_layers,batch_first = True)


    # def forward(self,x,x_lengths,nb_max):
    def forward(self,x,x_lengths):

        hidden = self.init_hidden_state(len(x_lengths))
        

        x = torch.nn.utils.rnn.pack_padded_sequence(x, x_lengths, batch_first=True)
        x,hidden = self.lstm(x,hidden)
        x, _ = torch.nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        # hidden, _ = torch.nn.utils.rnn.pad_packed_sequence(hidden, batch_first=True)

        # hidden tuple( B*N enc,B*N enc)
        # B*N enc
        hidden = (hidden[0].permute(1,2,0), hidden[1].permute(1,2,0)) # put batch size first to revert sort by x_lengths
        return x[:,-1,:], hidden



    def init_hidden_state(self,batch_size):
        h_0 = torch.rand(self.num_layers,batch_size,self.hidden_size).to(self.device)
        c_0 = torch.rand(self.num_layers,batch_size,self.hidden_size).to(self.device)

        return (h_0,c_0)



class S2sSocialAtt(nn.Module):
    # def __init__(self,input_size,hidden_size,num_layers,output_size,batch_size,seq_len):

    def __init__(self,args):
        super(S2sSocialAtt,self).__init__()

        self.args = args

        self.device = args["device"]
        self.input_dim = args["input_dim"]
        
        self.enc_hidden_size = args["enc_hidden_size"]
        self.enc_num_layers = args["enc_num_layers"]

        self.dec_hidden_size = self.enc_hidden_size
        self.dec_num_layer = self.enc_num_layers

        self.embedding_size = args["embedding_size"]
        self.output_size = args["output_size"]
        self.pred_length = args["pred_length"]
        self.projection_layers = args["projection_layers"]
        self.encoder_features_embedding = args["enc_feat_embedding"]
        self.condition_decoder_on_outputs = args["condition_decoder_on_outputs"]
        self.joint_optimisation = args["joint_optimisation"]

        # assert(self.enc_hidden_size == self.dec_hidden_size)
        # assert(self.embedding_size == self.encoder_features_embedding)

        self.coordinates_embedding = nn.Linear(self.input_dim,self.embedding_size) # input 2D coordinates to embedding dim
        self.hdec2coord = nn.Linear(self.dec_hidden_size,self.output_size) # decoder hidden to 2D coordinates space
        self.k_embedding = nn.Linear(self.enc_hidden_size,self.encoder_features_embedding) # embedding enc_hidden_size to dmodel
        self.q_embedding = nn.Linear(self.dec_hidden_size,self.encoder_features_embedding) # embedding dec_hidden_size to dmodel


        self.encoder = encoderLSTM(self.device,self.embedding_size,self.enc_hidden_size,self.enc_num_layers)

        # self.encoder2decoder = nn.Linear(self.enc_hidden_size,self.dec_hidden_size)

        if self.condition_decoder_on_outputs:
            self.decoder = decoderLSTM(self.device,self.encoder_features_embedding + self.embedding_size,self.dec_hidden_size,self.dec_num_layer)
        else:
            self.decoder = decoderLSTM(self.device,self.encoder_features_embedding ,self.dec_hidden_size,self.dec_num_layer)


        self.attention = soft_attention.SoftAttention(self.device,self.encoder_features_embedding,self.projection_layers)

    def forward(self,x):
        active_agents = x[2]
        points_mask = x[3][1]
        points_mask_in = x[3][0]
        x = x[0] # B N S 2


        ##### Dynamic part ####################################
        # embed coordinates
        x_e = self.coordinates_embedding(x) # B N S E
        x_e = f.relu(x_e)
        B,N,S,E = x_e.size()

      

        # get lengths for padding
        x_lengths = np.sum(points_mask_in[:,:,:,0],axis = -1).reshape(B*N)
        x_lengths = np.add(x_lengths, (x_lengths == 0).astype(int)) # put length 1 to padding agent, not efficient but practical
        
        # get the indices of the descending sorted lengths
        arg_ids = list(reversed(np.argsort(x_lengths)))

        # order input vector based on descending sequence lengths
        x_e_sorted = x_e.view(B*N,S,E) [arg_ids]# B*N S E
        
        # get ordered/unpadded sequences lengths for pack_padded object   
        sorted_x_lengths = x_lengths[arg_ids] 
        encoder_hiddens,hidden = self.encoder(x_e_sorted,sorted_x_lengths)

        # reverse ordering of indices
        rev_arg_ids = np.argsort(arg_ids)
        # reverse ordering of encoded sequence

        encoder_hiddens = encoder_hiddens[rev_arg_ids]
        hidden = (hidden[0][rev_arg_ids].permute(2,0,1).contiguous(), hidden[1][rev_arg_ids].permute(2,0,1).contiguous())

        # embed encoder hidden states features
        encoder_hiddens = self.k_embedding(encoder_hiddens)
        encoder_hiddens = f.relu(encoder_hiddens)


        ######################################################
        ######################################################

        
        # set keys and values to embedded values of encoder hidden states
        k = v = encoder_hiddens.view(B,N,self.encoder_features_embedding) 
        k = nn.functional.relu(k)
        v = nn.functional.relu(v)  

        # embedded last point of input sequence
        out = x_e[:,:,-1] # B,N,embedding_size

        ######## Prediction part ##############################
        outputs = []

        for _ in range(self.pred_length):

            ########## Attention #############################
            # set query to last decoder hidden state
            # q = hidden[0].view(B,N,self.enc_hidden_size)
            q = hidden[0][0].view(B,N,self.enc_hidden_size)

            q = self.q_embedding(q) # B N encoder_features_embedding
            q = f.relu(q)


             
            # attention features          
            # att = self.attention(q,k,v,points_mask, self.joint_optimisation) # B N encoder_features_embedding
            att = self.attention(q,k,v,points_mask) # B N encoder_features_embedding
            
            ##################################################
            ####### Prediction ###############################

            if self.condition_decoder_on_outputs:
                in_dec = torch.cat([out,att],dim = 2) # B N embedding_size + encoder_features_embedding (embedding_size == encoder_features_embedding)
            else:
                in_dec = att
            
            in_dec = in_dec.unsqueeze(2) # B N 1 2*encoder_features_embedding
            in_dec = in_dec.view(B*N,1,in_dec.size()[-1]) # B*N 1 2*encoder_features_embedding  (batch,seqlen,feat_size)
            
            
            out,hidden = self.decoder(in_dec ,hidden) # out: B*N 1 dec_hidden_size (dec_hidden == enc_hidden)
            out = self.hdec2coord(out) # B*N 1 2 
            out = out.view(B,N,1,self.input_dim)           
            outputs.append(out) # out: B N 1 2

            if self.condition_decoder_on_outputs:
                out = self.coordinates_embedding(out.detach()) # out: B*N 1 2  2d coordinates to embedding space
                out = f.relu(out)
                out = out.squeeze(2)
            ################################################

        outputs = torch.cat(outputs, dim = 2)

        ####################################################

        return outputs





