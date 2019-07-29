import torch
import torch.nn as nn
import torch.nn.functional as f
import random
import numpy as np
import torchvision
import imp
import time
from models.soft_attention import SoftAttention
# from models.pretrained_vgg import customCNN
from models.pretrained_vgg import customCNN2



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



class S2sSpatialAtt(nn.Module):
    # def __init__(self,input_size,hidden_size,num_layers,output_size,batch_size,seq_len):

    def __init__(self,args):
        super(S2sSpatialAtt,self).__init__()

        self.args = args

        self.device = args["device"]
        self.input_dim = args["input_dim"]
        self.enc_hidden_size = args["enc_hidden_size"]
        self.enc_num_layers = args["enc_num_layers"]
        # self.dec_hidden_size = args["dec_hidden_size"]
        # self.dec_num_layer = args["dec_num_layer"]
        self.dec_hidden_size = self.enc_hidden_size
        self.dec_num_layer = self.enc_num_layers
        self.embedding_size = args["embedding_size"]
        self.output_size = args["output_size"]
        self.pred_length = args["pred_length"]
        self.projection_layers = args["projection_layers"]
        self.att_features_embedding = args["att_feat_embedding"]
        self.spatial_projection = args["spatial_projection"]
        self.condition_decoder_on_outputs = args["condition_decoder_on_outputs"]
        self.joint_optimisation = args["joint_optimisation"]



        # assert(self.enc_hidden_size == self.dec_hidden_size)
        # assert(self.embedding_size == self.encoder_features_embedding)

        self.coordinates_embedding = nn.Linear(self.input_dim,self.embedding_size) # input 2D coordinates to embedding dim
        self.hdec2coord = nn.Linear(self.dec_hidden_size,self.output_size) # decoder hidden to 2D coordinates space
        self.k_embedding = nn.Linear(self.spatial_projection,self.att_features_embedding) # embedding spatial feature to dmodel
        self.q_embedding = nn.Linear(self.dec_hidden_size,self.att_features_embedding) # embedding dec_hidden_size to dmodel


        self.encoder = encoderLSTM(self.device,self.embedding_size,self.enc_hidden_size,self.enc_num_layers)
        
        if self.condition_decoder_on_outputs:
            self.decoder = decoderLSTM(self.device,self.att_features_embedding + self.embedding_size,self.dec_hidden_size,self.dec_num_layer)
        else:
            self.decoder = decoderLSTM(self.device,self.att_features_embedding,self.dec_hidden_size,self.dec_num_layer)

            
        
        self.attention = SoftAttention(self.device,self.att_features_embedding,self.projection_layers)

        ##### Spatial part ##############################################

        ############# features ##########################################
        # self.cnn = customCNN(self.device,nb_channels_projection= self.spatial_projection)
        self.cnn = customCNN2(self.device,nb_channels_projection= self.spatial_projection)

        # self.spatt2att = nn.Linear(self.spatial_projection,self.att_features_embedding)


    def forward(self,x):
        active_agents = x[2]
        points_mask = x[3][1]
        points_mask_in = x[3][0]
        imgs = x[4]

        x = x[0] # B N S 2

        ##### Dynamic part ####################################

        # embed coordinates
        x_e = self.coordinates_embedding(x) # B N S E
        x_e = nn.functional.relu(x_e)
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
        _,hidden = self.encoder(x_e_sorted,sorted_x_lengths)

        # reverse ordering of indices
        rev_arg_ids = np.argsort(arg_ids)
        # reverse ordering of encoded sequence

        hidden = (hidden[0][rev_arg_ids].permute(2,0,1).contiguous(), hidden[1][rev_arg_ids].permute(2,0,1).contiguous())

        ### Spatial ##############

        spatial_features = self.cnn(imgs)
        b,f,w,h = spatial_features.size()
        spatial_features = spatial_features.view(b,f,w*h).permute(0,2,1)# B,Nfeaturevectors,spatial projection
        # spatial_features = self.spatt2att(spatial_features)
        spatial_features = nn.functional.relu(spatial_features) # B,Nfeaturevectors,dmodel

        ##########################

        # # set keys and values to embedded values of encoder hidden states
        k = v = self.k_embedding(spatial_features)
        k = nn.functional.relu(k)
        v = nn.functional.relu(v)

        

        # embedded last point of input sequence
        out = x_e[:,:,-1] # B,N,embedding_size

        ######## Prediction part ##############################
        outputs = []

        for _ in range(self.pred_length):
            ########## Attention #############################
            # set query to last decoder hidden state
            q = hidden[0][0].view(B,N,self.dec_hidden_size)
            q = self.q_embedding(q) # B N att_features_embedding
            q = nn.functional.relu(q)


             
            # attention features          
            att = self.attention(q,k,v) # B N att_features_embedding
            ##################################################
            ####### Prediction ###############################

            if self.condition_decoder_on_outputs:

                in_dec = torch.cat([out,att],dim = 2) # B N embedding_size + att_features_embedding (embedding_size == encoder_features_embedding)
            else:
                in_dec = att
            
            
            in_dec = in_dec.unsqueeze(2) # B N 1 2*att_features_embedding
            in_dec = in_dec.view(B*N,1,in_dec.size()[-1]) # B*N 1 2*att_features_embedding  (batch,seqlen,feat_size)
            
            
            out,hidden = self.decoder(in_dec ,hidden) # out: B*N 1 dec_hidden_size (dec_hidden == enc_hidden)
            out = self.hdec2coord(out) # B*N 1 2 
            out = out.view(B,N,1,self.input_dim)           
            outputs.append(out) # out: B N 1 2

            if self.condition_decoder_on_outputs:
                out = self.coordinates_embedding(out.detach()) # out: B*N 1 2  2d coordinates to embedding space
                out = nn.functional.relu(out)
                out = out.squeeze(2)
            ################################################

        outputs = torch.cat(outputs, dim = 2)

        ####################################################
        

        return outputs





