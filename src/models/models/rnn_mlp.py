import torch
import torch.nn as nn
import torch.nn.functional as f 
import numpy as np 



class RNN_MLP(nn.Module):
    def __init__(self,args):
        super(RNN_MLP, self).__init__()

        self.args = args

        self.device = args["device"]
        self.batch_size = args["batch_size"]
        self.input_dim = args["input_dim"]

        self.hidden_size = args["hidden_size"]
        self.recurrent_layer = args["recurrent_layer"]
        self.mlp_layers = args["mlp_layers"]
        self.output_size = args["output_size"]

        self.encoder = nn.LSTM(input_size = self.input_dim,hidden_size = self.hidden_size,num_layers = self.recurrent_layer,batch_first = True)

        self.mlp = nn.Sequential()

        

        self.mlp.add_module("layer0",nn.Linear(self.hidden_size,self.mlp_layers[0]))


        self.mlp.add_module("relu0",  nn.ReLU())
        for i in range(1,len(self.mlp_layers)):
            self.mlp.add_module("layer{}".format(i),nn.Linear(self.mlp_layers[i-1],self.mlp_layers[i]))
            self.mlp.add_module("relu{}".format(i), nn.ReLU())

        self.mlp.add_module("layer{}".format(len(self.mlp_layers)), nn.Linear(self.mlp_layers[-1],self.output_size) )

        

        

    def forward(self,x):
        x = x[0]
        x = x.squeeze(1)
      

        h = self.init_hidden_state(x.size()[0])
        output,h = self.encoder(x,h)
        output = output[:,-1]

        x = self.mlp(output).view(x.size()[0],1,int(self.output_size/self.input_dim),self.input_dim)       
        return x

    def init_hidden_state(self,batch_size):
        # print(self.batch_size)
        h_0 = torch.rand(self.recurrent_layer,batch_size,self.hidden_size).to(self.device)
        c_0 = torch.rand(self.recurrent_layer,batch_size,self.hidden_size).to(self.device)

        return (h_0,c_0)