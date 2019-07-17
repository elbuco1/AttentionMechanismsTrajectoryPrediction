import torch
import torch.nn as nn
import torch.nn.functional as f 


class CNN(nn.Module):
    def __init__(self, num_inputs, nb_kernel,cnn_feat_size,obs_len = 8, kernel_size=3,nb_conv = 4):
        super(CNN, self).__init__()
        self.kernel_size = kernel_size
        self.nb_conv = nb_conv
        self.nb_kernel = nb_kernel
        self.num_inputs = num_inputs
        self.cnn_feat_size = cnn_feat_size
        self.obs_len = obs_len

        print(self.nb_conv)
        self.cnn = nn.Sequential()
        
        # if self.kernel_size % 2 == 1:
        padding = int((self.kernel_size-1)/2.0)
        # else:
        #     pg = int((self.kernel_size)/2.0)
        #     padding = (pg,pg-1)
        # padding = 0
        # self.padding_layer = nn.ConstantPad1d(padding, 0)
        for i in range(self.nb_conv):
            conv = nn.Conv1d(self.nb_kernel , self.nb_kernel , self.kernel_size, padding=padding)
            # conv = nn.Conv1d(self.nb_kernel , self.nb_kernel , self.kernel_size)


            if i == 0:
                # conv = nn.Conv1d(self.num_inputs, self.nb_kernel , self.kernel_size)

                conv = nn.Conv1d(self.num_inputs, self.nb_kernel , self.kernel_size, padding=padding)
            self.cnn.add_module("conv{}".format(i),conv)
            # self.cnn.add_module("relu{}".format(i),nn.ReLU())

        
        self.project_cnn = nn.Linear(self.obs_len*self.nb_kernel,self.cnn_feat_size)
    def forward(self, x):
        # x = self.padding_layer(x)
        x = self.cnn(x)# x: B,n_kernels,Tobs
        x = x.permute(0,2,1).contiguous() # x: B,Tobs,n_kernels
        batch_size = x.size()[0]
        x = x.view(batch_size,-1)# x: B,Tobs*n_kernels
        x = f.relu(x) # ?

        x = self.project_cnn(x) # x: B,cnn_feat_size

        output = f.relu(x)
        
        return output