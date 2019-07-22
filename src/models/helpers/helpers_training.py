import torch
import time
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
from joblib import load
import matplotlib.cm as cm
import json
from datasets.datasets import Hdf5Dataset,CustomDataLoader
from matplotlib.lines import Line2D
import random 


def load_data_loaders(parameters_project,prepare_param,training_param,net_params,data_file,scenes):
    train_eval_scenes,train_scenes,test_scenes,eval_scenes = scenes

       
    if training_param["set_type_train"] == "train_eval":
        train_scenes = train_eval_scenes
    if training_param["set_type_test"] == "eval":
        test_scenes = eval_scenes
    


    train_dataset = Hdf5Dataset(
        hdf5_file= data_file,
        scene_list= train_scenes,
        t_obs=prepare_param["t_obs"],
        t_pred=prepare_param["t_pred"],
        set_type = training_param["set_type_train"], 
        data_type = "trajectories",
        use_neighbors = net_params["use_neighbors"],
        use_masks = 1,
        predict_offsets = net_params["offsets"],
        offsets_input = net_params["offsets_input"],
        padding = prepare_param["padding"],
        use_images = net_params["use_images"],
        images_path = parameters_project["raw_images"]
        )

    eval_dataset = Hdf5Dataset(
        hdf5_file= data_file,
        scene_list= test_scenes, #eval_scenes
        t_obs=prepare_param["t_obs"],
        t_pred=prepare_param["t_pred"],
        set_type = training_param["set_type_test"], #eval
        
        data_type = "trajectories",
        use_neighbors = net_params["use_neighbors"],
        use_masks = 1,
        predict_offsets = net_params["offsets"],
        offsets_input = net_params["offsets_input"],
        padding = prepare_param["padding"],
        use_images = net_params["use_images"],
        images_path = parameters_project["raw_images"]

        )

    train_loader = CustomDataLoader( batch_size = training_param["batch_size"],shuffle = True,drop_last = True,dataset = train_dataset,test=training_param["test"])
    eval_loader = CustomDataLoader( batch_size = training_param["batch_size"],shuffle = False,drop_last = True,dataset = eval_dataset,test=training_param["test"])
    
    
    return train_loader,eval_loader,train_dataset,eval_dataset



class MaskedLoss(nn.Module):
    def __init__(self,criterion):
        super(MaskedLoss, self).__init__()
        self.criterion = criterion

    def forward(self, outputs, targets, mask = None, first_only = 1):

        if mask is None:
            mask = torch.ones_like(targets)
        
        if first_only:
            mask[:,1:,:,:] = 0
        
        loss =  self.criterion(outputs*mask, targets*mask)
        
        a = (mask.sum(-1).sum(-1)>0).cuda().float()
        # loss = torch.sqrt(loss.sum(dim = -1))
        loss = loss.sum(dim = -1)
        loss = loss.sum(dim = -1)
        loss = loss.sum(dim = -1)/(a.sum(-1))
        loss = loss.mean(dim = -1)
        return loss    


def ade_loss(outputs,targets,mask = None,first_only = 1):

    if mask is None:
        mask = torch.ones_like(targets)

    if first_only:
        mask[:,1:,:,:] = 0

    # if mask is not None:
    outputs,targets = outputs*mask, targets*mask

    
    # outputs = outputs.contiguous().view(-1,2)
    # targets = targets.contiguous().view(-1,2)
    mse = nn.MSELoss(reduction= "none")
    

    mse_loss = mse(outputs,targets )
    mse_loss = torch.sum(mse_loss,dim = 3 )
    mse_loss = torch.sqrt(mse_loss )
    # if mask is not None:
    mse_loss = mse_loss.sum()/(mask.sum()/2.0)
    # else:
    #     mse_loss = torch.mean(mse_loss )

    return mse_loss

def fde_loss(outputs,targets,mask,first_only = 0):

    if mask is None:
        mask = torch.ones_like(targets)

    if first_only:
        mask[:,1:,:,:] = 0

    # if mask is not None:
    outputs,targets = outputs*mask, targets*mask

    b,n,s,i = outputs.size()

    outputs = outputs.view(b*n,s,i)
    targets = targets.view(b*n,s,i)
    mask = mask.view(b*n,s,i)
    ids = (mask.sum(dim = -1) > 0).sum(dim = -1)

    points_o = []
    points_t = []
    mask_n = []

    for seq_o,seq_t,m,id in zip(outputs,targets,mask,ids):
        if id == 0 or id == s:
            points_o.append(seq_o[-1])
            points_t.append(seq_t[-1])
            mask_n.append(m[-1])



        else:
            points_o.append(seq_o[id-1])
            points_t.append(seq_t[id-1])
            mask_n.append(m[id-1])

    points_o = torch.stack([po for po in points_o],dim = 0)
    points_t = torch.stack([pt for pt in points_t], dim = 0)
    mask_n = torch.stack([m for m in mask_n], dim = 0)




    mse = nn.MSELoss(reduction= "none")

    mse_loss = mse(points_o,points_t )
    mse_loss = torch.sum(mse_loss,dim = 1 )
    mse_loss = torch.sqrt(mse_loss )

    # if mask is not None:
    mask = mask[:,-1]
    mse_loss = mse_loss.sum()/(mask.sum()/2.0)
    # else:
    #     mse_loss = torch.mean(mse_loss )

    return mse_loss




def plot_grad_flow(named_parameters,epoch,root = "./data/reports/gradients/"):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.
    
    Usage: Plug this function in Trainer class after loss.backwards() as 
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads= []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    
    fig, ax = plt.subplots()

    
    ax.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    ax.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    ax.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
    ax.set_xticks(range(0, len(ave_grads), 1))
    ax.set_xticklabels(layers, rotation='vertical', fontsize='small')
    ax.set_yscale('log')
    ax.set_xlabel("Layers")
    ax.set_ylabel("Gradient magnitude")
    ax.set_title('Gradient flow')
    ax.grid(True)
    lgd = ax.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])

    # plt.savefig("{}gradients_{}.jpg".format(root,epoch), bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.savefig("{}gradients_{}.jpg".format(root,time.time()), bbox_extra_artists=(lgd,), bbox_inches='tight')

    plt.close()


    

def offsets_to_trajectories(inputs,labels,outputs,offsets,offsets_input,last_points,input_last):

    if offsets_input == 1:
        inputs = input_last
    
    if offsets == 1:
        labels = np.add(last_points,labels)
        outputs = np.add(last_points,outputs)
        return inputs,labels,outputs
    elif offsets == 2:
        print("offset 2 not allowed")
    else :
        return inputs,labels,outputs


