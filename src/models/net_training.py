import torch
import torch.nn as nn
# import torch.nn.functional as f
import torch.optim as optim

import numpy as np
import time
import os

from models.rnn_mlp import RNN_MLP
from models.social_attention import SocialAttention
from models.cnn_mlp import CNN_MLP
import random
from classes.training_class import NetTraining
import helpers.helpers_training as helpers


import sys
import json


def main():
          
    # set pytorch manual seed
    torch.manual_seed(10)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")    
    print("using gpu: ",torch.cuda.is_available())  


    #loading parameters
    parameters_path = "./src/parameters/project.json"
    parameters_project = json.load(open(parameters_path))  
    data_processed_parameters = json.load(open(parameters_project["data_processed_parameters"]))
    training_parameters = json.load(open(parameters_project["training_parameters"])) 
    models_parameters = json.load(open(parameters_project["models_parameters"])) 
    processed_parameters = json.load(open(parameters_project["data_processed_parameters"]))

    # loading training data
    data_file = parameters_project["training_hdf5"]

    # scene lists for train, eval and test
    eval_scenes = data_processed_parameters["eval_scenes"]
    train_eval_scenes = data_processed_parameters["train_scenes"]
    test_scenes = data_processed_parameters["test_scenes"]
    train_scenes = [scene for scene in train_eval_scenes if scene not in eval_scenes]   
    scenes = [train_eval_scenes,train_scenes,test_scenes,eval_scenes]

    
    # loading models hyperparameters
    model_name = training_parameters["model"]
    net_params = json.load(open(models_parameters[model_name]))

    net_type = None
    args_net = {}

    # set network arguments depending on its type
    if model_name == "rnn_mlp":
        args_net = {
        "device" : device,
        "batch_size" : training_parameters["batch_size"],
        "input_dim" : net_params["input_dim"],
        "hidden_size" : net_params["hidden_size"],
        "recurrent_layer" : net_params["recurrent_layer"],
        "mlp_layers" : net_params["mlp_layers"],
        "output_size" : net_params["output_size"],
        "t_pred":processed_parameters["t_pred"],
        "t_obs":processed_parameters["t_obs"],
        "use_images":net_params["use_images"],
        "use_neighbors":net_params["use_neighbors"],
        "offsets":training_parameters["offsets"],
        "offsets_input" : training_parameters["offsets_input"],
        "model_name":model_name
        }
        net_type = RNN_MLP
    
    elif model_name == "cnn_mlp":
        args_net = {
        "device" : device,
        "batch_size" : training_parameters["batch_size"],
        "input_length" : processed_parameters["t_obs"],
        "output_length" : processed_parameters["t_pred"],
        "num_inputs" : net_params["input_dim"],
        "mlp_layers" : net_params["mlp_layers"],
        "output_size" : net_params["output_size"],
        "input_dim" : net_params["input_dim"],
        "kernel_size": net_params["kernel_size"],
        "nb_conv": net_params["nb_conv"],
        "nb_kernel": net_params["nb_kernel"],
        "cnn_feat_size": net_params["cnn_feat_size"],
        "use_images":net_params["use_images"],
        "use_neighbors":net_params["use_neighbors"],
        "offsets":training_parameters["offsets"],
        "offsets_input" : training_parameters["offsets_input"],
        "model_name":model_name

        }
        net_type = CNN_MLP
    
    elif model_name == "social_attention":
        args_net = {
            "device" : device,
            "input_dim" : net_params["input_dim"],
            "input_length" : processed_parameters["t_obs"],
            "output_length" : processed_parameters["t_pred"],
            
            "dmodel" : net_params["dmodel"],
            "predictor_layers" : net_params["predictor_layers"],
            "pred_dim" : processed_parameters["t_pred"] * net_params["input_dim"] ,

            "nb_conv": net_params["nb_conv"],
            "nb_kernel": net_params["nb_kernel"],
            "cnn_feat_size": net_params["cnn_feat_size"],
            "kernel_size" : net_params["kernel_size"],
            
            "dropout_tfr" : net_params["dropout_tfr"],
            "projection_layers":net_params["projection_layers"],
            
            "use_images":net_params["use_images"],
            "use_neighbors":net_params["use_neighbors"],
            "offsets":training_parameters["offsets"],
            "offsets_input" : training_parameters["offsets_input"],
            "model_name":model_name

        }     

        net_type = SocialAttention
    
        
    # init neural network
    net = net_type(args_net)

    train_loader,eval_loader,_,_ = helpers.load_data_loaders(processed_parameters,training_parameters,args_net,data_file,scenes)
    
    net = net.to(device)
    optimizer = optim.Adam(net.parameters(),lr = training_parameters["lr"])
    criterion = helpers.MaskedLoss(nn.MSELoss(reduction="none"))

    args_training = {
        "n_epochs" : training_parameters["n_epochs"],
        "batch_size" : training_parameters["batch_size"],
        "device" : device,
        "train_loader" : train_loader,
        "eval_loader" : eval_loader,
        "criterion" : criterion,
        "optimizer" : optimizer,
        "use_neighbors" : net_params["use_neighbors"],
        "plot" : training_parameters["plot"],
        "load_path" : training_parameters["load_path"],
        "plot_every" : training_parameters["plot_every"],
        "save_every" : training_parameters["save_every"],
        "offsets" : training_parameters["offsets"],
        "offsets_input" : training_parameters["offsets_input"],
        "net" : net,
        "print_every" : training_parameters["print_every"],
        "nb_grad_plots" : training_parameters["nb_grad_plots"],
        "train" : training_parameters["train"],
        "gradients_reports": parameters_project["gradients_reports"],
        "losses_reports": parameters_project["losses_reports"],
        "models_reports": parameters_project["models_reports"]

         
    }

    # starting training
    trainer = NetTraining(args_training)
    trainer.training_loop()
    


if __name__ == "__main__":
    main()


