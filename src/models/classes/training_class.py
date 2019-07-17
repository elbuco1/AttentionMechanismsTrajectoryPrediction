import torch
import time
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
from joblib import load
import helpers.helpers_training as helpers
import json
import os
from scipy import stats
# from tensorboardX import SummaryWriter


class NetTraining():
    def __init__(self,args):
        self.args = args 
        self.n_epochs = args["n_epochs"]
        self.batch_size = args["batch_size"]
        self.device = args["device"]
        self.train_loader = args["train_loader"]
        self.eval_loader = args["eval_loader"]
        self.criterion = args["criterion"]
        self.optimizer = args["optimizer"]
        self.use_neighbors = args["use_neighbors"]
        self.plot = args["plot"]
        self.load_path = args["load_path"]
        self.plot_every = args["plot_every"]
        self.save_every = args["save_every"]
        self.offsets = args["offsets"]
        self.offsets_input = args["offsets_input"]


        self.net = args["net"]
        self.print_every = args["print_every"]
        self.nb_grad_plots = args["nb_grad_plots"]
        self.nb_sample_plots = args["nb_sample_plots"]
        self.train_model = args["train"]
        self.gradient_reports = args["gradients_reports"]
        self.losses_reports = args["losses_reports"]
        self.models_reports = args["models_reports"]



    def training_loop(self):
        losses = {
        "train":{ "loss": []},
        "eval":{
            "loss": [],
            "fde":[],
            "ade":[]}
        }

        start_epoch = 0
        s = time.time()

        # restart model training if a path is given
        if self.load_path != "":
            print("loading former model from {}".format(self.load_path))
            checkpoint = torch.load(self.load_path)
            self.net.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            losses = checkpoint["losses"]
            start_epoch = checkpoint["epoch"]

        try:
            best_harmonic_fde_ade = float('inf')
            best_ade = 0
            best_fde = 0
            
            # iterations
            for epoch in range(start_epoch,self.n_epochs):
                train_loss = 0.
                # train model
                if self.train_model:
                    train_loss,_ = self.train(epoch)
                # eval model
                eval_loss,fde,ade = self.evaluate(epoch)                   
                
                # store losses
                losses["train"]["loss"].append(train_loss)
                losses["eval"]["loss"].append(eval_loss)
                losses["eval"]["ade"].append(ade)
                losses["eval"]["fde"].append(fde)

                # display losses graphs
                if self.plot and epoch % self.plot_every == 0:
                    self.plot_losses(losses,s,root = self.losses_reports)

                # save model regularly
                if epoch % self.save_every == 0:
                    self.save_model(epoch,epoch,self.net,self.optimizer,losses,save_root= self.models_reports)

                h = stats.hmean([ade,fde])

                # if new best model save it
                if h < best_harmonic_fde_ade:
                    print("harmonic mean {} is better than {}, saving new best model!".format(h,best_harmonic_fde_ade))
                    self.save_model(epoch,"best",self.net,self.optimizer,losses,remove=0 ,save_root= self.models_reports)
                    best_harmonic_fde_ade = h
                    best_ade = ade
                    best_fde = fde
        
        except Exception as e: 
            print(e)

   
        return best_harmonic_fde_ade,best_ade,best_fde



    """
        Train loop for an epoch
        Uses cuda if available
        LOss is averaged for a batch
        THen averaged batch losses are averaged
        over the number of batches
    """
    def train(self,epoch):
        self.net.train()
        epoch_loss = 0.
        batches_loss = []
        

        # select randomly batches for gradient disaying
        ids_grads = np.arange(int(self.train_loader.nb_batches) )
        np.random.shuffle(ids_grads)
        ids_grads = ids_grads[:self.nb_grad_plots]

        # iterate over batches
        for batch_idx, data in enumerate(self.train_loader):

            start_time = time.time()
            
            # Load data
            inputs, labels,types,points_mask, active_mask,_,_ = data
            inputs = inputs.to(self.device)
            labels =  labels.to(self.device)
            types =  types.to(self.device)       
            active_mask = active_mask.to(self.device)

            
            # gradients to zero
            self.optimizer.zero_grad()
            # predict using network
            outputs = self.net((inputs,types,active_mask,points_mask))
    
            # keep mask for prediction part only
            points_mask = points_mask[1]
            points_mask = torch.FloatTensor(points_mask).to(self.device).detach()

            
            # compute loss and backprop
            loss = self.criterion(outputs.clone(), labels.clone(),points_mask.clone())
            loss.backward()

            # if this batch was selected, plot gradients
            if batch_idx in ids_grads:
                try:
                    helpers.plot_grad_flow(self.net.named_parameters(),epoch,self.gradient_reports)
                except Exception as e: 
                    print(e)

            self.optimizer.step()

            epoch_loss += loss.item()
            batches_loss.append(loss.item())

            # print batch loss <-- mean batch loss for last print_every timesteps
            if batch_idx % self.print_every == 0:
                print(batch_idx,loss.item(),time.time()-start_time)     
        
        epoch_loss = np.mean(batches_loss)  

        print('Epoch n {} Loss: {}'.format(epoch,epoch_loss))

        return epoch_loss,batches_loss



    """
        Evaluation loop for an epoch
        Uses cuda if available
        LOss is averaged for a batch
        THen averaged batch losses are averaged
        over the number of batches

        FDE loss is added using MSEerror on the last point of prediction and target
        sequences

        model: 0 rnn_mlp
            1 iatcnn
    """
    def evaluate(self,epoch):
        self.net.eval()
        eval_loss = 0.
        fde = 0.
        ade = 0.
        eval_loader_len =   float(self.eval_loader.nb_batches)

        batch_losses = []


        for i,data in enumerate(self.eval_loader):

            # Load data
            inputs, labels,types,points_mask, active_mask,target_last,input_last = data
            inputs = inputs.to(self.device)
            labels =  labels.to(self.device)
            types =  types.to(self.device)
            active_mask = active_mask.to(self.device)
            
            outputs = self.net((inputs.clone(),types,active_mask,points_mask))
            
         
            
            # if relative positions used as inputs or outputs, set them back to absolute
            inputs,labels,outputs = helpers.offsets_to_trajectories(inputs.detach().cpu().numpy(),
                                                                labels.detach().cpu().numpy(),
                                                                outputs.detach().cpu().numpy(),
                                                                self.offsets,self.offsets_input,target_last,input_last)
        
            inputs,labels,outputs = torch.FloatTensor(inputs).to(self.device),torch.FloatTensor(labels).to(self.device),torch.FloatTensor(outputs).to(self.device)
            

            # we don't count the prediction error for end of trajectory padding
            points_mask = points_mask[1]
            points_mask = torch.FloatTensor(points_mask).to(self.device)#
            loss = self.criterion(outputs.clone(), labels.clone(),points_mask.clone())
            batch_losses.append(loss.item())

         
            ade += helpers.ade_loss(outputs.clone(),labels.clone(),points_mask.clone()).item() ######
            fde += helpers.fde_loss(outputs.clone(),labels.clone(),points_mask.clone()).item()
        
            eval_loss += loss.item()
        eval_loss = np.mean(batch_losses)

        ade /= eval_loader_len      
        fde /= eval_loader_len        

        print('Epoch n {} Evaluation Loss: {}, ADE: {}, FDE: {}'.format(epoch,eval_loss,ade,fde))
        return eval_loss,fde,ade


    def evaluate_analysis(self,eval_loader,verbose = 1):
        self.net.eval()
        eval_loss = 0.
        fde = 0.
        ade = 0.
        eval_loader_len =   float(eval_loader.nb_batches)

        batch_losses = []

        
        for i,data in enumerate(eval_loader):

            # Load data
            inputs, labels,types,points_mask, active_mask,target_last,input_last = data
            inputs = inputs.to(self.device)
            labels =  labels.to(self.device)
            types =  types.to(self.device)
            imgs =  imgs.to(self.device)        
            active_mask = active_mask.to(self.device)
            
            outputs = self.net((inputs,types,active_mask,points_mask))
            
            
            inputs,labels,outputs = helpers.offsets_to_trajectories(inputs.detach().cpu().numpy(),
                                                                    labels.detach().cpu().numpy(),
                                                                    outputs.detach().cpu().numpy(),
                                                                    self.offsets)
           
            inputs,labels,outputs = torch.FloatTensor(inputs).to(self.device),torch.FloatTensor(labels).to(self.device),torch.FloatTensor(outputs).to(self.device)
            

            # we don't count the prediction error for end of trajectory padding
            points_mask = points_mask[1]
            points_mask = torch.FloatTensor(points_mask).to(self.device)#


            loss = self.criterion(outputs.clone(), labels.clone(),points_mask.clone())
            batch_losses.append(loss.item())

            

            ade += helpers.ade_loss(outputs.clone(),labels.clone(),points_mask.clone()).item()
            fde += helpers.fde_loss(outputs.clone(),labels.clone(),points_mask.clone()).item()
        
            eval_loss += loss.item()
        eval_loss = np.mean(batch_losses)
        ade /= eval_loader_len      
        fde /= eval_loader_len        
        if verbose:
            print('Evaluation Loss: {}, ADE: {}, FDE: {}'.format(eval_loss,ade,fde))
        return eval_loss,fde,ade

    def plot_losses(self,losses,idx,root):
        plt.plot(losses["train"]["loss"],label = "train_loss")
        plt.plot(losses["eval"]["loss"],label = "eval_loss")
        plt.legend()

        # plt.show()
        plt.savefig("{}losses_{}.jpg".format(root,idx))
        plt.close()

        plt.plot(losses["eval"]["ade"],label = "ade")
        plt.plot(losses["eval"]["fde"],label = "fde")
        plt.legend()

        plt.savefig("{}ade_fde_{}.jpg".format(root,idx))
        plt.close()




    """
        Saves model and optimizer states as dict
        THe current epoch is stored
        THe different losses at previous time_steps are loaded

    """
    def save_model(self,epoch,name,net,optimizer,losses,remove = 1,save_root = "./learning/data/models/" ):

        dirs = os.listdir(save_root)

        # save_path = save_root + "model_{}_{}.tar".format(name,time.time())
        save_path = save_root + "model_{}.tar".format(name)


        # print("args {}".format(net.args))
        state = {
            'epoch': epoch,
            'state_dict': net.state_dict(),
            # 'named_parameters': net.named_parameters(),

            'optimizer': optimizer.state_dict(),             
            'losses': losses,
            'args': net.args
            }
        # state = {
        #     'state_dict': net.state_dict(),
        #     }
        torch.save(state, save_path)

        if remove:
            for dir_ in dirs:
                if dir_ != "model_best.tar":
                    os.remove(save_root+dir_)
        
        print("model saved in {}".format(save_path))







