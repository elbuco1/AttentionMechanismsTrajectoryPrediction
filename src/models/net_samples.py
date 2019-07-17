from models.rnn_mlp import RNN_MLP
from models.social_attention import SocialAttention
from models.cnn_mlp import CNN_MLP


import time

import json
import torch
import sys
import helpers.helpers_training as helpers
import helpers.helpers_evaluation as helpers_evaluation
import torch.nn as nn
import numpy as np
import os 

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")    
    torch.manual_seed(42)
   
    #loading parameters
    parameters_path = "./src/parameters/project.json"
    parameters_project = json.load(open(parameters_path))  
    data_processed_parameters = json.load(open(parameters_project["data_processed_parameters"]))
    evaluation_parameters = json.load(open(parameters_project["evaluation_parameters"])) 
    processed_parameters = json.load(open(parameters_project["data_processed_parameters"]))


    # loading training data
    data_file = parameters_project["hdf5_samples"]

    # scene lists for train, eval and test
    eval_scenes = data_processed_parameters["eval_scenes"]
    train_eval_scenes = data_processed_parameters["train_scenes"]
    test_scenes = data_processed_parameters["test_scenes"]
    train_scenes = [scene for scene in train_eval_scenes if scene not in eval_scenes]   
    scenes = [train_eval_scenes,train_scenes,test_scenes,eval_scenes]

    report_name = evaluation_parameters["report_name"]
    model_name = evaluation_parameters["model_name"]
    models_path = parameters_project["models_evaluation"] + "{}.tar".format(model_name)

    print("loading trained model {}".format(model_name))
    checkpoint = torch.load(models_path)
    args_net = checkpoint["args"]
    model = args_net["model_name"]  

    net = None
    if model == "rnn_mlp":
        net = RNN_MLP(args_net)
    elif model == "cnn_mlp":        
        net = CNN_MLP(args_net)     
    elif model == "social_attention":
        net = SocialAttention(args_net)

    # loading trained network
    net.load_state_dict(checkpoint['state_dict'])
    net = net.to(device)
    net.eval()


    scenes = test_scenes
    set_type_test = evaluation_parameters["set_type_test"]

    if set_type_test == "train":
        scenes = train_scenes
    elif set_type_test == "eval":
        scenes = eval_scenes
    elif set_type_test == "train_eval":
        scenes = train_eval_scenes

    times = 0 # sum time for every prediction
    nb_samples = 0 # number of predictions


    dir_name = parameters_project["evaluation_reports"] + "{}/".format(report_name)
    sub_dir_name = parameters_project["evaluation_reports"] + "{}/scene_reports/".format(report_name) 

    
        
    if os.path.exists(dir_name):
        os.system("rm -r {}".format(dir_name))
    os.system("mkdir {}".format(dir_name))
    if os.path.exists(sub_dir_name):
        os.system("rm -r {}".format(sub_dir_name))
    os.system("mkdir {}".format(sub_dir_name))

    s = time.time()
    for z,scene in enumerate(scenes):

        sample_id = 0
        
        print(scene)

        scene_dict = {} # save every sample in the scene
              
        # get dataloader
        data_loader = helpers_evaluation.get_data_loader(data_file,scene,args_net,processed_parameters,evaluation_parameters)
        
        sample_id = 0

        
        print(time.time()-s)
        
        for batch_idx, data in enumerate(data_loader):
                
            inputs, labels,types,points_mask, active_mask,target_last,input_last = data
            inputs = inputs.to(device)
            labels =  labels.to(device)

            # active mask for training, along batch*numbr_agent axis
            active_mask = active_mask.to(device)

            points_mask = list(points_mask)
            if not args_net["offsets_input"]:
                input_last = np.zeros_like(inputs.cpu().numpy()) 
            
            b,n,_,_ = inputs.shape

            start = time.time()
            if not args_net["use_neighbors"]:
                outputs,inputs,types,active_mask,points_mask = helpers_evaluation.predict_naive(inputs,types,active_mask,points_mask,net,device)


            else:
                outputs,inputs,types,active_mask,points_mask = helpers_evaluation.predict_neighbors_disjoint(inputs,types,active_mask,points_mask,net,device)

            end = time.time() - start 
            times += end
            nb_samples += b*n

            active_mask = helpers_evaluation.get_active_mask(points_mask[1])
            points_mask = torch.FloatTensor(points_mask[1]).to(device)                    
            outputs = torch.mul(points_mask,outputs)
            labels = torch.mul(points_mask,labels) # bon endroit?
            # active mask per sample in batch


            for i,l,o,t,p, a, il, tl  in zip(inputs, labels, outputs, types, points_mask,active_mask, input_last, target_last):

                i = i[a].detach().cpu().numpy()
                l = l[a].detach().cpu().numpy()
                t = t[a]
                p = p[a]
                o = o[a].detach().cpu().numpy()
                tl = tl[a]
                il = il[a]

                # revert offsets
                i,l,o = helpers.offsets_to_trajectories( i,l,o,args_net["offsets"],args_net["offsets_input"],tl,il)
                           

                # apply active mask
                scene_dict[sample_id] = {}
                scene_dict[sample_id]["inputs"] = i.tolist()
                scene_dict[sample_id]["labels"] = l.tolist()
                scene_dict[sample_id]["outputs"] = o.tolist()
                # scene_dict[sample_id]["active_mask"] = a.cpu().numpy().tolist()
                scene_dict[sample_id]["types"] = t.tolist()
                scene_dict[sample_id]["points_mask"] = p.cpu().numpy().tolist()

                sample_id += 1
        json.dump(scene_dict, open(sub_dir_name + "{}_samples.json".format(scene),"w"),indent= 0)
    timer = {
        "total_time":times,
        "nb_trajectories":nb_samples,
        "time_per_trajectory":times/nb_samples
    }
    # save the time
    json.dump(timer, open(dir_name + "time.json","w"),indent= 0)   
    

if __name__ == "__main__":
    main()