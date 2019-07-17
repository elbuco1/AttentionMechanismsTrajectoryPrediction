from sklearn.preprocessing import OneHotEncoder
from scipy.spatial import distance_matrix,distance
from scipy.stats import norm

from scipy.spatial.distance import euclidean
from scipy.stats import wasserstein_distance


import matplotlib.image as mpimg
import cv2
import copy

import time

import json
import torch
import sys
import helpers.helpers_training as helpers
import torch.nn as nn
import numpy as np
import os 

from datasets.datasets import Hdf5Dataset,CustomDataLoader
# from classes.evaluation import Evaluation

def ade(outputs,targets,mask = None):

    if mask is None:
        # mask = torch.ones_like(targets)
        mask = np.ones_like(targets)
    # if mask is not None:
    outputs,targets = outputs*mask, targets*mask

    ade = np.subtract(outputs,targets) ** 2
    ade = np.sqrt(ade.sum(-1))
    ade = ade.sum()/(mask.sum()/2)
    return ade

def fde(outputs,targets,mask):

    if len(targets.shape) < 3:
        outputs = np.expand_dims(outputs,0)
        targets = np.expand_dims(targets,0)
        mask = np.expand_dims(mask,0)


    if mask is None:
        mask = np.ones_like(targets)     
    # if mask is not None:
    outputs,targets = outputs*mask, targets*mask

    # n,s,i = outputs.shape
    ids = (mask.sum(-1) > 0).sum(-1)

    points_o = []
    points_t = []
    mask_n = []

    for seq_o,seq_t,m,id in zip(outputs,targets,mask,ids):
        if id == 0 or id == len(seq_t):
            points_o.append(seq_o[-1])
            points_t.append(seq_t[-1])
            mask_n.append(m[-1])
        else:
            points_o.append(seq_o[id-1])
            points_t.append(seq_t[id-1])
            mask_n.append(m[id-1])

    points_o = np.array(points_o)
    points_t = np.array(points_t)
    mask_n = np.array(mask_n)


    fde = np.subtract(points_o,points_t) ** 2
    fde = np.sqrt(fde.sum(-1))
    fde = fde.sum()/(mask_n.sum()/2.0)

    return fde

# i = i.detach().cpu().numpy()
def revert_scaling_evaluation(offsets_input,scalers_path,i):
    scaler = json.load(open(scalers_path))
    if offsets_input:
        meanx =  scaler["standardization"]["meanx"]
        meany =  scaler["standardization"]["meany"]
        stdx =  scaler["standardization"]["stdx"]
        stdy =  scaler["standardization"]["stdy"]
        

        i[:,:,0] = helpers.revert_standardization(i[:,:,0],meanx,stdx)
        i[:,:,1] = helpers.revert_standardization(i[:,:,1],meany,stdy)
    else:
        min_ =  scaler["normalization"]["min"]
        max_ =  scaler["normalization"]["max"]
        i = helpers.revert_min_max_scale(i,min_,max_)
    return i        

    # i = torch.FloatTensor(i).to(device)        



def types_ohe(types,nb_types):       
    cat = np.arange(1,nb_types+1).reshape(nb_types,1)
    
    ohe = OneHotEncoder(sparse = False,categories = "auto")
    ohe = ohe.fit(cat)

    b,n = types.shape
    # types = types - 1 
    types = ohe.transform(types.reshape(b*n,-1)) 

    types = types.reshape(b,n,nb_types)

    return types

def get_active_mask(mask_target):
    sample_sum = (np.sum(mask_target.reshape(list(mask_target.shape[:2])+[-1]), axis = 2) > 0).astype(int)
                
    active_mask = []
    for b in sample_sum:
        ids = np.argwhere(b.flatten()).flatten()
        active_mask.append(torch.LongTensor(ids))
    return active_mask

def get_data_loader(data_file,scene,args_net,prepare_param,eval_params):
    
    dataset = Hdf5Dataset(
        hdf5_file= data_file,
        scene_list= [scene],
        t_obs=prepare_param["t_obs"],
        t_pred=prepare_param["t_pred"],
        set_type = eval_params["set_type_test"], 
        data_type = "trajectories",
        use_neighbors = 1,
        use_masks = 1,
        predict_offsets = args_net["offsets"],
        offsets_input = args_net["offsets_input"],
        padding = prepare_param["padding"],
        evaluation = 1
        )

    data_loader = CustomDataLoader( batch_size = eval_params["batch_size"],shuffle = False,drop_last = False,dataset = dataset,test=0)
    return data_loader

#
#    COnsiders each pair of agent as an interaction
#    Counts the number of problematic interaction
#    averages the percentage over all timesteps
#    returns conflicts coordinates without specifying their timestep
def conflicts(output,threshold = 0.5):
    timesteps = []
    timesteps_disjoint = []

    conflict_points = np.array([])
    for t in range(output.shape[1]):
        points = np.array(output[:,t])
        d = distance_matrix(points,points)

        m = (d < threshold).astype(int) - np.eye(len(points))
        total_count = np.ones_like(m)
        m = np.triu(m,1)
        total_count = np.triu(total_count,1)

        # disjoint (only first trajectory)
        if float(total_count[0].sum()) > 0.:
            conflict_prop_disjoint = m[0].sum() / float(total_count[0].sum())
        else:
            conflict_prop_disjoint = 0
        # joint
        if float(total_count.sum()) > 0.:           
            conflict_prop = m.sum() / float(total_count.sum())
        else:
            conflict_prop = 0

        timesteps.append(conflict_prop)
        timesteps_disjoint.append(conflict_prop_disjoint)


        # select points where conflict happens
        ids = np.unique( np.argwhere(m)[:,0] ) 
        if len(ids) > 0:
            points = points[ids]
            if len(conflict_points) > 0:
                conflict_points = np.concatenate([conflict_points,points], axis = 0)
            else:
                conflict_points = points



    return np.mean(timesteps),np.mean(timesteps_disjoint),timesteps,timesteps_disjoint,conflict_points.tolist()

def dynamic_eval(output,types,dynamics,types_dic,delta_t,dynamic_threshold = 0.0):
    acc_lhood = []
    speed_lhood = []

    for a in range(output.shape[0]):
        coordinates = output[a]
        type_ = types[a]
        type_ = types_dic[str(int(type_+1))]

        speeds = np.array(get_speeds(coordinates,delta_t))
        accelerations = np.array(get_accelerations(speeds,delta_t))
        dynamic_type = dynamics[type_]

        acc_props = norm.pdf(accelerations, loc = dynamic_type["accelerations"]["mean"], scale=dynamic_type["accelerations"]["std"]) 
        speed_props = norm.pdf(accelerations, loc = dynamic_type["speeds"]["mean"], scale=dynamic_type["speeds"]["std"]) 

        acc_lhood.append( acc_props)
        speed_lhood.append(speed_props)

    acc_lhood_disjoint = acc_lhood[0]
    speed_lhood_disjoint = speed_lhood[0]

    return acc_lhood,speed_lhood,acc_lhood_disjoint,speed_lhood_disjoint,np.mean(acc_lhood),np.mean(speed_lhood),np.mean(acc_lhood_disjoint),np.mean(speed_lhood_disjoint)


def get_speed(point1,point2,deltat):
    d = distance.euclidean(point1,point2)
    v = d/deltat
    return v
def get_speeds(coordinates,framerate):
    speeds = []
    for i in range(1,len(coordinates)):
        speed = get_speed(coordinates[i-1],coordinates[i],framerate)
        speeds.append(speed)
    return speeds

def get_acceleration(v1,v2,deltat):
    a = (v2-v1)/deltat
    return a

def get_accelerations(speeds,framerate):
    accelerations = []
    for i in range(1,len(speeds)):
        acceleration = get_acceleration(speeds[i-1],speeds[i],framerate)
        accelerations.append(acceleration)
    return accelerations


# def scene_mask(scene,img_path,class_category,annotations_path):
#         img = mpimg.imread(img_path.format(scene))
#         empty_mask = np.zeros_like(img[:,:,0]).astype(np.int32)
#         annotations = json.load(open(annotations_path.format(scene)))
#         polygons = []
#         for object_ in annotations["objects"]:
#                 if object_["classTitle"] == class_category:
#                         pts = object_["points"]["exterior"]
#                         a3 = np.array( [pts] ).astype(np.int32)          
#                         cv2.fillPoly( empty_mask, a3, 1 )
#         return empty_mask

# returns array of two masks, first one is pedestrian, second one is wheels
def scene_mask(scene,img_path,annotations_path,spatial_profiles):
        img = mpimg.imread(img_path.format(scene))

        masks = []
        masks_ids = []

        for spatial_profile in spatial_profiles:
        
            empty_mask = np.zeros_like(img[:,:,0]).astype(np.int32)
            annotations = json.load(open(annotations_path.format(scene)))
            polygons = []
            for object_ in annotations["objects"]:
                if object_["classTitle"] == spatial_profile:
                    pts = object_["points"]["exterior"]
                    a3 = np.array( [pts] ).astype(np.int32)          
                    cv2.fillPoly( empty_mask, a3, 1 )
            masks.append(empty_mask)
            masks_ids.append(spatial_profiles[spatial_profile])

        arg_ids = np.argsort(masks_ids)
        masks = [masks[i] for i in arg_ids]
        
        return masks

def spatial_conflicts(mask,trajectory_p):
        ctr = 0
        # print(mask.shape)
        for point in trajectory_p:
                #case out of frame
                if point[1] in range(0,mask.shape[0]) and point[0] in range(0,mask.shape[1]):
                    if mask[point[1],point[0]]:
                            ctr += 1
        return ctr / float(len(trajectory_p))

# def spatial_conflicts(mask,trajectory_p):
#         ctr = 0
#         # print(mask.shape)
#         frame_conflicts = np.zeros(len(trajectory_p))
#         for i,point in enumerate(trajectory_p):
#                 #case out of frame
#                 if point[1] in range(0,mask.shape[0]) and point[0] in range(0,mask.shape[1]):
#                     if mask[point[1],point[0]]:
#                         frame_conflicts[i] = 0
                    
#         return frame_conflicts

def spatial_loss(spatial_profile_ids,spatial_masks,outputs,pixel2meters):
    spatial_losses = []
    for id_,trajectory_p in zip(spatial_profile_ids,outputs):
        trajectory_p *= pixel2meters
        trajectory_p = trajectory_p.astype(np.int32)
        res = spatial_conflicts(spatial_masks[id_],trajectory_p)
        spatial_losses.append(res)
    return spatial_losses[0], np.mean(spatial_losses)

# def spatial_loss(spatial_profile_ids,spatial_masks,outputs,pixel2meters):
#     spatial_losses = []
#     frames = []
#     for id_,trajectory_p in zip(spatial_profile_ids,outputs):
#         trajectory_p *= pixel2meters
#         trajectory_p = trajectory_p.astype(np.int32)
#         frame_conflicts = spatial_conflicts(spatial_masks[id_],trajectory_p)
#         frames.append(frame_conflicts)
#     frames = np.array(frames)

#     # joint loss
#     nb_conflicts_per_frame = []
#     for t in range(frames.shape[1]):
#         frame = frames[:,t]
#         nb_conflicts_per_frame.append(np.sum(frame))
#     # disjoint loss
#     nb_conflicts_per_frame_disjoint = []
#     for t in range(frames.shape[1]):
#         frame = frames[0,t]
#         nb_conflicts_per_frame_disjoint.append(np.sum(frame))

#     return np.mean(nb_conflicts_per_frame_disjoint), np.mean(nb_conflicts_per_frame) ,nb_conflicts_per_frame_disjoint, nb_conflicts_per_frame

# for t in range(output.shape[1]):
#     points = np.array(output[:,t])


def get_factor(scene,correspondences_trajnet,correspondences_manual):
    if scene in correspondences_trajnet:
        row = correspondences_trajnet[scene]
        pixel2meter_ratio = row["pixel2meter"]
        meter2pixel_ratio = 1/pixel2meter_ratio
    else:
        row = correspondences_manual[scene]
        meter_dist = row["meter_distance"]
        pixel_coord = row["pixel_coordinates"]
        pixel_dist = euclidean(pixel_coord[0],pixel_coord[1])
        pixel2meter_ratio = meter_dist/float(pixel_dist)
        meter2pixel_ratio = float(pixel_dist)/meter_dist

    return meter2pixel_ratio


def predict_neighbors_disjoint(inputs,types,active_mask,points_mask,net,device):
    b,n,s,i = points_mask[0].shape
    b,n,p,i = points_mask[1].shape

    # permute every samples
    batch_perms = []
    batch_p0 = []
    batch_p1 = []
    for batch_element,p0,p1 in zip(inputs,points_mask[0],points_mask[1]):
        batch_element_perms = []  
        batch_p0_perms = []
        batch_p1_perms = []

        ids_perm = np.arange(n)

        for ix in range(n):
            ids_perm = np.roll(ids_perm,-ix)
            batch_element_perms.append(batch_element[torch.LongTensor(ids_perm)])
            batch_p0_perms.append(p0[ids_perm])
            batch_p1_perms.append(p1[ids_perm])
        
        
        batch_element_perms = torch.stack(batch_element_perms)
        batch_perms.append(batch_element_perms)
        batch_p0_perms = np.array(batch_p0_perms)
        batch_p0.append(batch_p0_perms)
        batch_p1_perms = np.array(batch_p1_perms)
        batch_p1.append(batch_p1_perms)

    # b,n,s,i -> b,n,n,s,i
    batch_perms = torch.stack(batch_perms)
    batch_p0 = np.array(batch_p0)
    batch_p1 = np.array(batch_p1)

    # b,n,n,s,i -> b*n,n,s,i
    batch_perms = batch_perms.view(-1,n,s,i)
    batch_p0 = batch_p0.reshape(-1,n,s,i)
    batch_p1 = batch_p1.reshape(-1,n,p,i)

    # save inputs
    inputs_temp = inputs
    points_mask_temp = points_mask
    active_mask_temp = active_mask

    # new inputs from permutations
    inputs = batch_perms
    points_mask = (batch_p0,batch_p1)
    active_mask = torch.arange(inputs.size()[0]*inputs.size()[1]).to(device)

    # prediction
    outputs = net((inputs,types,active_mask,points_mask))

    # reset inputs
    inputs = inputs_temp
    points_mask = points_mask_temp
    active_mask = active_mask_temp

    # select outputs
    outputs = outputs[:,0]
    outputs = outputs.view(b,n,p,i)
    return outputs,inputs,types,active_mask,points_mask
        
def predict_naive(inputs,types,active_mask,points_mask,net,device):
    b,n,s,i = points_mask[0].shape
    b,n,p,i = points_mask[1].shape

    inputs = inputs.view(-1,s,i).unsqueeze(1)
    types = types.view(-1).unsqueeze(1)
    points_mask[0] = np.expand_dims(points_mask[0].reshape(-1,s,i),1)
    points_mask[1] = np.expand_dims(points_mask[1].reshape(-1,p,i),1)
    
    # prediction
    outputs = net((inputs,types,active_mask,points_mask))

    # b*n,s,i -> b,n,s,i
    outputs = outputs.squeeze(1).view(b,n,p,i)
    inputs = inputs.squeeze(1).view(b,n,s,i)
    types = types.squeeze(1).view(b,n)
    points_mask[0] = points_mask[0].squeeze(1).reshape(b,n,s,i)
    points_mask[1] = points_mask[1].squeeze(1).reshape(b,n,p,i)

    return outputs,inputs,types,active_mask,points_mask
    

################ Criterions ###########################


def convert_losses(losses,prefix,losses_in):
    for key in losses_in:
        if key not in losses:
            losses[key] = {}
        for loss in losses_in[key]:
            losses[key][prefix+loss] = losses_in[key][loss]
    return losses

def apply_criterion(criterion,scene_files):
    results = {"global":{"joint":[],"disjoint":[]}}
    for scene_file in scene_files:
        scene = scene_file.split("/")[-1].split(".")[0].split("_")[0]
        results[scene] = {"joint":[],"disjoint":[]}
        # print(scene)

        scene_file = json.load(open(scene_file))
        for sample in scene_file:
            sample = scene_file[sample]
            labels = np.array(sample["labels"])
            outputs = np.array(sample["outputs"])
            point_mask = np.array(sample["points_mask"])
            
            loss = criterion(outputs.copy(), labels.copy(),point_mask.copy())
            loss_unjoint = criterion(outputs[0].copy(), labels[0].copy(),point_mask[0].copy())

            results[scene]["joint"].append(loss)
            results[scene]["disjoint"].append(loss_unjoint)

            results["global"]["joint"].append(loss)
            results["global"]["disjoint"].append(loss_unjoint)
        
        results[scene]["joint"] = np.mean(results[scene]["joint"])
        results[scene]["disjoint"] = np.mean(results[scene]["disjoint"])
    results["global"]["joint"] = np.mean(results["global"]["joint"])
    results["global"]["disjoint"] = np.mean(results["global"]["disjoint"])
    return results
            
def spatial(scene_files,types_to_spatial,images,spatial_annotations,spatial_profiles,correspondences_trajnet,correspondences_manual):
    nb_sample = 0
    spatial_conflicts_results = { "global": {"groundtruth":0,"pred":0}}
    for scene_file in scene_files:
        scene = scene_file.split("/")[-1].split(".")[0].split("_")[0]

        spatial_conflicts_results[scene] = {"groundtruth":0,"pred":0}

        scene_file = json.load(open(scene_file))

        #compute mask for spatial structure
        spatial_masks = scene_mask(scene,images,spatial_annotations,spatial_profiles)
        #get rtio for meter to pixel conversion
        meters_to_pixel = get_factor(scene,correspondences_trajnet,correspondences_manual)

        # print(scene)
        nb_sample_scene = 0
        for sample in scene_file:
            sample = scene_file[sample]
            label = np.array(sample["labels"][0]) * meters_to_pixel
            output = np.array(sample["outputs"][0]) * meters_to_pixel
            type_ = sample["types"][0]
            spatial_profile_id = types_to_spatial[str(int(type_))]
            mask = spatial_masks[spatial_profile_id]
            nb_pt = len(output)
            
            spatial_pred = spatial_conflicts(mask,output)
            spatial_gt = spatial_conflicts(mask,label)

            spatial_conflicts_results["global"]["groundtruth"] += spatial_gt
            spatial_conflicts_results["global"]["pred"] += spatial_pred

            spatial_conflicts_results[scene]["groundtruth"] += spatial_gt
            spatial_conflicts_results[scene]["pred"] += spatial_pred

            nb_sample_scene += 1
        spatial_conflicts_results[scene]["groundtruth"] /= float(nb_pt*nb_sample_scene)
        spatial_conflicts_results[scene]["pred"] /= float(nb_pt*nb_sample_scene)
        nb_sample += nb_sample_scene
    spatial_conflicts_results["global"]["groundtruth"] /= float(nb_pt*nb_sample)
    spatial_conflicts_results["global"]["pred"] /= float(nb_pt*nb_sample)  
    return spatial_conflicts_results     

def spatial_conflicts(mask,trajectory_p):
    ctr = 0
    # print(mask.shape)
    for point in trajectory_p:
        #case out of frame
        if int(point[0]) in range(0,mask.shape[0]) and int(point[1]) in range(0,mask.shape[1]):
            if mask[int(point[0]),int(point[1])]:
                    ctr += 1
    return ctr

# def spatial_loss(spatial_mask,trajectory,meters_to_pixel):
#     trajectory *= meters_to_pixel
#     nb = spatial_conflicts(spatial_mask,trajectory)    
#     return nb

def social_conflicts(scene_files):
    social_results = {}
    conflict_thresholds = [0.1,0.5,1.0]
    social_results["global"] = {}
    for thresh in conflict_thresholds:
        social_results["global"]["joint_"+str(thresh)] = []
        social_results["global"]["disjoint_"+str(thresh)] = []
        social_results["global"]["groundtruth_"+str(thresh)] = []


    for scene_file in scene_files:
        scene = scene_file.split("/")[-1].split(".")[0].split("_")[0]
        social_results[scene] = {}
        for thresh in conflict_thresholds:
            social_results[scene]["joint_"+str(thresh)] = []
            social_results[scene]["disjoint_"+str(thresh)] = []
            social_results[scene]["groundtruth_"+str(thresh)] = []
        # print(scene)
        scene_file = json.load(open(scene_file))
        for sample in scene_file:
            sample = scene_file[sample]
            labels = sample["labels"]
            outputs = sample["outputs"]

    # social loss
            social_sample = copy.copy(labels)
            social_sample[0] = outputs[0]
            social_sample = np.array(social_sample)
            labels = np.array(labels)
            outputs = np.array(outputs)    

            # social loss
            for thresh in conflict_thresholds:
                # print(thresh)
                frames_joint = conflicts(outputs,thresh)
                frames_disjoint = conflicts(social_sample,thresh)
                frames_gt = conflicts(labels,thresh)


                social_results["global"]["joint_"+str(thresh)] += frames_joint
                social_results["global"]["disjoint_"+str(thresh)] += frames_disjoint
                social_results["global"]["groundtruth_"+str(thresh)] += frames_gt

                social_results[scene]["joint_"+str(thresh)] += frames_joint
                social_results[scene]["disjoint_"+str(thresh)] += frames_disjoint
                social_results[scene]["groundtruth_"+str(thresh)] += frames_gt
        for thresh in conflict_thresholds:
            social_results[scene]["joint_"+str(thresh)] = np.mean(social_results[scene]["joint_"+str(thresh)])
            social_results[scene]["disjoint_"+str(thresh)] = np.mean(social_results[scene]["disjoint_"+str(thresh)])
            social_results[scene]["groundtruth_"+str(thresh)] = np.mean(social_results[scene]["groundtruth_"+str(thresh)])

    for thresh in conflict_thresholds:
        social_results["global"]["joint_"+str(thresh)] = np.mean(social_results["global"]["joint_"+str(thresh)])
        social_results["global"]["disjoint_"+str(thresh)] = np.mean(social_results["global"]["disjoint_"+str(thresh)])
        social_results["global"]["groundtruth_"+str(thresh)] = np.mean(social_results["global"]["groundtruth_"+str(thresh)])
    return social_results

                    



        # print(wasserstein_distance(distrib_pred,distrib_real))
def conflicts(trajectories,threshold = 0.5):
    timesteps = []
    for t in range(trajectories.shape[1]):
        points = np.array(trajectories[:,t])
        conflict_prop = conflicts_frame(points,threshold)
        timesteps.append(conflict_prop)
    return timesteps

def conflicts_frame(points,threshold):
    d = distance_matrix(points,points)

    m = (d < threshold).astype(int) - np.eye(len(points))
    total_count = np.ones_like(m)
    m = np.triu(m,1)
    total_count = np.triu(total_count,1)


    if float(total_count.sum()) > 0.:           
        conflict_prop = m.sum() / float(total_count.sum())
    else:
        conflict_prop = 0
    return conflict_prop
        


def get_distrib_conflicts(scene_files):
    distrib_pred_disjoint = {"global":[]}
    distrib_pred = {"global":[]}
    distrib_real = {"global":[]}
    results = {}


    for scene_file in scene_files:
        scene = scene_file.split("/")[-1].split(".")[0].split("_")[0]
        # print(scene)

        distrib_pred[scene] = []
        distrib_pred_disjoint[scene] = []
        distrib_real[scene] = []



        scene_file = json.load(open(scene_file))
        for sample in scene_file:
            sample = scene_file[sample]
            labels = sample["labels"]
            outputs = sample["outputs"]
            # inputs = sample["inputs"]
            types = sample["types"]
            point_mask = sample["points_mask"]

            social_sample = copy.copy(labels)
            social_sample[0] = outputs[0]
            social_sample = np.array(social_sample)
            labels = np.array(labels)
            outputs = np.array(outputs)


            distances_pred_disjoint = get_distances_agents_interval(social_sample)
            distances_pred = get_distances_agents_interval(outputs)
            distances_real = get_distances_agents_interval(labels)

            distrib_pred_disjoint["global"] += distances_pred_disjoint
            distrib_pred["global"] += distances_pred
            distrib_real["global"] += distances_real

            distrib_pred_disjoint[scene] += distances_pred_disjoint
            distrib_pred[scene] += distances_pred
            distrib_real[scene] += distances_real
        results[scene] = {}
        results[scene]["disjoint"] = wasserstein_distance(distrib_pred_disjoint[scene],distrib_real[scene])
        results[scene]["joint"] = wasserstein_distance(distrib_pred[scene],distrib_real[scene])



    results["global"] = {}
    results["global"]["disjoint"] = wasserstein_distance(distrib_pred_disjoint["global"],distrib_real["global"])
    results["global"]["joint"] = wasserstein_distance(distrib_pred["global"],distrib_real["global"])

    return results
    

def get_distances_agents_frame(points):
    d = distance_matrix(points,points)
    d = np.triu(d,1).flatten()
    distances = [e for e in d if e != 0.]
    return distances 
def get_distances_agents_interval(trajectories):
    distances_interval = []
    for t in range(trajectories.shape[1]):
        points = np.array(trajectories[:,t])
        distances = get_distances_agents_frame(points)
        distances_interval += distances
    return distances_interval

def speeds_distance(scene_files,types_dic,delta_t):
    speed_real_distribution = {}
    speed_predicted_distribution = {}

    speed_real_distribution["global"] = []
    speed_predicted_distribution["global"] = []


    for scene_file in scene_files:
        scene_file = json.load(open(scene_file))
        for sample in scene_file:
            sample = scene_file[sample]
            labels = sample["labels"]
            outputs = sample["outputs"]
            # inputs = sample["inputs"]
            types = sample["types"]
            point_mask = sample["points_mask"]

            type_str = types_dic[str(int(types[0]))]
            if type_str not in speed_predicted_distribution:
                speed_predicted_distribution[type_str] = []
            if type_str not in speed_real_distribution:
                speed_real_distribution[type_str] = []

            speeds_labels = get_speeds(labels[0],delta_t)
            speeds_outputs = get_speeds(outputs[0],delta_t)

            speed_real_distribution[type_str] += speeds_labels
            speed_predicted_distribution[type_str] += speeds_outputs
            speed_predicted_distribution["global"] += speeds_outputs
            speed_real_distribution["global"] += speeds_labels

    results = {}
    for type_ in speed_real_distribution:
        results[type_] = wasserstein_distance(speed_predicted_distribution[type_],speed_real_distribution[type_])
    return results

def accelerations_distance(scene_files,types_dic,delta_t):
    acceleration_real_distribution = {}
    acceleration_predicted_distribution = {}

    acceleration_real_distribution["global"] = []
    acceleration_predicted_distribution["global"] = []


    for scene_file in scene_files:
        scene_file = json.load(open(scene_file))
        for sample in scene_file:
            sample = scene_file[sample]
            labels = sample["labels"]
            outputs = sample["outputs"]
            # inputs = sample["inputs"]
            types = sample["types"]
            point_mask = sample["points_mask"]

            type_str = types_dic[str(int(types[0]))]
            if type_str not in acceleration_predicted_distribution:
                acceleration_predicted_distribution[type_str] = []
            if type_str not in acceleration_real_distribution:
                acceleration_real_distribution[type_str] = []

            speeds_labels = get_speeds(labels[0],delta_t)
            speeds_outputs = get_speeds(outputs[0],delta_t)

            accelerations_labels = get_accelerations(speeds_labels,delta_t)
            accelerations_outputs = get_accelerations(speeds_outputs,delta_t)

            acceleration_real_distribution[type_str] += accelerations_labels
            acceleration_predicted_distribution[type_str] += accelerations_outputs
            acceleration_predicted_distribution["global"] += accelerations_outputs
            acceleration_real_distribution["global"] += accelerations_labels

    results = {}
    for type_ in acceleration_real_distribution:
        results[type_] = wasserstein_distance(acceleration_predicted_distribution[type_],acceleration_real_distribution[type_])
    return results