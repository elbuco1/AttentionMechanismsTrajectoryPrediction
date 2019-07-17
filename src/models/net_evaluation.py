
import matplotlib.image as mpimg
import cv2

import copy

import time

import json
import torch
import sys
import helpers.helpers_training as helpers
import helpers.helpers_evaluation as helpers_evaluation
import torch.nn as nn
import numpy as np
import os 
import copy
import matplotlib.pyplot as plt



def main():

    

    #loading parameters
    parameters_path = "./src/parameters/project.json"
    parameters_project = json.load(open(parameters_path))  
    evaluation_parameters = json.load(open(parameters_project["evaluation_parameters"])) 
    processed_parameters = json.load(open(parameters_project["data_processed_parameters"]))
    raw_parameters = json.load(open(parameters_project["data_raw_parameters"]))



    # loading training data
    report_name = evaluation_parameters["report_name"]

    dir_name = parameters_project["evaluation_reports"] + "{}/".format(report_name)
    sub_dir_name = parameters_project["evaluation_reports"] + "{}/scene_reports/".format(report_name) 

    scenes = [ f.split("_")[0]  for f in  os.listdir(sub_dir_name) if "json" in f]
    print(scenes)
    types_dic = processed_parameters["types_dic_rev"]
    start = time.time()
    if os.path.exists(dir_name) and os.path.exists(sub_dir_name):

        dynamic_losses = {}
        scene_files = [ sub_dir_name+f  for f in  os.listdir(sub_dir_name) if "json" in f]
        print("speed distribution")
        speed_results = helpers_evaluation.speeds_distance(scene_files,types_dic,1.0/float(raw_parameters["new_framerate"]))
        print(time.time()-start)
        print("acceleration distribution")
        acceleration_results = helpers_evaluation.accelerations_distance(scene_files,types_dic,1.0/float(raw_parameters["new_framerate"]))
        print(time.time()-start)

        dynamic_losses["speed"] = speed_results
        dynamic_losses["acceleration"] = acceleration_results
        json.dump(dynamic_losses,open(dir_name + "dynamic_losses.json","w"),indent=2)

        losses = {}

        print("social conflicts distribution")
        conflicts_distrib_results = helpers_evaluation.get_distrib_conflicts(scene_files)
        helpers_evaluation.convert_losses(losses,"social_wsstn_",conflicts_distrib_results)

       
            
        # # print(conflicts_distrib_results)
        # # print(time.time()-start)

        print("social_conflicts")
        social_results = helpers_evaluation.social_conflicts(scene_files)
        helpers_evaluation.convert_losses(losses,"social_",social_results)

        print(time.time()-start)
        
        
        # print(spatial_conflicts_results)
        print("ade")        
        results_ade = helpers_evaluation.apply_criterion(helpers_evaluation.ade,scene_files)
        helpers_evaluation.convert_losses(losses,"ade_",results_ade)       
        print(time.time()-start)

        print("fde")
        results_fde = helpers_evaluation.apply_criterion(helpers_evaluation.fde,scene_files)
        helpers_evaluation.convert_losses(losses,"fde_",results_fde)
        print(time.time()-start)

        json.dump(losses,open(dir_name + "losses.json","w"),indent=2)





if __name__ == "__main__":
    main()