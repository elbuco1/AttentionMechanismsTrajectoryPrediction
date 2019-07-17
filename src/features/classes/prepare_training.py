import csv
import torch
from scipy.spatial.distance import euclidean
import random
import os
import json
import h5py
import numpy as np
from sklearn.preprocessing import OneHotEncoder


class PrepareTraining():
    def __init__(self,project_args):
    # def __init__(self,data,torch_params):
        project_args = json.load(open(project_args))
        processed_parameters = json.load(open(project_args["data_processed_parameters"]))

        self.padding = processed_parameters["padding"]     

        self.original_hdf5 = project_args["hdf5_samples"]
        self.training_hdf5 = project_args["training_hdf5"]  

        self.test_scenes = list(processed_parameters["test_scenes"])
        self.train_eval_scenes = list(processed_parameters["train_scenes"])
        self.eval_scenes = list(processed_parameters["eval_scenes"])
        self.train_scenes = [ scene for scene in self.train_eval_scenes if scene not in self.eval_scenes]


        self.seq_len = processed_parameters["t_obs"] + processed_parameters["t_pred"]
        self.nb_types = len(processed_parameters["types_dic"].keys()) + 1
        self.ohe = self.__get_ohe_types()

    
       

    def create_training_file(self):
        max_neighboors = self.__max_neighbors()

        if os.path.exists(self.training_hdf5):
            os.remove(self.training_hdf5)

        print("Creating the test dataset")
        self.split_dset("test_trajectories",max_neighboors,"trajectories",self.test_scenes,1.0)
        print("Creating the train_eval dataset")
        self.split_dset("train_eval_trajectories",max_neighboors,"trajectories",self.train_eval_scenes,1.0)
        print("Creating the train dataset")

        self.split_dset("train_trajectories",max_neighboors,"trajectories",self.train_scenes,1.0)
        print("Creating the eval dataset")

        self.split_dset("eval_trajectories",max_neighboors,"trajectories",self.eval_scenes,1.0)    


        with h5py.File(self.training_hdf5,"r") as dest_file:
            print("Datasets dimensions")
            for key in dest_file:
                print(key ," ",dest_file[key].shape)



    def __max_neighbors(self):
        max_n_1 = 0
        with h5py.File(self.original_hdf5,"r") as original_file:
            for key in original_file["trajectories"]:
                dset = original_file["trajectories"][key]
                max_ = dset[0].shape[0]
                if max_ > max_n_1:
                    max_n_1 = max_ 
        return max_n_1


    """ if prop > 0 then we take samples up to prop * nb_samples
     if prop < then we take samples from  prop * nb_samples to the end """
    def split_dset(self,name,max_neighboors,sample_type,scene_list,prop):
        with h5py.File(self.original_hdf5,"r") as original_file:
            with h5py.File(self.training_hdf5,"a") as dest_file:
                   
                # create partitions in hdf5 file for samples and agents types
                samples = dest_file.create_dataset("samples_{}".format(name),shape=(0,max_neighboors,self.seq_len,2),maxshape = (None,max_neighboors,self.seq_len,2),dtype='float32',chunks=(15,max_neighboors,self.seq_len,2))
                types = dest_file.create_dataset("types_{}".format(name),shape=(0,max_neighboors,self.nb_types-1),maxshape = (None,max_neighboors,self.nb_types-1),dtype='float32')

                
                # for each scene in the former hdf5 file, add
                # the samples to a global dataset, mixing all the scenes
                for key in scene_list:                    
                    dset = original_file[sample_type][key]
                    dset_types = original_file[sample_type][key+"_types"]

                    nb_neighbors = dset[0].shape[0] 
                    nb_samples = int(prop*dset.shape[0])

                    # variable number of agents in samples so we need padding
                    padding = np.ones(shape = (np.abs(nb_samples), max_neighboors-nb_neighbors,self.seq_len,2))
                    padding = padding * self.padding
                    padding_types = np.zeros(shape = (np.abs(nb_samples), max_neighboors-nb_neighbors,self.nb_types - 1))


                    # resize hdf5 array size
                    samples.resize(samples.shape[0]+np.abs(nb_samples),axis=0)
                    types.resize(types.shape[0]+np.abs(nb_samples),axis=0)


                    
                    # add the first nb_samples samples of the scene to the new dataset
                    if nb_samples > 0:
                        samples[-nb_samples:] = np.concatenate((dset[:nb_samples],padding),axis = 1)
                        ohe_types = np.array([self.ohe.transform(d.reshape(-1,1))  for d in dset_types[:nb_samples]])
                        types[-nb_samples:] = np.concatenate((ohe_types[:,:,1:],padding_types),axis = 1)
                    # add the last nb_samples samples of the scene to the new dataset
                    else:
                        samples[nb_samples:] = np.concatenate((dset[nb_samples:],padding),axis = 1)
                        ohe_types = np.array([self.ohe.transform(d.reshape(-1,1))  for d in dset_types[nb_samples:]])
                        types[nb_samples:] = np.concatenate((ohe_types[:,:,1:],padding_types),axis = 1)

    def __get_ohe_types(self):
        cat = np.arange(self.nb_types).reshape(self.nb_types,1)

        ohe = OneHotEncoder(sparse = False,categories = "auto")
        ohe = ohe.fit(cat)
        return ohe


                    