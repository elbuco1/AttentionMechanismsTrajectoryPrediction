import torch
from torch.utils import data
# from classes.pretrained_vgg import customCNN1

import cv2
import numpy as np 
import json
import h5py
import cv2
import time
import helpers.helpers_training as helpers
from models.pretrained_vgg import customCNN1

# import helpers
from joblib import load
import sys
from PIL import Image
from torchvision import transforms

class CustomDataLoader():
      def __init__(self,batch_size,shuffle,drop_last,dataset,test = 0):
            self.shuffle = shuffle 
            self.dataset = dataset
            self.data_len = self.dataset.get_len()

            self.batch_size = batch_size
            self.drop_last = drop_last
            self.test = test
            self.split_batches()

      def split_batches(self):
            self.batches = list(torch.utils.data.BatchSampler(
                  torch.utils.data.RandomSampler(range(self.data_len)),
                  batch_size = self.batch_size,
                  drop_last =self.drop_last))
            self.batch_idx = 0
            self.nb_batches = len(self.batches)
            if self.test :
                  self.nb_batches = 30

            

      def __iter__(self):
            return self
      def __next__(self):

            if self.batch_idx >= self.nb_batches:
                  self.split_batches()
                  raise StopIteration
            else:     
                  ids = sorted(self.batches[self.batch_idx])
                  self.batch_idx += 1 
                  return self.dataset.get_ids(ids)



"""
      set_type:  train eval  test train_eval
      use_images: True False
      use_neighbors: True False
      predict_offsets: 0: none, 1: based on last obs point, 2: based on previous point

      data_type: frames trajectories
"""
class Hdf5Dataset():
      'Characterizes a dataset for PyTorch'
      def __init__(self,padding,hdf5_file,scene_list,t_obs,t_pred,set_type, data_type,use_neighbors,use_images,images_path,
                  pixel_to_meters = {}, froze_cnn = 1, use_masks = False,reduce_batches = True, predict_offsets = 0, 
                  offsets_input = 0,evaluation = 0, data_augmentation = False):
            
            self.froze_cnn = froze_cnn
            self.use_images = use_images   
            self.images_path = images_path + "{}.jpg"

            self.set_type = set_type
            self.scene_list = scene_list

            self.data_type = data_type
            self.use_neighbors = use_neighbors
            self.use_masks = use_masks

            self.evaluation = evaluation

            self.reduce_batches = reduce_batches
            self.predict_offsets = predict_offsets
            self.offsets_input = offsets_input

            self.hdf5_file = h5py.File(hdf5_file,"r")
            self.pixel_to_meters = pixel_to_meters

            self.data_augmentation = data_augmentation

            if self.evaluation:
                  self.dset_name = self.scene_list[0]
                  self.dset_types = "{}_types".format(self.scene_list[0])
                  self.coord_dset = self.hdf5_file[self.data_type][self.dset_name]
                  self.types_dset = self.hdf5_file[self.data_type][self.dset_types]  

            else: 
                  self.dset_name = "samples_{}_{}".format(set_type,data_type)
                  self.dset_types = "types_{}_{}".format(set_type,data_type)   
                  self.coord_dset = self.hdf5_file[self.dset_name]  
                  self.types_dset = self.hdf5_file[self.dset_types]  
                  self.dset_img_name = "images_{}_{}".format(set_type,data_type)
                  self.scenes_dset = self.hdf5_file[self.dset_img_name]  
            
            self.t_obs = t_obs
            self.t_pred = t_pred
            self.seq_len = t_obs + t_pred
            self.padding = padding

            if self.data_augmentation:      
                  self.augmentation_angles = [0, 90, 180, 270]           
                  self.scene_centers = self.__get_scenes_centers()
                  self.rotation_matrices = self.__get_matrices()


            self.shape = self.coord_dset.shape
            if self.use_images:
                  if self.froze_cnn:
                        self.images = self.__load_images_features()
                  else:
                        self.images = self.__load_images()

            
      def __del__(self):
            self.hdf5_file.close()
      def get_len(self):
            if self.data_augmentation:
                  return self.shape[0] * len(self.augmentation_angles)
            else:
                  return self.shape[0]

      


      def get_ids(self,ids):
            if self.data_augmentation:
                  ids, matrices_ids = self.__get_real_ids(ids)
                  ids,repetitions = self.__augmented_ids_repetition(ids)

            types,X,y,seq = [],[],[],[]
            max_batch = self.coord_dset.shape[1]
 
            scenes = [self.scene_list[0] for _ in range(len(ids))]
            if not self.evaluation:                             
                  scenes = [img.decode('UTF-8') for img in self.scenes_dset[ids]]
            
            seq = self.coord_dset[ids]
            if self.use_neighbors:
                  types = self.types_dset[ids,:max_batch] #B,N,tpred,2
            else:
                  types =  self.types_dset[ids,0] #B,1,tpred,2
            
            
            if self.data_augmentation:
                  scenes, seq, types = self.__repeat_augmentation(scenes,seq,types, repetitions)
                  rotation_matrices = self.__get_rotation_matrices(matrices_ids)
                  translation_matrices = self.__get_translation_matrices(scenes)
                  seq = self.__augment_batch(seq, rotation_matrices, translation_matrices)



            

            X = seq[:,:,:self.t_obs]
            y = seq[:,:,self.t_obs:self.seq_len]

            # compute max nb of agents in a frame
            if self.reduce_batches:
                  max_batch = self.__get_batch_max_neighbors(X)

            X = X[:,:max_batch]
            y = y[:,:max_batch]
            seq = seq[:,:max_batch]

            


            points_mask = []
            if self.use_neighbors:
                  X,y,points_mask,y_last,X_last = self.__get_x_y_neighbors(X,y,seq)
            else:       
                  X,y,points_mask,y_last,X_last = self.__get_x_y(X,y,seq)                      


            sample_sum = (np.sum(points_mask[1].reshape(points_mask[1].shape[0],points_mask[1].shape[1],-1), axis = 2) > 0).astype(int)
            active_mask = np.argwhere(sample_sum.flatten()).flatten()


            out = [
                  torch.FloatTensor(X).contiguous(),
                  torch.FloatTensor(y).contiguous(),
                  torch.FloatTensor(types)
            ]   
            if self.use_masks:
                  out.append(points_mask)
                  out.append(torch.LongTensor(active_mask))
            
            imgs = torch.FloatTensor([])
            if self.use_images:
                  imgs = torch.stack([self.images[img] for img in scenes],dim = 0) 
            out.append(imgs)

            out.append(y_last)
            out.append(X_last)

            return tuple(out)


      def __get_batch_max_neighbors(self,X):
           
            active_mask = (X == self.padding).astype(int)
            a = np.sum(active_mask,axis = 3)
            b = np.sum( a, axis = 2)
            nb_padding_traj = b/float(2.0*self.t_obs) #prop of padded points per traj
            active_traj = nb_padding_traj < 1.0 # if less than 100% of the points are padding points then its an active trajectory
            nb_agents = np.sum(active_traj.astype(int),axis = 1)                      
            max_batch = np.max(nb_agents)

            return max_batch

           
      def __get_x_y_neighbors(self,X,y,seq):
            active_mask = (y != self.padding).astype(int)    
            active_mask_in = (X != self.padding).astype(int)            
            active_last_points = []
            original_x = []

            if self.predict_offsets:
                  if self.predict_offsets == 1:
                        # offsets according to last obs point, take last point for each obs traj and make it an array of dimension y
                        last_points = np.repeat(  np.expand_dims(X[:,:,-1],2),  self.t_pred, axis=2)#B,N,tpred,2
                  elif self.predict_offsets == 2:# y shifted left

                        # offsets according to preceding point point, take points for tpred shifted 1 timestep left
                        last_points = seq[:,:,self.t_obs-1:self.seq_len-1]


                  
                  active_last_points = np.multiply(active_mask,last_points)
                  y = np.subtract(y,active_last_points)
            if self.offsets_input:
                  first_points = np.concatenate([np.expand_dims(X[:,:,0],2), X[:,:,0:self.t_obs-1]], axis = 2)
                  active_first_points = np.multiply(active_mask_in,first_points)
                  original_x = X
                  original_x = np.multiply(original_x,active_mask_in) # put padding to 0

                  
                  X = np.subtract(X,active_first_points)


            y = np.multiply(y,active_mask) # put padding to 0
            X = np.multiply(X,active_mask_in) # put padding to 0
            
            return X,y,(active_mask_in,active_mask),active_last_points,original_x 

      def __get_x_y(self,X,y,seq):

            X = np.expand_dims( X[:,0] ,1) # keep only first neighbors and expand nb_agent dim 
            y = np.expand_dims( y[:,0], 1) #B,1,tpred,2 # keep only first neighbors and expand nb_agent dim 
            seq = np.expand_dims( seq[:,0], 1) #B,1,tpred,2 # keep only first neighbors and expand nb_agent dim 
            
            active_last_points = []
            original_x = []

            
            active_mask = (y != self.padding).astype(int)
            active_mask_in = (X != self.padding).astype(int)            

            if self.predict_offsets:

                  if self.predict_offsets == 1 :
                        last_points = np.repeat(  np.expand_dims(X[:,:,-1],2),  self.t_pred, axis=2) #B,1,tpred,2
                  
                  elif self.predict_offsets == 2: # y shifted left
                        last_points = seq[:,:,self.t_obs-1:self.seq_len-1]

                  active_last_points = np.multiply(active_mask,last_points)
                  y = np.subtract(y,active_last_points)

            if self.offsets_input:
                  # concatenate the first point of X to X in order to get as many offsets as position
                  first_points = np.concatenate([np.expand_dims(X[:,:,0],2), X[:,:,0:self.t_obs-1]], axis = 2)

                  # apply active mask of input points
                  active_first_points = np.multiply(active_mask_in,first_points)

                  # keep original inputs
                  original_x = X
                  # apply the input active mask on the original inputs to remove the padding
                  original_x = np.multiply(original_x,active_mask_in) # put padding to 0

                  # subtract x shifted right to x in order to get offsets, offsets[0] = 0
                  X = np.subtract(X,active_first_points)
                  
            y = np.multiply(y,active_mask) # put padding to 0
            X = np.multiply(X,active_mask_in) # put padding to 0


            return X,y,(active_mask_in,active_mask),active_last_points,original_x
      
      
      def __load_images_features(self):#cuda
            images = {}
            print("loading images features")
            # cnn = customCNN1()
            # cnn.eval()
            # cnn = cnn.cuda()
            paddings = self.__get_paddings()
            for scene,pad in zip(self.scene_list,paddings):
                  img = Image.open(self.images_path.format(scene))
                  transform = transforms.Compose([
                        transforms.Pad(pad),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                  ])

                  cnn = customCNN1()
                  cnn.eval()
                  # cnn = cnn.cuda()

                  img = transform(img)
                  # img = img.cuda()
                  
                  img = img.unsqueeze(0)
                  img = cnn(img)
                  img = img.squeeze(0)

                  # img = img.cpu()
                  # cnn = cnn.cpu()

                  images[scene] = img
                  

            print("Done!")
            
            return images

      def __load_images(self):#cuda
            images = {}
            print("loading images ")
            paddings = self.__get_paddings()
            for scene,pad in zip(self.scene_list,paddings):
                  img = Image.open(self.images_path.format(scene))
                  transform = transforms.Compose([
                        transforms.Pad(pad),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                  ])               
                  img = transform(img)
                  images[scene] = img                

            print("Done!")            
            return images
            
      def __get_paddings(self):
            widths,heights = [],[]
            for scene in self.scene_list:
                  img = np.array(Image.open(self.images_path.format(scene)))
                  height,width,_ = img.shape
                  heights.append(height)
                  widths.append(width)
            max_height = np.max(heights)
            max_width = np.max(widths)
            max_dim = max(max_height,max_width)
            paddings = []
            for scene in self.scene_list:
                  img = np.array(Image.open(self.images_path.format(scene)))
                  height,width,_ = img.shape
                  pad_height = max_dim - height
                  pad_width = max_dim  - width 

                  pad_height = self.__get_pad(pad_height)
                  pad_width = self.__get_pad(pad_width)
                  padding = (pad_width[0],pad_height[0],pad_width[1],pad_height[1])
                  paddings.append(padding)
            return paddings

      def __get_pad(self,v):
            if v % 2 == 0:
                  v = int(v/2)
                  return (v,v)
            else:
                  v = int(v/2)
                  return (v,v+1)










      #### Data augmentation methods #####
      def __get_scenes_centers(self):
            centers = {}
            for scene in self.scene_list:
                  img = Image.open(self.images_path.format(scene))
                  pixel_to_meter = self.pixel_to_meters[scene]
                  width, height = img.size 
                  width_meter = pixel_to_meter * width
                  height_meter = pixel_to_meter * height
                  width_middle = int(width_meter/2.0 * 10**2) / 10 ** 2
                  height_middle = int(height_meter/2.0 * 10**2) / 10**2
                  center = (width_middle, height_middle)
                  centers[scene] = center    
            return centers

      def __get_matrices(self, divisible_by = 90):
            rotation_matrices = {}
            for i, theta in enumerate(self.augmentation_angles):
                  assert theta % divisible_by == 0
                  rotation_matrices[i] = self.__get_matrix(theta)
            return rotation_matrices

      def __get_matrix(self, theta_deg):
            theta_rad = np.radians(theta_deg)
            c, s = int(np.cos(theta_rad)), int(np.sin(theta_rad))
            r = np.array([
                        [c,-s],
                        [s,c]
                        ])
            return r
      
      def __get_real_ids(self, augmented_ids):
            ids = np.array(augmented_ids) % self.shape[0]
            theta_ids = (np.array(augmented_ids) / self.shape[0]).astype(int)

            for id_ in ids:
                  assert id_ > -1 and id_ < self.shape[0]

            arg_ids = np.argsort(ids)
            ids = ids[arg_ids]
            theta_ids = theta_ids[arg_ids]

            return ids.tolist(), theta_ids.tolist()

      def __augmented_ids_repetition(self, augmented_ids ):
            unique_ids = []
            nb_ids = []
            for id_ in augmented_ids:
                  if id_ not in unique_ids:
                        unique_ids.append(id_)
                        nb_ids.append(0)
                  nb_ids[-1] += 1
            return unique_ids,nb_ids

      def __repeat_augmentation(self, scenes,seq,types, repetitions):
            rscenes, rseq, rtypes = [], [], []
            for scene, s, t, repetition in zip(scenes,seq,types, repetitions):
                  for _ in range(repetition):
                        rscenes.append(scene)
                        rseq.append(s)
                        rtypes.append(t)
            seq = np.array(rseq)
            types = np.array(rtypes)
            return scenes,seq,types

      def __get_rotation_matrices(self, matrices_ids):
            matrices = []
            for id_ in matrices_ids:
                  matrices.append(self.rotation_matrices[id_])
            matrices = np.array(matrices)
            return matrices

      def __get_translation_matrices(self, scenes):
            translation_matrices = []
            
            for scene in scenes:
                  translation_matrix = list(self.scene_centers[scene])      
                  translation_matrices.append(translation_matrix)
            translation_matrices = np.array(translation_matrices)
            translation_matrices *= -1
            return translation_matrices

      def __augment_batch(self, seq, rotation_matrices, translation_matrices):
            b,n,s,i = seq.shape
            _,r,_ = rotation_matrices.shape
            
            real_positions = (seq != self.padding).astype(int)
            
            rotation_matrices = np.expand_dims(rotation_matrices, 0)
            translation_matrices = np.expand_dims(translation_matrices,  0)

            rotation_matrices = np.repeat(rotation_matrices, n*s, axis = 0 )
            translation_matrices = np.repeat(translation_matrices, n*s, axis = 0 )

            rotation_matrices = np.transpose(rotation_matrices,(1,0,2,3))
            translation_matrices = np.transpose(translation_matrices,(1,0,2))

            rotation_matrices = rotation_matrices.reshape(b,n,s,r,r)
            translation_matrices = translation_matrices.reshape(b,n,s,r)

            translation_matrices = np.multiply(real_positions,translation_matrices)
            real_positions = np.expand_dims(real_positions, -1)
            rotation_matrices = np.multiply(real_positions,rotation_matrices)



            seq = np.add(seq, translation_matrices)

            seq = np.expand_dims(seq,  -1)

            seq = np.matmul(rotation_matrices, seq)
            seq = np.squeeze(seq,  -1)
            seq = np.subtract(seq, translation_matrices )

            return seq
      