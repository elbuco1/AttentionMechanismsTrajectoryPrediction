import helpers.helpers as helpers
import json
import csv

import numpy as np 
from scipy.interpolate import splev, splrep


class FramerateManager():
    def __init__(self, project_args):

        project_args = json.load(open(project_args))
        raw_parameters = json.load(open(project_args["data_raw_parameters"]))

        self.old_framerate = raw_parameters["old_framerate"]
        self.new_framerate = raw_parameters["new_framerate"]
        self.scenes_list = raw_parameters["scenes"]


        self.destination_file = project_args["interim_data"] + "{}.csv"
        self.original_file = project_args["raw_dataset"] + "{}.csv"
        self.temp_file = project_args["interim_data"] + "temp.csv"

    def manage_framerate(self):
        print("Managing framerate")
        for scene in self.scenes_list:
            print("scene: {}".format(scene))
            self.change_rate(scene)

    def change_rate(self,scene_name):
               
        helpers.remove_file(self.destination_file.format(scene_name))
        helpers.remove_file(self.temp_file)

        helpers.extract_trajectories(self.original_file.format(scene_name),self.temp_file,save=True)

        with open(self.temp_file) as trajectories:
            with open(self.destination_file.format(scene_name),"a") as destination_file:
                csv_writer = csv.writer(destination_file)


                for k,trajectory in enumerate(trajectories):     
                                
                        trajectory = json.loads(trajectory)
                        coordinates = np.array(trajectory["coordinates"])
                        downsample_coordinates = self.__resample_trajectory(coordinates)
                        trajectory["coordinates"] = downsample_coordinates.tolist()
                        rows = self.__trajectory_to_rows(trajectory)
                        
                        for row in rows:
                            csv_writer.writerow(row)           
        helpers.remove_file(self.temp_file) 

    def __resample_trajectory(self,trajectory):
        rate_ratio = int(self.old_framerate/self.new_framerate)    
        #processing 
        x = trajectory[:,0]
        y = trajectory[:,1]

        nb_sample = len(x)
        t = np.array([1/self.old_framerate*i for i in range(nb_sample)])

        nb_sample1 = int(nb_sample/rate_ratio) + 1 ######
        t1 = np.array([1/self.new_framerate*i for i in range(nb_sample1)])

        sx = splrep(t, x, s = 0)
        sy = splrep(t, y, s = 0)

        x_down = splev(t1, sx)
        y_down = splev(t1, sy)

        down_sampled_trajectory = np.concatenate([x_down[:,np.newaxis],y_down[:,np.newaxis]],axis = 1)   
        return down_sampled_trajectory 


        

    def __trajectory_to_rows(self,trajectory):

        coordinates = trajectory["coordinates"]
        bboxes = trajectory["bboxes"]
        frames = trajectory["frames"]

        scene = trajectory["scene"]
        user_type = trajectory["user_type"]
        dataset = trajectory["dataset"]
        id_ = trajectory["id"]


        traj_len = len(coordinates)

        scenes = [scene for _ in range(traj_len)]
        user_types = [user_type for _ in range(traj_len)]
        datasets = [dataset for _ in range(traj_len)]
        ids = [id_ for _ in range(traj_len)]

        rows = []
        for d,s,f,i,c,b,t in zip(datasets,scenes,frames,ids,coordinates,bboxes,user_types):
            row = []
            row.append(d)
            row.append(s)
            row.append(f)
            row.append(i)
            for e in c:
                row.append(e)
            for e in b:
                row.append(e)
            row.append(t)
            rows.append(row)

        return rows

            
            
            

