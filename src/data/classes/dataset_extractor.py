import time
import csv
import helpers.helpers as helpers
import os
import sys
import json

class SddExtractor():
    def __init__(self, project_args):
        
        'Initializing parameters'
        project_args = json.load(open(project_args))
        data_parameters = json.load(open(project_args["data_external_parameters"]))

        self.scenes_list = data_parameters["used_scenes"]
        self.types_to_custom = data_parameters["types"]
        self.dataset_name = data_parameters["dataset_name"]

        self.external_dataset = project_args["external_dataset"]
        self.raw_dataset = project_args["raw_dataset"]

        self.scene_path = self.external_dataset + "{}/"
        self.subscene_path = self.external_dataset + "{}/{}/annotations.txt"
        self.subscene_image_path = self.external_dataset + "{}/{}/reference.jpg"

        self.destination_file = self.raw_dataset + "{}.csv"
        self.destination_image = project_args["raw_images"] + "{}.jpg"


    def extract(self):
        for scene in self.scenes_list:
            print("Extracting scene: " + scene)
            subscene_names = helpers.get_dir_names(self.scene_path.format(scene))           
            for i,subscene in enumerate(subscene_names):
                new_scene_name = scene + str(i)
                print("------subscene: " + new_scene_name)
                self.__extract_trajectories(scene,subscene,new_scene_name)
                self.__extract_images(scene,subscene,new_scene_name)


    def __extract_trajectories(self,scene,subscene,new_scene_name):                    
        subscene_path = self.subscene_path.format(scene,subscene)
        scene_csv = self.destination_file.format(new_scene_name)
        helpers.remove_file(scene_csv)

        with open(scene_csv,"a") as csv_scene:
            writer_scene = csv.writer(csv_scene)   
            with open(subscene_path) as subscene_csv:
                csv_reader = csv.reader(subscene_csv)                    
                for row in csv_reader:
                    new_row = self.__parse_row(row,new_scene_name)                            
                    writer_scene.writerow(new_row)


    def __extract_images(self,scene,subscene,new_scene_name):             
        subscene_image_path = self.subscene_image_path.format(scene,subscene)
        copy_cmd = "cp {} {}".format( subscene_image_path , self.destination_image.format(new_scene_name))
        os.system(copy_cmd)

    def __parse_row(self,row,scene):
        row = row[0].split(" ")
        new_row = []            

        bbox = [float(e) for e in row[1:5]]
        pos = self.__bbox_to_pos(bbox)

        new_row.append(self.dataset_name) # dataset label
        new_row.append(scene)   # subscene label
        new_row.append(row[5]) #frame
        new_row.append(row[0]) #id

        new_row.append(pos[0]) #x
        new_row.append(pos[1]) #y
        new_row.append(row[1]) # xmin. The top left x-coordinate of the bounding box.
        new_row.append(row[2]) # ymin The top left y-coordinate of the bounding box.
        new_row.append(row[3]) # xmax. The bottom right x-coordinate of the bounding box.
        new_row.append(row[4]) # ymax. The bottom right y-coordinate of the bounding box.

        agent_type = row[9].replace('"','').lower()
        new_row.append(self.types_to_custom[agent_type]) # label type of agent    

        return new_row
    
    def __bbox_to_pos(self,bbox):
        x = (bbox[0] + bbox[2])/2
        y = (bbox[1] + bbox[3])/2
        return (x,y)
