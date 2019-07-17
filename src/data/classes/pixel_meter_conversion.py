import helpers.helpers as helpers

import json
import os
import csv

class Pixel2Meters():
    def __init__(self,project_args,pixel2meters = True):

        project_args = json.load(open(project_args))
        raw_parameters = json.load(open(project_args["data_raw_parameters"]))

        self.pixel_meter_ratios = raw_parameters["pixel_meter_ratios"]
        self.scenes_list = raw_parameters["scenes"]


        self.temp_file = project_args["interim_data"] + "temp.csv"
        self.file = project_args["interim_data"] + "{}.csv"
        self.pixel2meters = pixel2meters

    def apply_conversions(self):
        print("Applying pixel to meters conversion")
        for scene in self.scenes_list:
            print("scene: {}".format(scene))
            conversion_ratio = self.pixel_meter_ratios[scene]
            if not self.pixel2meters:
                conversion_ratio = 1.0/conversion_ratio
            self.convert(scene,conversion_ratio)

    def convert(self,scene,ratio):
        helpers.remove_file(self.temp_file)
        os.rename(self.file.format(scene),self.temp_file)
        helpers.remove_file(self.file.format(scene))

        with open(self.file.format(scene),"a+") as data_csv:
            data_writer = csv.writer(data_csv)
            with open(self.temp_file) as scene_csv:
                data_reader = csv.reader(scene_csv)
                for row in data_reader:
                    new_row = row
                    new_coords = [ratio * float(row[i]) for i in range(4,10)]                    

                    for i,c in enumerate(new_coords):
                        new_row[i+4] = c 
                    data_writer.writerow(new_row)     

        helpers.remove_file(self.temp_file)
                    

