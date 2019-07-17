import os
import helpers.helpers as helpers

import json
import csv


class DigitManager():
    def __init__(self,project_args):
        project_args = json.load(open(project_args))
        raw_parameters = json.load(open(project_args["data_raw_parameters"]))

        self.scenes_list = raw_parameters["scenes"]
        self.digit_number = raw_parameters["digit_number"]
        self.temp_file = project_args["interim_data"] + "temp.csv"
        self.file = project_args["interim_data"] + "{}.csv"

    def manage_digit_number(self):
        print("Managing number of digits")
        for scene in self.scenes_list:
            print("scene: {}".format(scene))
            self.change_digit_number(scene)

    def change_digit_number(self,scene_name):
        helpers.remove_file(self.temp_file)
        os.rename(self.file.format(scene_name),self.temp_file)
        helpers.remove_file(self.file.format(scene_name))

        with open(self.temp_file) as scene_csv:
            csv_reader = csv.reader(scene_csv)
            with open(self.file.format(scene_name),"a") as new_csv:
                csv_writer = csv.writer(new_csv)            
                for row in csv_reader:
                    new_row = row
                    for i in range(4,10):
                        new_row[i] = self.__round_coordinates(float(row[i]))
                    csv_writer.writerow(new_row)
        helpers.remove_file(self.temp_file)

        
    def __round_coordinates(self,point):
        point = int( point * 10**self.digit_number)/float(10**self.digit_number)
        return point  
