from classes.prepare_samples_hdf5 import PrepareSamplesHdf5

import sys
import json
import csv
import helpers
import h5py
import os
import time

def main():
   
    parameters_path = "./src/parameters/project.json"

    sampler = PrepareSamplesHdf5(parameters_path)
    sampler.extract_scenes_hdf5()

   

    
    

if __name__ == "__main__":
    main()