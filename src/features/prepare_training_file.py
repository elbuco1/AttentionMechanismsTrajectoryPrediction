import time
from classes.prepare_training import PrepareTraining
import json 
import sys 
import os

def main():
    
    parameters_path = "./src/parameters/project.json"

    prepare_training = PrepareTraining(parameters_path)
    prepare_training.create_training_file()

   
   

if __name__ == "__main__":
    main()




