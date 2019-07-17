
import json
import sys 
from classes.animation import Animation



def main():
    parameters_path = "./src/parameters/project.json"
    
    animate = Animation(parameters_path)
    animate.animate_sample()


    


if __name__ == "__main__":
    main()