import json
import sys 
import matplotlib.pyplot as plt
import numpy as np



def main():
    parameters_path = "./src/parameters/project.json"
    parameters_project = json.load(open(parameters_path))
    visualization_parameters = json.load(open(parameters_project["visualization_parameters"]))
    
    dir_name = parameters_project["evaluation_reports"] + "{}/"

    models_list = visualization_parameters["models"]
    metrics_list = visualization_parameters["metrics"]


    

    # social loss
    social_0 = []
    social_5 = []
    social_1 = []

    for model in models_list:
        print(dir_name.format(model))
        losses = json.load(open(dir_name.format(model)+"losses.json"))
        social_0.append(100*losses["global"]["social_joint_0.1"])
        social_5.append(100*losses["global"]["social_joint_0.5"])
        social_1.append(100*losses["global"]["social_joint_1.0"])

    
    
    socials = np.array([social_0,social_5,social_1])
    print(socials)
    unit = ["0.1","0.5","1.0"]
    print(socials)

    fig, axs = plt.subplots(3, 1)
    
    axs[0].plot(unit,socials[:,0])
    axs[0].plot(unit,socials[:,1])
    axs[0].set_title('Social losses')
    axs[0].set(xlabel='distance threshold (m)', ylabel='conflict percentage')

    # ade fde 
    ade = []
    fde = []

    for model in models_list:
        print(dir_name.format(model))
        losses = json.load(open(dir_name.format(model)+"losses.json"))
        ade.append(losses["global"]["ade_disjoint"])
        fde.append(losses["global"]["fde_disjoint"])
    positions = np.array([ade,fde])


    for x,y in zip(positions[0,:],positions[1,:]):
        print(x,y)
        axs[1].scatter(x,y)
    axs[1].set_title('Positionnal losses')
    axs[1].set(xlabel='Averade Displacement Error (m)', ylabel='Final Displacement Error (m)')

        
    
    
    
    
    
    
    
    
    
    
    
    fig.tight_layout()
    plt.show()

    # print(social_0)
    # print(social_5)
    # print(social_1)



    # fig, axs = plt.subplots(3, 1)




    # for ax,metric in zip(axs.flat,metrics_list):
    #     print(metric)



    # plt.show()
    # print(nb_rows)

    


    


if __name__ == "__main__":
    main()