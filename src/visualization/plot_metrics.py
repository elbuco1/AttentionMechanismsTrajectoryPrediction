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

    save_dir = parameters_project["metrics_reports"]


    

    # social loss
    social_0 = []
    social_5 = []
    social_1 = []

    for model in models_list:
        losses = json.load(open(dir_name.format(model)+"losses.json"))
        social_0.append(100*losses["global"]["social_joint_0.1"])
        social_5.append(100*losses["global"]["social_joint_0.5"])
        social_1.append(100*losses["global"]["social_joint_1.0"])

    
    
    socials = np.array([social_0,social_5,social_1])
    
    unit = ["0.1","0.5","1.0"]
    

    fig, ax = plt.subplots()
    
    for i in range( socials.shape[1]):
        ax.plot(unit,socials[:,i], label = models_list[i])
    # axs[0].plot(unit,socials[:,1])
    ax.set_title('Social metrics')
    ax.set(xlabel='distance threshold (m)', ylabel='conflict percentage')
    ax.legend()

    fig.tight_layout()
    plt.savefig(save_dir+"social_losses.png")
    plt.close()
    # ade fde 
    ade = []
    fde = []

    fig, ax = plt.subplots()

    for model in models_list:
        losses = json.load(open(dir_name.format(model)+"losses.json"))
        ade.append(losses["global"]["ade_disjoint"])
        fde.append(losses["global"]["fde_disjoint"])
    positions = np.array([ade,fde])


    for x,y,model in zip(positions[0,:],positions[1,:],models_list):
        ax.scatter(x,y,label = model)
    ax.set_title('Displacement metrics')
    ax.set(xlabel='Average Displacement Error (m)', ylabel='Final Displacement Error (m)')
    ax.legend()
    fig.tight_layout()
    plt.savefig(save_dir+"displacement_losses.png")
    plt.close()


    # dynamic losses
    fig, ax = plt.subplots()

    speeds = []
    accelerations = []
    for model in models_list:
        losses = json.load(open(dir_name.format(model)+"dynamic_losses.json"))
        speeds.append(losses["speed"]["global"])
        accelerations.append(losses["acceleration"]["global"])
    dynamics = np.array([speeds,accelerations])

    for x,y,model in zip(dynamics[0,:],dynamics[1,:],models_list):
        ax.scatter(x,y,label = model)
    ax.set_title('Dynamic metrics')
    ax.set(xlabel='Speed distributions distance', ylabel='Acceleration distributions distance')
    ax.legend()

    fig.tight_layout()
    plt.savefig(save_dir+"dynamic_losses.png")
    plt.close()

    
    fig, ax = plt.subplots()

    spatial = []
    models = []
    for model in models_list:
        losses = json.load(open(dir_name.format(model)+"losses.json"))
        spatial.append(losses["global"]["spatial_pred"]*100)
        models.append("")
    spatials = np.array([models,spatial])

    i = 0
    for x,y,model in zip(spatials[0,:],spatials[1,:],models_list):
        ax.bar(i,y,label = model)
        i+=1
    ax.set_title('Spatial metrics')
    ax.set(xlabel='Models', ylabel='Spatial conflicts proportion')
    ax.legend()

    fig.tight_layout()
    plt.savefig(save_dir+"spatial_losses.png")
    plt.close()




if __name__ == "__main__":
    main()