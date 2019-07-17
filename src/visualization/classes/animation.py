import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.cm as cm
import matplotlib.patches as mpatches



import json
import sys 
from helpers.helpers_visualisation import get_colors

from scipy.misc import imread
import matplotlib.image as mpimg

class Animation():
    def __init__(self,parameters_path):

        parameters_project = json.load(open(parameters_path))
        processed_parameters = json.load(open(parameters_project["data_processed_parameters"]))
        evaluation_parameters = json.load(open(parameters_project["evaluation_parameters"])) 
        raw_parameters = json.load(open(parameters_project["data_raw_parameters"]))



        visualization_parameters = json.load(open(parameters_project["visualization_parameters"]))

        self.scene = visualization_parameters["scene"]
        self.sample_id = visualization_parameters["sample_id"]


        self.pixel_meter_ratios = raw_parameters["pixel_meter_ratios"]
        self.meter2pixel_ratio = 1.0/ self.pixel_meter_ratios[self.scene]


        report_name = evaluation_parameters["report_name"]
          

        sub_dir_name = parameters_project["evaluation_reports"] + "{}/scene_reports/".format(report_name) 

        self.scene_samples = sub_dir_name + "{}_samples.json".format(self.scene)
        self.gif_name = parameters_project["figures_reports"] + "{}_{}.gif".format(self.scene,self.sample_id)


        self.image = parameters_project["raw_images"] + "{}.jpg".format(self.scene)
        self.rev_dict_types = processed_parameters["types_dic_rev"]
        


        

    def animate_sample(self):
        file_ = json.load(open(self.scene_samples))
        sample = file_[str(self.sample_id)]
        inputs = np.array(sample["inputs"])
        

        labels = np.array(sample["labels"])
        outputs = np.array(sample["outputs"])
        types = np.array(sample["types"])
        print(types)
        types = [ self.rev_dict_types[str(int(type_))] for type_ in types]



        img = mpimg.imread(self.image)

        prediction = np.concatenate([inputs,outputs], axis = 1)
        gt = np.concatenate([inputs,labels], axis = 1)

        prediction = prediction * self.meter2pixel_ratio
        gt = gt * self.meter2pixel_ratio


        nb_colors = gt.shape[0]

        colors = get_colors(nb_colors)

        animator = Animate(prediction,gt,colors,img,types,self.gif_name)
        animator.animate()

        

class Animate():
    def __init__(self,data_pred,data_gt,colors,img,types,gif_name = "test.gif", plot_ = False, save = True):

        self.img = img
        self.xs_pred = data_pred[:,:,0]
        self.ys_pred = data_pred[:,:,1]

        self.xs_gt = data_gt[:,:,0]
        self.ys_gt = data_gt[:,:,1]

        self.types = types 


        self.nb_agents = self.xs_pred.shape[0]
        self.margin = 1

        self.nb_frames = self.xs_pred.shape[1]
        self.gif_name = gif_name
        self.plot_ = plot_
        self.save = save

        self.fps = 1
        self.colors = colors

        self.lin_size = 100

        lin = np.linspace(0.6, 0.8, self.lin_size)
 
        self.color_dict = {
            "bicycle":cm.Blues(lin),
            "pedestrian":cm.Reds(lin),
            "car":cm.Greens(lin),
            "skate":cm.Greys(lin),
            "cart":cm.Purples(lin),
            "bus":cm.Oranges(lin)
        }

        self.colors = [self.color_dict[type_][np.random.randint(self.lin_size)] for type_ in self.types]

        self.history = 4

        self.get_plots()



    def get_plots(self):
        self.fig, self.ax = plt.subplots(1,2,squeeze= False)


        red_patch = mpatches.Patch(color='red', label='Pedestrians')
        blue_patch = mpatches.Patch(color='b', label='Bycicles')
        green_patch = mpatches.Patch(color='green', label='Cars')
        grey_patch = mpatches.Patch(color='grey', label='Skates')
        purple_patch = mpatches.Patch(color='purple', label='Carts')
        orange_patch = mpatches.Patch(color='orange', label='Buses')


        plt.legend(handles=[red_patch,blue_patch,green_patch,grey_patch,purple_patch,orange_patch],loc='best',fontsize = 3.5)
        self.ax[0][0].imshow(self.img,origin = "upper")
        self.ax[0][1].imshow(self.img,origin = "upper")

        self.plots1 = []
        self.plots2 = []


        for i in range(self.nb_agents):
            tup = self.ax[0][0].plot([], [], color = self.colors[i],marker = 'o',markersize = 2,linewidth = 0.5)[0]

            if i == 0:
                tup = self.ax[0][0].plot([], [], color = self.colors[i],marker = '^',markersize = 2,linewidth = 0.5)[0]            
                
            self.plots1.append(tup)

            tup = self.ax[0][1].plot([], [], color = self.colors[i],marker = 'o',markersize = 2,linewidth = 0.5)[0]

            if i == 0:
                tup = self.ax[0][1].plot([], [], color = self.colors[i],marker = '^',markersize = 2,linewidth = 0.5)[0]
            
            self.plots2.append(tup)
        
            

    def animate(self):
        self.ax[0][1].set_title("Groundtruth",loc = "left", fontsize=8)
        self.ax[0][0].set_title("Predictions",loc = "left", fontsize=8)

        plt.tight_layout()

        ani = matplotlib.animation.FuncAnimation(self.fig, self.update, frames=self.nb_frames,repeat=True)

        if self.plot_:
            plt.show()
        if self.save:
            ani.save(self.gif_name, writer='imagemagick', fps=self.fps,dpi = 200)



    def update(self,frame):
        frame = int(frame)
        end = frame + 1
        start = max(0,end-self.history)

        if end < 9:
            self.fig.suptitle("Timestep: {}, observation time".format(frame+1), fontsize=8)
        else:
            self.fig.suptitle("Timestep: {}, prediction time".format(frame+1), fontsize=8)
        
        for i,p in enumerate(self.plots1):
            
            xs = self.xs_pred[i]
            ys = self.ys_pred[i]

            c = 0
            for x,y in zip(xs,ys):
                if x == 0 and y == 0:
                    c += 1
                else:
                    break
            xs = xs[c:]
            ys = ys[c:]


            p.set_data(xs[start:end], ys[start:end])
            # p.set_color(self.colors[i])

            if frame > 7 :
                p.set_marker("+")
                p.set_markersize(3)

                # p.set_fillstyle("none")


        for i,p in enumerate(self.plots2):
            xs = self.xs_gt[i]
            ys = self.ys_gt[i]

            c = 0
            for x,y in zip(xs,ys):
                if x == 0 and y == 0:
                    c += 1
                else:
                    break
            xs = xs[c:]
            ys = ys[c:]


            p.set_data(xs[start:end], ys[start:end])
            # p.set_data(self.xs_gt[i,start:end], self.ys_gt[i,start:end])
            # p.set_color(self.colors[i])

            if frame > 7 :
                p.set_marker("+")
                p.set_markersize(3)

  


if __name__ == "__main__":
    main()