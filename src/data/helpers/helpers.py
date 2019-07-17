import os
import csv
import json
import scipy
import matplotlib.pyplot as plt
from scipy.interpolate import splev, splrep
import numpy as np
from scipy.spatial.distance import euclidean


# X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))

# X_scaled = X_std * (max - min) + min

def find_file_by_extension(root,ext):
    dir_list = os.listdir(root)
    file_list = [d for d in dir_list if os.path.isfile(d) ]
    ext_list = [d for d in file_list if d.split(".")[-1] == ext ]
    return ext_list



def min_max_scale(x,min_,max_,frange = (0,1)):
    x_std = (x - min_ )/(max_-min_)
    x_scaled = x_std * (frange[1]-frange[0]) + frange[0]
    return x_scaled

def revert_min_max_scale(x_scaled,min_,max_,frange = (0,1)):

    x_std = (x_scaled - frange[0]) / (frange[1]-frange[0]) 
    x = x_std * (max_-min_) + min_  
    return x


def get_speed(point1,point2,deltat):
    d = euclidean(point1,point2)
    v = d/deltat
    return v
def get_speeds(coordinates,framerate):
    speeds = []
    for i in range(1,len(coordinates)):
        speed = get_speed(coordinates[i-1],coordinates[i],framerate)
        speeds.append(speed)
    return speeds
def get_offsets(values):
    offsets = []
    for i in range(1,len(values)):
        offset = values[i] - values[i-1]
        offsets.append(offset)
    return offsets

def get_acceleration(v1,v2,deltat):
    a = (v2-v1)/deltat
    return a

def get_accelerations(speeds,framerate):
    accelerations = []
    for i in range(1,len(speeds)):
        acceleration = get_acceleration(speeds[i-1],speeds[i],framerate)
        accelerations.append(acceleration)
    return accelerations


"""
    Get the directories/files contained in a directory
    path: directory path
    lower: set the names to lower case
    ordered: order directory names lexygraphically
    descending: descending order for directory names
"""
def get_dir_names(path,lower = True,ordered = True,descending = False):
    dir_names = []
    dirs = os.listdir(path)
    if ordered:
        dirs = sorted(dirs,key = str, reverse = descending)
    for x in dirs:
        if lower:
            x = x.lower()
        dir_names.append(x)
    return dir_names

""" 
    check if file exists and delete it
"""
def remove_file(file):
    if os.path.exists(file):
        os.remove(file)


"""
    In dir dir_path, delete every file
    which name contains one of the strings
    in strings
"""
def del_files_containing_string(strings,dir_path):
    csv_files = get_dir_names(dir_path)
    for csv_ in csv_files:
        for string in strings:
            if string in csv_:
                file_ = dir_path + csv_
                remove_file(file_)


def clip_scene(clips,scene_path,new_path = "./data/temp/temp.csv"):
    x_low,x_up,y_low,y_up = clips
    with open(scene_path) as scene_csv:
        csv_reader = csv.reader(scene_csv)

        with open(new_path,"a") as new_csv:
            csv_writer = csv.writer(new_csv)

            for line in csv_reader:
                x = float(line[4])
                y = float(line[5])

                if x > x_low and x < x_up and y > y_low and y < y_up:
                    csv_writer.writerow(line)
    remove_file(scene_path)

    os.rename(new_path,scene_path)

    remove_file(new_path)

def augment_scene_list(scene_list,angles):
    new_list = []

    for scene in scene_list:
        new_list.append(scene)
        for angle in angles:
            scene_angle = scene + "_{}".format(angle)
            new_list.append(scene_angle)
    return new_list
    
# {
#     "coordinates" : [],
#     "bboxes" : [],
#     "frames" : [],
#     "scene" : line[1],
#     "user_type" : line[10],
#     "id" : id_,
#     "dataset" : line[0]
# }
# filename is 
def save_traj(trajectory):
    rows = []
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

def save_trajs(trajectories_temp,original_path,smooth_params,smooth = False):

    extract_trajectories(original_path,trajectories_temp,save=True,smooth = smooth,framerate = smooth_params["framerate"])
    if not smooth:
        remove_file(original_path)

    remove_file(smooth_params["destination_path"])
    with open(trajectories_temp) as trajectories:
        f = None

        if smooth:
            f = open(smooth_params["destination_path"],"a+")
        else:
            f = open(original_path,"a+")
        
        csv_writer = csv.writer(f)
        for k,trajectory in enumerate(trajectories):                
            trajectory = json.loads(trajectory)
            
            rows = save_traj(trajectory)
            for row in rows:
                csv_writer.writerow(row)


        f.close()
    remove_file(trajectories_temp)




"""
    Input: Standardized file_path
    Output a dictionnary of trajectories:
    {
        traj_id: {
            coordinates: [],
            bboxes: [],
            frames: [],
            scene:

        }
    }
"""
    

def extract_trajectories(file_name,destination_path = "", save = False,smooth = False,framerate = 1):

    trajectories = {}

    with open(file_name) as file_:
        file_ = csv.reader(file_, delimiter=',')
        for line in file_:

            id_ = int(line[3])
            # print(id_)
            coordinates = [float(line[4]),float(line[5])]
            bbox = [float(line[6]),float(line[7]),float(line[8]),float(line[9])]
            frame = int(line[2])

            if id_ not in trajectories:

                trajectories[id_] = {
                    "coordinates" : [],
                    "bboxes" : [],
                    "frames" : [],
                    "scene" : line[1],
                    "user_type" : line[10],
                    "id" : id_,
                    "dataset" : line[0]
                }
            trajectories[id_]["coordinates"].append(coordinates)
            trajectories[id_]["bboxes"].append(bbox)
            trajectories[id_]["frames"].append(frame)

    if smooth:
        err_tot = 0
        for i,id_ in enumerate(trajectories):
            trajectories[id_]["coordinates"],err = smooth_trajectory(trajectories[id_]["coordinates"],framerate)
            err_tot += err
        err_tot /= float(i)
        print("mean error {}".format(err_tot))
    if save:

        remove_file(destination_path)
        
        dict_frame = reindex_frames(file_name)

        with open(destination_path,"a") as scene_txt:
            for key in trajectories:
                
                new_frames = []
                for frame in trajectories[key]["frames"]:
                    new_frames.append(dict_frame[frame])
                trajectories[key]["frames"] = new_frames
                line = trajectories[key]
                # trajectories["id"] = key
                line = json.dumps(line)
                # print(line)
                # print("------")
                scene_txt.write(line + "\n" )
    else:
        return trajectories
    return

"""
    Input: Standardized file_path
    Output a dictionnary of frames:
    {
        frame: {
            object_id : {
                coordinates : [],
                bbox : []
            }

        }
    }
"""
# coordinates: traj_len,2
def smooth_trajectory(coordinates,framerate,k = 4,s = 0.0001):
   
    x = [e[0] for e in coordinates]
    y = [e[1] for e in coordinates]
    t = [framerate*i for i in range(len(coordinates))]

    # sx = scipy.interpolate.make_interp_spline(t,x,k)
    # sy = scipy.interpolate.make_interp_spline(t,y,k)

    try:
        sx = splrep(t, x, s = s)
        sy = splrep(t, y, s = s)
    except TypeError as e :
        print(e)
        return coordinates,0.
    


    # x_smooth = sx[t]
    # y_smooth = sy[t]

    x_smooth = splev(t, sx)
    y_smooth = splev(t, sy)

    err_x = np.mean([euclidean(a,b) for a,b in zip(x,sx)])
    err_y = np.mean([euclidean(a,b) for a,b in zip(y,sy)])

    


    # fig,axs = plt.subplots(2,1,squeeze = False,sharex=True,sharey=True)
    # axs[0][0].plot(x,y)
    # axs[1][0].plot(x_smooth,y_smooth)
    
    # plt.show()


    coordinates_smooth = [[x_s,y_s] for x_s,y_s in zip(x_smooth,y_smooth)]

    err = np.mean([euclidean(a,b) for a,b in zip(coordinates,coordinates_smooth)])
    
    return coordinates_smooth,err


def extract_frames(file_path,destination_path = "", save = False):
    frames = {}

    
    with open(file_path) as file_:
        csv_reader = csv.reader(file_)
        for line in csv_reader:
            # line = line.split(",")
            
            id_ = int(line[3])
            # print(id_)
            coordinates = [float(line[4]),float(line[5])]
            bbox = [float(line[6]),float(line[7]),float(line[8]),float(line[9])]
            frame = int(line[2])
            
            type_ = line[10]
            

            if frame not in frames:
                frames[frame] = {"ids":{}}
    
            frames[frame]["ids"][id_] = {
                "coordinates" : coordinates,
                "bbox" : bbox,
                "type" : type_,
                "scene" : line[1],
                "dataset" : line[0]

                }
            # if save:
            #     frames[frame]["frame"] = frame


        if save:

            remove_file(destination_path)
            

            # current_frame = 0
            # dict_frame = {}

            dict_frame = reindex_frames(file_path)

            with open(destination_path,"a") as scene_txt:
                for key in sorted(frames):

                    # if key not in dict_frame:
                    #     dict_frame[key] = current_frame
                    #     current_frame += 1

                    line = frames[key]
                    line["frame"] = dict_frame[key]
                    # line["frame"] = key
                    line = json.dumps(line)
                    # print(line)
                    # print("------")
                    scene_txt.write(line + "\n" )
        else:
            return frames
    return





def reindex_frames(file_path):
    frames = {}
    with open(file_path) as file_:
        csv_reader = csv.reader(file_)
        for line in csv_reader:
           
            frame = int(line[2])
           

            if frame not in frames:
                frames[frame] = -1
    
        current_frame = 0
        dict_frame = {}

        for key in sorted(frames):
            if key not in dict_frame:
                dict_frame[key] = current_frame
                current_frame += 1
    return dict_frame


"""
    Intersection over Union between two bounding boxes
    box = [xtopleft,ytopleft,xbottomright,ybottomright]
""" 

def bb_intersection_over_union(boxA, boxB):
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
 
	# compute the area of intersection rectangle
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
 
	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
 
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / float(boxAArea + boxBArea - interArea)
 
	# return the intersection over union value
	return iou