import numpy as np 
import matplotlib.cm as cm

def get_colors(nb_colors,nb_colors_per_map = 20,maps = [cm.tab20,cm.tab20b,cm.tab20c,cm.gist_rainbow,cm.gist_ncar] ):
    max_colors = len(maps) * nb_colors_per_map
    if nb_colors >= max_colors:
        return []

    nb_colors_per_map = int(nb_colors/len(maps)) + 1
    max_colors = len(maps) * nb_colors_per_map
    # print(nb_colors,nb_colors_per_map)
    # x = np.arange(nb_colors)

    colors =  np.concatenate([ map_( np.linspace(0, 1, nb_colors_per_map)) for map_ in maps ], axis = 0)

    ids = np.arange(max_colors)
    np.random.shuffle(ids)
    selected_ids = ids[:nb_colors]
    colors = colors[selected_ids]
    return colors
