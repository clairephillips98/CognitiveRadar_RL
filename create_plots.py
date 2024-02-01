import numpy as np
from matplotlib.patches import Circle, Wedge, Polygon
import matplotlib.pyplot as plt
from utils import radians_to_degrees, min_max_radar_breadth
import os
import imageio


def plot_radar_and_targets(radars, targets, seenobjects,t):
    # Create an array of angles from 0 to 2*pi
    theta = np.linspace(0, 2 * np.pi, 100)

    # Create the plot
    fig, ax = plt.subplots(figsize=(6, 6))
    x_lower, x_upper, y_lower, y_upper = min_max_radar_breadth(radars[0])
    #plot the radars and where theyre looking
    for x, radar in enumerate(radars):
        wedge = Wedge(radar.cartesian_coordinates, radar.max_distance, radians_to_degrees(radar.viewing_angle),
                      radians_to_degrees(radar.viewing_angle + radar.radians_of_view))
        ax.add_artist(wedge)
        plt.plot(np.cos(theta)*radar.max_distance + radar.cartesian_coordinates[0],
                 np.sin(theta)*radar.max_distance + radar.cartesian_coordinates[1], label='radar' + str(x))
        if x >0:
            xl, xu, yl, yu = min_max_radar_breadth(radar)
            x_lower, x_upper = min(x_lower,xl),max(x_upper,xu)
            y_lower, y_upper = min(y_lower, yl), max(y_upper, yu)
    #plot the targets
    target_coords = list(zip(*[target.cartesian_coordinates for target in targets]))
    plt.scatter(target_coords[0], target_coords[1], marker='x', color='red')

    #plot the observations
    object_coords = list(zip(*[seenobjects[key].cartesian_coordinates for key in seenobjects if
                               seenobjects[key].object_there==True]))
    if len(object_coords) > 0:
        plt.scatter(object_coords[0], object_coords[1], marker='o',facecolors='none', edgecolors='blue')
    plt.xlim([x_lower, x_upper])
    plt.ylim([y_lower,y_upper])
    plt.savefig(f'./images/test_plot_{t}.png')

def make_gif(series):
    # Create a GIF from the plots
    images = []
    for x in series:
        images.append(imageio.imread(x))
        os.remove(x)
    imageio.mimsave('./images/my_gif.gif', images)
