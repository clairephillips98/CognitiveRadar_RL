# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import radar
import target
import target_radar_relationship
from create_plots import plot_radar_and_targets,make_gif
from math import pi
from utils import merge_dictionaries

def go():
    # Use a breakpoint in the code line below to debug your script.
    radar_1 = radar.Radar(peak_power=2, duty_cycle=3,
                 pulsewidth=4, bandwidth=1, frequency=3,
                 pulse_repetition_rate=3, antenna_size=4, cartesian_coordinates=(0,0), wavelength=3, radians_of_view=pi/4)
    radar_2 = radar.Radar(peak_power=2, duty_cycle=3,
                 pulsewidth=4, bandwidth=1, frequency=3,
                 pulse_repetition_rate=3, antenna_size=4, cartesian_coordinates=(1,1), wavelength=3, radians_of_view=pi/6)
    radar_1.update_t(7)
    target_1 = target.Target(radius=3,path_eqn=path_1)
    target_2 = target.Target(radius=1, path_eqn=path_2)
    target_1.update_t(7)
    rad1_tar1 = target_radar_relationship.RadarTargetRelationShip(radar_1,target_1)
    radar_1.update_seen_list((target_1,target_2))
    rad2_tar2 = target_radar_relationship.RadarTargetRelationShip(radar_2,target_2)
    radar_2.update_seen_list((target_1,target_2))
    for x in range(100):
        rad1_tar1.update_t(x)
        rad2_tar2.update_t(x)
        radar_1.update_seen_list((target_1, target_2))
        radar_2.update_seen_list((target_1, target_2))
        plot_radar_and_targets((radar_1,radar_2),(target_1,target_2),merge_dictionaries(radar_1.seen_list,radar_2.seen_list),x)
    make_gif(100)

def path_1(t):
    return 0.12 * t + 1, -0.001 * t

def path_2(t):
    return -0.0003 * t + -0.2, 0.01 * t+0.1

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    go()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
