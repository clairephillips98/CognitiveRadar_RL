"""""
Claire Phillips
ECE2500
Jan 18 2024

Defining a radar
"""""
import random
random.seed(10)
from math import pi
from utils import is_angle_between,in_wedge_cartesian,cartesian_to_polar,relative_location

class Radar:

    def __init__(self, peak_power, duty_cycle,
                 pulsewidth, bandwidth, frequency,
                 pulse_repetition_rate, antenna_size, cartesian_coordinates, wavelength, radians_of_view):
        self.peak_power = peak_power
        self.duty_cycle = duty_cycle
        self.pulsewidth = pulsewidth
        self.bandwidth = bandwidth
        self.frequency = frequency
        self.pulse_repetition_rate = pulse_repetition_rate
        self.antenna_size = antenna_size
        self.cartesian_coordinates = cartesian_coordinates  # (x,y)
        self.wavelength = wavelength
        self.viewing_angle = look_new_direction()
        self.radians_of_view = radians_of_view
        self.max_distance = self.max_distance_viewable()
        self.t=0
        self.seen_list = {}
        self.num_states = (2*pi)/self.radians_of_view

    def update_viewing_angle(self, new_angle):
        self.viewing_angle= new_angle
        return

    def update_t(self, t, random_dir):
        # just incase I decide I want to have the radar move around
        self.t=t
        if random_dir:
            self.viewing_angle = random_dir*2*pi/self.num_states
        else:
            self.viewing_angle = look_new_direction()
        return

    def max_distance_viewable(self):
        distance = self.peak_power
        return distance

    def visible_targets(self, targets):
        targets = [target for target in targets if in_wedge_cartesian(target.cartesian_coordinates,
                                                                     self.cartesian_coordinates,
                                                                     self.max_distance, self.viewing_angle,
                                                                      self.viewing_angle+self.radians_of_view)]
        return targets

def look_new_direction(radians = None):
    if radians is None:
        radians = random.uniform(0, 2 * pi)
    return radians
