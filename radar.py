"""""
Claire Phillips
ECE2500
Jan 18 2024

Defining a radar
"""""
from random import uniform
from math import pi
from utils import is_angle_between,in_wedge_cartesian,cartesian_to_polar,relative_location
from target_radar_relationship import SeenObject

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

    def update_viewing_angle(self, new_angle):
        self.viewing_angle= new_angle
        return

    def update_t(self, t, random_dir: bool=True):
        # just incase I decide I want to have the radar move around
        self.t=t
        if random_dir:
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

    def update_seen_list(self,targets):
        targets = self.visible_targets(targets).copy()
        for radius,angle in self.seen_list.keys():
            if (radius < self.max_distance) & is_angle_between(angle, self.viewing_angle,self.viewing_angle+self.radians_of_view):
                if self.seen_list[(radius,angle)].cartesian_coordinates in [target.cartesian_coordinates for target in targets]:
                    self.seen_list[(radius, angle)].rechecked(self.t, True)
                    targets=[target for target in targets if target.cartesian_coordinates != self.seen_list[(radius,angle)].cartesian_coordinates]
                else:
                    self.seen_list[(radius, angle)].rechecked(self.t, False)
        for target in targets:
            self.seen_list[cartesian_to_polar(relative_location(target.cartesian_coordinates,self.cartesian_coordinates))]=SeenObject(target.cartesian_coordinates, self.t, self)

def look_new_direction(radians = None):
    if radians is None:
        radians = uniform(0, 2 * pi)
    return radians
