"""""
Claire Phillips
ECE2500
Jan 18 2024

Defining a radar
"""""
import random
from utils import in_wedge_cartesian,in_circle_cartesian

class Radar:

    def __init__(self, peak_power, duty_cycle,
                 pulsewidth, bandwidth, frequency,
                 pulse_repetition_rate, antenna_size, cartesian_coordinates, wavelength, radians_of_view,seed=None):
        random.seed(seed)
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
        self.num_states = 360/self.radians_of_view

    def update_viewing_angle(self, new_angle):
        self.viewing_angle= new_angle
        return

    def update_t(self, t, given_dir=None):
        # just incase I decide I want to have the radar move around
        self.t=t
        if given_dir != None:
            self.viewing_angle = (given_dir*360)/self.num_states
        else:
            self.viewing_angle = look_new_direction()
        return

    def max_distance_viewable(self):
        distance = self.peak_power
        return distance

    def visible_targets(self, targets, viewed_targets):
        for target in targets:
            if target.views[-1] != self.t:
                if in_circle_cartesian(target.cartesian_coordinates, self.cartesian_coordinates,self.max_distance):
                    if in_wedge_cartesian(target.cartesian_coordinates,self.cartesian_coordinates,self.max_distance,
                                      self.viewing_angle,self.viewing_angle+self.radians_of_view):
                        target.collect_stats(self.t, True)
                        viewed_targets.append(target)
                    else:
                        target.collect_stats(self.t,False)
        return viewed_targets

def look_new_direction(degrees = None):
    if degrees is None:
        degrees = random.uniform(0, 360)
    return degrees

