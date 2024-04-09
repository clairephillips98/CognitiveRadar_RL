"""""
Claire Phillips
ECE2500
Jan 18 2024

Defining a radar and its capabilities
"""""
import random
from utils import in_wedge_cartesian,in_circle_cartesian,is_angle_between
import math
class Radar:

    def __init__(self, max_distance, duty_cycle,
                 pulsewidth, bandwidth, frequency,
                 pulse_repetition_rate, antenna_size, cartesian_coordinates, wavelength, radians_of_view,seed=None, radar_num=0):
        random.seed(seed)
        self.radar_num=radar_num
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
        self.max_distance = max_distance
        self.t=0
        self.seen_list = {}
        self.num_states = 360/self.radians_of_view
        self.rho_0 = 0.00032241 # rho_0 such that an object with a rho of 0.01km is seen 75 percent of the time at furtherest distance
        self.prob_f = 10e-4
        self.SNR_0 = 16

    def update_viewing_angle(self, new_angle):
        self.viewing_angle = new_angle
        return

    def update_t(self, t, given_dir=None, relative_change = False):
        # if relative_change = True then update the angle relative to the last position, otherwise change it to the absolute angel
        self.t=t
        if given_dir != None:
            if relative_change is True:
                self.viewing_angle = (self.viewing_angle +(given_dir*360)/self.num_states)%360
            else:
                self.viewing_angle = (given_dir * 360) / self.num_states
        else:
            self.viewing_angle = look_new_direction()
        return

    def object_detected(self, target_rho, target_distance):
        # calculate the likelihood of a target being seen and return a boolean of whether the target is detected
        snr_target = ((self.max_distance/target_distance)**4)*((target_rho/self.rho_0))*self.SNR_0
        prob_detection = self.prob_f**(1/(1+snr_target))
        detected = random.random() < prob_detection
        return detected

    def visible_targets(self, targets):
        viewed_targets= []
        for target in targets:
            in_circle, radius, angle = in_circle_cartesian(target.cartesian_coordinates,
                                   self.cartesian_coordinates,
                                   self.max_distance)
            if in_circle:
                target.target_angle[self.radar_num] = angle
                target.calc_doppler_vel(self.radar_num)
                if target.views[-1] != self.t:
                    in_wedge = is_angle_between(angle, self.viewing_angle, self.viewing_angle+self.radians_of_view)
                    target_rho = target.calculating_rho(self)
                    if in_wedge & self.object_detected(target_rho,radius):
                        target.collect_stats(self.t, True)
                        viewed_targets.append(target)
                    else:
                        target.collect_stats(self.t, False)
                else:
                    target.doppler_velocity[self.radar_num] = 0
        return viewed_targets

def look_new_direction(degrees = None):
    if degrees is None:
        degrees = random.uniform(0, 360)
    return degrees

