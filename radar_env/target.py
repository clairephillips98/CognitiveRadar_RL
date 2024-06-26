"""""
Claire Phillips
ECE2500
Jan 18 2024

Defining a moving target
"""""
import random
import numpy as np
import torch
from random import randint, randrange
from rl_agents.config import GPU_NAME
from utils import cartesian_to_polar
from math import cos, sin, pi, log

device = torch.device(GPU_NAME if torch.cuda.is_available() else "cpu")
print(device)

dwell_time = 0.1  # seconds
speed = 100  # km /hour
speed_in_dt = speed / (360 / dwell_time)


class Target:

    def __init__(self, bounds, args, name=None, seed=None, id=None):
        self.scale = args.scale
        self.bounds = self.bounds_expanded(bounds,0)
        self.common_destination = self.point_in_square(args.common_destination)
        self.common_destination_likelihood = args.cdl
        self.t = 0
        self.shift = 0
        self.chance = random.random()  # chance take off or land location is random
        self.avg_rho = randrange(0,100) / 10000  # average radar cross section (this is 5m)
        self.manuever_t = randrange(1, 100)
        self.x_start, self.y_start = self.x_y_start()
        self.vel = self.x_y_vel()
        self.abs_vel = sum(map(lambda v: v ** 2, self.vel)) ** (1 / 2)
        self.acc = 0, 0
        self.stats = torch.empty(0, 6).to(device)
        self.first_in_view = None
        self.first_viewed = None
        self.time_observable = 0
        self.views = 0
        self.last_viewed = None
        self.sum_velocity = 0
        self.sum_doppler_velocity = 0
        self.cartesian_coordinates = None
        self.update_t(0)
        self.name = name
        self.target_angle = {}
        self.radius = self.avg_rho
        self.doppler_velocity = {}
        self.viewed_twice = 0
        self.views_r1 = [None]
        self.views_r2 = [None]
        self.observable_both_indc = None
        self.time_observable_both = 0

    def calculating_rho(self):
        # swerling I
        # pdf(rho) = (1/avg_rho)*e^(rho/avg_rho)
        # cdf(rho) =[integral(pdf(x)) from 0 to x ] = 1-e^(-rho/avg_rho)
        # rho = inverse_cdf(x) = -ln(1-x)*avg_rho (x is num between 0 and 1)
        x = random.random()
        rho = -log(1 - x) * self.avg_rho
        return rho

    def tensor_cart_coords(self):
        scaled_coords = (self.scale * int(self.cartesian_coordinates[0] / self.scale),
                         self.scale * int(self.cartesian_coordinates[1] / self.scale))
        return scaled_coords

    def x_y_start(self):
        if self.chance < (self.common_destination_likelihood / 2):
            return self.common_destination
        else:
            x = random.uniform(
                self.bounds['x_lower'] * 50, self.bounds['x_upper'] * 50) / 50
            y = random.uniform(
                self.bounds['y_lower'] * 50, self.bounds['y_upper'] * 50) / 50
            return x, y

    def x_y_vel(self):
        if (self.common_destination_likelihood / 2) < self.chance < (self.common_destination_likelihood):
            x_vel = self.common_destination[0] - self.x_start
            y_vel = self.common_destination[1] - self.y_start
            scale = (x_vel ** 2 + y_vel ** 2) ** (1 / 2)
            x_vel = x_vel / scale
            y_vel = y_vel / scale
        else:
            x_vel = randint(-10, 10)
            y_vel = randint(-10, 10)
        scale = (x_vel ** 2 + y_vel ** 2) ** (1 / 2)
        scale = scale if scale != 0 else 1
        self.vary = randint(50, 900)*self.avg_rho  # speed is between 50 km/hr to 900 km/hr
        x_vel = self.vary * (x_vel / scale) * speed_in_dt
        y_vel = self.vary * (y_vel / scale) * speed_in_dt
        return x_vel, y_vel

    def manuever(self):
        if self.t >= self.manuever_t:
            self.manuever_t = random.randrange(1+self.t+self.shift, 100+self.t+self.shift)
            new_angle = random.random()*2*pi
            self.shift = self.t
            self.x_start,self.y_start = self.cartesian_coordinates
            self.vel = (self.vary*cos(new_angle), self.vary*sin(new_angle)) #change direction not speed




    def point_in_square(self, point):
        x, y = point
        # Check if point is inside the square
        if self.bounds['x_lower'] <= x <= self.bounds['x_upper'] and self.bounds['y_lower'] <= y <= self.bounds[
            'y_upper']:
            return [x, y]  # Point is inside, return the original point
        else:
            # Find the closest point on the perimeter of the square to the given point
            closest_x = max(self.bounds['x_lower'], min(x, self.bounds['x_upper']))
            closest_y = max(self.bounds['y_lower'], min(y, self.bounds['y_upper']))

            # Determine which side of the square is closest to the point
            if x < self.bounds['x_lower']:
                closest_x = self.bounds['x_lower']
            elif x > self.bounds['x_upper']:
                closest_x = self.bounds['x_upper']

            if y < self.bounds['y_lower']:
                closest_y = self.bounds['y_lower']
            elif y > self.bounds['y_upper']:
                closest_y = self.bounds['y_upper']

            return [closest_x, closest_y]  # Point is outside, return closest point on perimeter

    def bounds_expanded(self, bounds: dict, expanded: int):
        # expand the bounds by x percent
        updated = {}
        updated['x_lower'] = bounds['x_lower'] - expanded * abs(bounds['x_lower'])
        updated['y_lower'] = bounds['y_lower'] - expanded * abs(bounds['y_lower'])
        updated['x_upper'] = bounds['x_upper'] + expanded * abs(bounds['x_upper'])
        updated['y_upper'] = bounds['y_upper'] + expanded * abs(bounds['y_upper'])
        return updated

    def re_init(self, pos, t):
        if not ((self.bounds['x_lower'] < pos[0] < self.bounds['x_upper']) & (
                self.bounds['y_lower'] < pos[1] < self.bounds['y_upper'])):
            self.shift = t
            self.x_start, self.y_start = self.x_y_start()
            self.vel = self.x_y_vel()
            self.abs_vel = sum(map(lambda v: v ** 2, self.vel)) ** (1 / 2)
            self.acc = 0, 0
            if self.first_in_view is not None:
                stats = self.final_stats(True)
                self.stats = torch.vstack((self.stats, torch.tensor(stats).to(
                    device)))  # view_rate, average_velocity, time_til_first_view, seen
            self.first_in_view = None
            self.first_viewed = None
            self.time_observable = 0
            self.views = 0
            self.last_viewed = None
            self.sum_velocity = 0
            self.sum_doppler_velocity = 0
            self.sum_doppler_velocity = 0
            self.time_observable = 0
            self.target_angle = {}
            self.doppler_velocity = {}
            self.viewed_twice = 0
            self.views_r1 = [None]
            self.views_r2 = [None]
            self.observable_both_indc = None
            self.time_observable_both = 0

    def calc_doppler_vel(self, radar_num):
        abs_vel, vel_angle = cartesian_to_polar(self.vel)
        self.doppler_velocity[radar_num] = abs_vel * cos(pi * (vel_angle - self.target_angle[radar_num]) / 180)

    def update_t(self, t):
        self.t = t
        self.cartesian_coordinates = (
            self.x_start + self.vel[0] * (t - self.shift) + self.acc[0] * (t - self.shift) ** 2,
            self.y_start + self.vel[1] * (t - self.shift) + self.acc[1] * (t - self.shift) ** 2)
        self.re_init(self.cartesian_coordinates, t)
        self.manuever()
        self.vel = (self.vel[0] + 2 * self.acc[0] * (t - self.shift),
                    self.vel[1] + 2 * self.acc[1] * (t - self.shift))

        self.acc = (self.acc[0], self.acc[1])

        return self.cartesian_coordinates

    def collect_stats(self, t, viewed, r=0):
        # collect stats on the target after each step
        if self.first_in_view is None:
            self.first_in_view = t
        if self.observable_both_indc != t:
            self.time_observable += 1
            self.sum_velocity += self.abs_vel
        if self.observable_both_indc == t:
            self.time_observable_both += 1
        self.observable_both_indc = t
        if viewed is True:
            if self.first_viewed is None:
                self.first_viewed = t
            if self.last_viewed == t:
                self.viewed_twice += 1
            self.views += 1
            self.last_viewed = t
        self.sum_doppler_velocity += self.doppler_velocity[r]  # observed doppler_velocity

    def final_stats(self, reinit=False):
        average_velocity = self.sum_velocity / self.time_observable
        average_doppler_velocity = self.sum_doppler_velocity / (self.time_observable * 2)
        if self.first_viewed is None:
            time_til_first_view = -1
            seen = 0
        else:
            time_til_first_view = (self.first_viewed - self.first_in_view)
            seen = 1
        view_rate = self.views / self.time_observable
        twice_view_rate = self.viewed_twice/ self.time_observable_both if self.time_observable_both else 0
        return view_rate, average_velocity, time_til_first_view, seen, average_doppler_velocity, twice_view_rate

    def episode_end(self):
        if self.first_in_view is not None:
            stats = self.final_stats()
            self.stats = torch.vstack(
                (self.stats, torch.tensor(stats).to(device)))  # view_rate, average_velocity, time_til_first_view
