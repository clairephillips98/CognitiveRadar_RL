"""""
Claire Phillips
ECE2500
Jan 18 2024

Defining a moving target
"""""
import random

import torch
from random import randint
import math


class Target:

    def __init__(self, radius, bounds, name=None, seed=None):
        self.bounds = self.bounds_expanded(bounds, 0.2)
        self.t = 0
        self.shift = 0
        self.x_start = randint(self.bounds['x_lower'] * 50, self.bounds['x_upper'] * 50) / 50
        self.y_start = randint(self.bounds['y_lower'] * 50, self.bounds['y_upper'] * 50) / 50
        self.x_vel = randint(-50, 50) / 100
        self.y_vel = randint(-50, 50) / 100
        self.x_ac = randint(-25, 25) / 400
        self.y_ac = randint(-25, 25) / 400
        self.bounds = bounds
        self.vel = None
        self.acc = None
        self.stats = torch.empty(0, 3)
        self.first_in_view = None
        self.first_viewed = None
        self.time_in_view = 0
        self.views = [None]
        self.sum_velocity = 0
        self.cartesian_coordinates = None
        self.update_t(0)
        self.radius = radius
        self.name = name


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
            self.x_start = randint(self.bounds['x_lower'] * 50, self.bounds['x_upper'] * 50) / 50
            self.y_start = randint(self.bounds['y_lower'] * 50, self.bounds['y_upper'] * 50) / 50
            self.x_vel = randint(-100, 100) / 100
            self.y_vel = randint(-100, 100) / 100
            self.x_ac = randint(-100, 100) / 200
            self.y_ac = randint(-100, 100) / 200
            if self.first_in_view is not None:
                stats = self.final_stats()
                self.stats = torch.vstack((self.stats, torch.tensor(stats))) # view_rate, average_velocity, time_til_first_view
            self.first_in_view = None
            self.first_viewed = None
            self.time_in_view = 0
            self.views = [None]
            self.sum_velocity = 0


    def update_t(self, t):
        self.cartesian_coordinates = (
            self.x_start + self.x_vel * (t - self.shift) + self.x_ac * (t - self.shift) ** 2,
            self.y_start + self.y_vel * (t - self.shift) + self.y_ac * (t - self.shift) ** 2)
        self.re_init(self.cartesian_coordinates, t)
        self.vel = (self.x_vel + 2 * self.x_ac * (t - self.shift),
                    self.y_vel + 2 * self.y_ac * (t - self.shift))

        self.acc = (2 * self.x_ac, 2 * self.y_ac)
        return self.cartesian_coordinates

    def collect_stats(self, t, viewed):
        #collect stats on the target after each step
        if self.first_in_view is None:
            self.first_in_view = t
            self.time_in_view=1
        else:
            self.time_in_view += 1
        if viewed is True:
            if (self.first_viewed is None):
                self.first_viewed = t

            self.views.append(t)
        self.sum_velocity += sum(map(lambda v: v ** 2, self.vel))

    def final_stats(self):
        average_velocity = self.sum_velocity / self.time_in_view
        time_til_first_view = self.first_viewed - self.first_in_view if self.first_viewed is not None else -1
        possible_observable_time = self.first_in_view - self.t
        view_rate = (len(self.views) - 1 )/possible_observable_time
        return view_rate, average_velocity, time_til_first_view