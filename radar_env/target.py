"""""
Claire Phillips
ECE2500
Jan 18 2024

Defining a moving target
"""""
import random

import torch
from random import randint, randrange
from rl_agents.config import GPU_NAME

device = torch.device(GPU_NAME if torch.cuda.is_available() else "cpu")
print(device)

time = 0.012

class Target:

    def __init__(self, radius, bounds, name=None, seed=None, common_destination=[0,0], common_destination_likelihood=0):
        self.bounds = self.bounds_expanded(bounds, 0.2)
        self.common_destination = self.point_in_square(common_destination)
        self.common_destination_likelihood = common_destination_likelihood
        self.t = 0
        self.shift = 0
        self.chance = randrange(0,100)/100  # chance take off or land location is random
        self.x_start, self.y_start = self.x_y_start()
        self.x_vel,self.y_vel = self.x_y_vel()
        self.x_ac,self.y_ac = 0,0
        self.bounds = bounds
        self.vel = None
        self.acc = None
        self.stats = torch.empty(0, 3).to(device)
        self.first_in_view = None
        self.first_viewed = None
        self.time_in_view = 0
        self.views = [None]
        self.sum_velocity = 0
        self.cartesian_coordinates = None
        self.update_t(0)
        self.radius = radius
        self.name = name

    def x_y_start(self):
        if self.chance < (self.common_destination_likelihood / 2):
            return self.common_destination
        else:
            x = randint(
                self.bounds['x_lower'] * 50, self.bounds['x_upper'] * 50) / 50
            y = randint(
            self.bounds['y_lower'] * 50, self.bounds['y_upper'] * 50) / 50
            return x,y

    def x_y_vel(self):
        if (self.common_destination_likelihood / 2)< self.chance < (self.common_destination_likelihood):
            x_vel = self.common_destination[0]-self.x_start
            y_vel = self.common_destination[1]-self.y_start
        else:
            x_vel = randint(-10,10)
            y_vel = randint(-10,10)
        scale = max((abs(x_vel) + abs(y_vel))/2,1)
        x_vel =( x_vel / scale ) * time
        y_vel =( y_vel / scale )* time
        return x_vel,y_vel


    def x_y_acc(self):
        # this is depreciated
        if (self.common_destination_likelihood / 2) < self.chance < (self.common_destination_likelihood):
            t = randint(100/time,300/time) #time to get to location
            x_displacement = self.common_destination[0] - (self.x_start + self.x_vel * t)
            y_displacement = self.common_destination[1] - (self.y_start + self.y_vel * t)
            abs_max = time * 25 / 400
            x_acc = max(-abs_max,min(2 * x_displacement / (t ** 2),time*25/400))
            y_acc = max(-abs_max,min(2 * y_displacement / (t ** 2),time*25/400))
        else:
            x_acc = (randint(-25, 25) / 400)* (time**2)
            y_acc = (randint(-25, 25) / 400)* (time**2)
        return x_acc, y_acc

    def point_in_square(self,point):
        x,y=point
        # Check if point is inside the square
        if self.bounds['x_lower'] <= x <= self.bounds['x_upper'] and self.bounds['y_lower'] <= y <= self.bounds['y_upper']:
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
            self.x_vel, self.y_vel = self.x_y_vel()
            self.x_ac,self.y_ac =0,0
            if self.first_in_view is not None:
                stats = self.final_stats()
                self.stats = torch.vstack((self.stats, torch.tensor(stats).to(device))) # view_rate, average_velocity, time_til_first_view
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
        time_til_first_view = (self.first_viewed - self.first_in_view) if self.first_viewed is not None else -1
        possible_observable_time = self.first_in_view - self.t
        view_rate = (len(self.views) - 1 )/possible_observable_time
        return view_rate, average_velocity, time_til_first_view

    def episode_end(self):
        if self.first_in_view is not None:
            stats = self.final_stats()
            self.stats = torch.vstack(
                (self.stats, torch.tensor(stats).to(device)))  # view_rate, average_velocity, time_til_first_view