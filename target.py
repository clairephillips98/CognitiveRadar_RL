"""""
Claire Phillips
ECE2500
Jan 18 2024

Defining a moving target
"""""
import math
import utils
from random import randint


class Target:

    def __init__(self, radius,bounds, path_eqn=None, name=None):
        self.t = 0
        self.shift = 0
        self.x_start = randint(bounds['x_lower']*50, bounds['x_upper']*50) / 50
        self.y_start = randint(bounds['y_lower']*50, bounds['y_upper']*50) / 50
        self.x_vel = randint(-50, 50) / 100
        self.y_vel = randint(-50, 50) / 100
        self.x_ac = randint(-25, 25) / 400
        self.y_ac = randint(-25, 25) / 400
        self.bounds = bounds
        self.path_eqn = self.path if path_eqn is None else path_eqn
        self.radius = radius
        self.cartesian_coordinates = self.path_eqn(0)
        self.name = name


    def re_init(self, pos,t):
        if not((self.bounds['x_lower']<pos[0]<self.bounds['x_upper'])&(self.bounds['y_lower']<pos[1]<self.bounds['y_upper'])):
            self.shift = t
            self.x_start = randint(self.bounds['x_lower'] * 50, self.bounds['x_upper'] * 50) / 50
            self.y_start = randint(self.bounds['y_lower'] * 50, self.bounds['y_upper'] * 50) / 50
            self.x_vel = randint(-100, 100) / 100
            self.y_vel = randint(-100, 100) / 100
            self.x_ac = randint(-100, 100) / 200
            self.y_ac = randint(-100, 100) / 200


    def path(self, t):
        pos = (
        self.x_start + self.x_vel * (t-self.shift) + self.x_ac * (t-self.shift) ** 2, self.y_start + self.y_vel * (t-self.shift) + self.y_ac * (t-self.shift) ** 2)
        self.re_init(pos,t)
        return pos

    def update_t(self, t):
        self.cartesian_coordinates = self.path_eqn(t)




