"""
Claire Phillips
Jan. 26, 2024

Pulling together radar and target to create the environment, and define the reward.
"""

from radar_env.radar import Radar
from radar_env.target import Target
from utils import min_max_radar_breadth
from PIL import Image, ImageDraw, ImageFilter
from math import ceil, floor
import numpy as np
import torch
from functools import reduce
import math
from math import pi
import torchvision.transforms as T
from rl_agents.config import GPU_NAME
import argparse

device = torch.device(GPU_NAME if torch.cuda.is_available() else "cpu")
print(device)


def create_radars(seed=None):
    radar_1 = Radar(max_distance=184, duty_cycle=3,
                    pulsewidth=4, bandwidth=1, frequency=3,
                    pulse_repetition_rate=3, antenna_size=4, cartesian_coordinates=(0, 0), wavelength=3,
                    radians_of_view=45, seed=seed, radar_num=0)
    radar_2 = Radar(max_distance=184, duty_cycle=3,
                    pulsewidth=4, bandwidth=1, frequency=3,
                    pulse_repetition_rate=3, antenna_size=4, cartesian_coordinates=(150, 0), wavelength=3,
                    radians_of_view=45, seed=seed, radar_num=1)
    return [radar_1, radar_2]  # just 1 radar for now


def bounds(radar):
    # create bounds around radars so we have image size
    x_lower, y_lower, x_upper, y_upper = min_max_radar_breadth(radar)  # update to include both radars
    return {'x_lower': x_lower, 'x_upper': x_upper, 'y_lower': y_lower, 'y_upper': y_upper}


def overall_bounds(radars):
    # create bounds around radars so we have image size
    return {'x_lower': min(map(lambda d: d['x_lower'], radars)),
            'x_upper': max(map(lambda d: d['x_upper'], radars)),
            'y_lower': min(map(lambda d: d['y_lower'], radars)),
            'y_upper': max(map(lambda d: d['y_upper'], radars))}


def create_targets(n_ts, bounds, args, seed=None):
    targets = [Target(bounds=bounds, args=args, name=n, seed=seed) for n in range(n_ts)]
    return targets


class Simulation:
    meta_data = {'game_types': ['single_agent', 'MARL_shared_view', 'MARL_shared_targets']}

    def __init__(self, args, seed=None, game_type='single_agent'):
        self.args = args
        self.game_type = game_type
        self.reward = None
        self.t = 0
        self.speed_scale = self.args.speed_scale
        self.radars = create_radars(seed)
        if self.args.radars == 1: self.radars = self.radars[0]
        self.bounds = [bounds(radar) for radar in self.radars]
        self.overall_bounds = overall_bounds(self.bounds)  # these are overall bounds for when there are multiple radars
        self.targets = create_targets(15, self.overall_bounds, args, seed=seed)
        self.scale = self.args.scale
        self.blur_radius = self.args.blur_radius
        self.shape = [self.blur_radius * 3 +
                      ceil((self.overall_bounds['y_upper'] - self.overall_bounds['y_lower']) / self.scale),
                      self.blur_radius * 3 +
                      ceil((self.overall_bounds['x_upper'] - self.overall_bounds['x_lower']) / self.scale)]
        self.next_image = (torch.ones((self.shape[0], self.shape[1])) * 0.5).to(device)
        self.x = (torch.arange(self.shape[1], dtype=torch.float32).view(1, -1).repeat(self.shape[0], 1)).to(device)
        self.y = (torch.arange(self.shape[0], dtype=torch.float32).view(-1, 1).repeat(1, self.shape[1])).to(device)
        self.transform = T.GaussianBlur(kernel_size=(self.blur_radius * 2 + 1, self.blur_radius * 2 + 1),
                                        sigma=(self.args.blur_sigma, self.args.blur_sigma)).to(device)
        masks = [
            self.draw_shape(self.x.clone(), self.y.clone(), radar.cartesian_coordinates, 0, 360, radar.max_distance) for
            radar in self.radars]
        self.mask_image = reduce(lambda x, y: torch.logical_or(x, y), masks)
        self.images = []
        self.last_tensor = None
        self.speed_layers = (torch.zeros((self.shape[0], self.shape[1]))).to(device)
        self.initial_scan()

    def draw_shape(self, x, y, center, start_angle, end_angle, radius):
        # Compute distances from the center
        scaled_center = [(center[0] - self.overall_bounds['x_lower']) / self.scale + self.blur_radius,
                         (center[1] - self.overall_bounds['y_lower']) / self.scale + self.blur_radius]
        distances = torch.sqrt((x - scaled_center[0]) ** 2 + (y - scaled_center[1]) ** 2).to(device)
        # Compute angles from the center
        angles = torch.atan2(y - scaled_center[1], x - scaled_center[0]).to(device)
        angles = (angles * 180 / torch.tensor(pi)).int() % 360  # Convert angles to degrees
        # Create a binary mask for the pie slice
        scaled_radius = radius / self.scale
        if start_angle <= end_angle:
            mask = (distances <= scaled_radius) & (angles >= start_angle) & (angles <= end_angle)
        else:
            mask = (distances <= scaled_radius) & ((angles >= start_angle) | (angles <= end_angle))
        return mask

    def to_action(self, i):
        # can only be up to 2 actions
        if len(self.radars) == 1:
            return [i]
        elif len(self.radars) == 2:
            return [i % self.radars[0].num_states, floor(i / self.radars[0].num_states)]
        else:
            raise 'error with number of radars.  we cant handle more than 2 yet'

    def initial_scan(self):
        # initial scan - just look in every direction for the max number of looks required
        steps = max([radar.num_states for radar in self.radars])
        for step in range(int(ceil(steps))):
            step = [step] * len(self.radars)
            self.update_t(dir_list=step, recording=False, agent_learning=False)

    def update_t(self, dir_list=None, recording=False, agent_learning=True):
        # radar direction moves
        # targets move
        # compute the reward
        self.t += 1
        [rad.update_t(self.t, dir_list[i], (bool(self.args.relative_change) & agent_learning)) for i, rad in
         enumerate(self.radars)]  # i think this is acutally pointless
        [tar.update_t(self.t) for tar in self.targets]
        visible_targets = self.get_visible_targets_and_update_stats()
        if self.game_type == 'single_agent':
            self.step_for_single_view(visible_targets, recording)
        elif self.game_type == 'MARL_shared_view':
            self.step_for_single_view(visible_targets, recording)
        else:
            self.step_for_shared_targets(visible_targets, recording)

    def step_for_shared_targets(self, visible_targets, recording):
        # create view for each radar
        # 2 radars
        # 2 seperate views
        # reward for each radar

        return

    def step_for_single_view(self, visible_targets, recording):
        self.create_image(visible_targets)  # makes next image

        if recording:
            self.images.append(self.next_tensor)
        if self.last_tensor is not None:
            self.reward = self.reward_slice_cross_entropy(self.last_tensor, self.next_image)
        self.last_tensor = self.next_image

    def get_visible_targets_and_update_stats(self, radars=None):
        radars = self.radars if radars is None else radars  # if radars isnt specified use all radars
        visible_targets = {}  # make a list of visible targets
        for radar in radars:
            visible_targets[radar.radar_num] = (radar.visible_targets(self.targets))  # check which targets are visible
        return visible_targets

    def create_image(self, visible_targets):
        # create the tensor image of the world in black and white [0,1] where 1 is nothing and 0 is something
        # blur the last image
        # fill in the seen area as 1
        # fill in the seen targets as 0
        # anything that is outside of the area of view set to 0.5
        self.next_image = self.transform(torch.stack([self.next_image] * 3, dim=0))[0, :, :]  # add blur
        # Create a meshgrid of coordinates
        for radar in self.radars:
            start_angle = radar.viewing_angle % 360
            end_angle = (radar.viewing_angle + radar.radians_of_view) % 360
            mask = self.draw_shape(self.x.clone(), self.y.clone(), radar.cartesian_coordinates, start_angle, end_angle,
                                   radar.max_distance)
            # Convert mask to tensor and invert it
            self.next_image[mask] = 1
            if self.speed_layers is not None:
                self.speed_layers[mask] = 0
        for radar in visible_targets:
            for target in visible_targets[radar]:
                mask = self.draw_shape(self.x.clone(), self.y.clone(), target.cartesian_coordinates, 0, 360,
                                       max(self.scale / 2 + 1, target.radius))
                self.next_image[mask] = 0
                if self.speed_layers is not None:
                    radial_vel = max(target.doppler_velocity.values())
                    vel_mask = abs(radial_vel) > self.speed_layers.squeeze(0).abs()
                    self.speed_layers[mask & vel_mask] = radial_vel

        # add mask of original value to everything outside mask
        self.next_image[~self.mask_image] = 0.5
        return self.next_image

    def create_hidden_target_tensor(self):
        # create overall view of the world. this is like create image but everything is included
        # this is for comparison purposes
        # we don't need to set a mask for outside the radar view, if were setting everything to 0.5 outside the view the
        # loss will be the same wheither there is a target(1) or no target(0).  given symmetry of loss
        # either way there is lower threshold of loss.
        world_view = torch.ones((self.shape[0], self.shape[1])).to(device)
        for target in self.targets:
            mask = self.draw_shape(self.x.clone(), self.y.clone(), target.cartesian_coordinates, 0, 360,
                                   max(self.scale / 2 + 1, target.radius))
            world_view[mask] = 0
        return world_view

    def measure_world_loss(self, input, target):
        # make it so there is no loss in the areas we cannot see
        input[~self.mask_image] = 0
        target[~self.mask_image] = 0
        loss = torch.nn.BCELoss(reduction='mean').to(device)
        world_loss = loss(input=input, target=target)
        return world_loss

    def reward_slice_cross_entropy(self, last_tensor, next_image, add_mask=True, speed_scale=True):
        # Using BCE loss to find the binary cross entropy of the  changing images
        # The model is rewarded for a large change == high entropy
        loss = torch.nn.BCELoss(reduction='none').to(device)
        loss = loss(input=last_tensor, target=torch.floor(next_image))
        if add_mask:
            mask = ((next_image != 1) & (next_image != 0))
            # this works because if the last pixel was white, and it stayed white (or white), loss is 0
            # if the last pixel was grey, it will only now be white/black if the pixel has been viewed
            # so we only need to mask the grey cells
            loss[mask] = 0
        loss_og = loss
        if speed_scale:
            # scale the rewards so something with an absolute
            loss = torch.mul(self.speed_layers.abs() * self.speed_scale + 1, loss)
        reward = (torch.mean(loss))
        return reward


def main():
    parser = argparse.ArgumentParser("Hyperparameter Setting for DQN")
    parser.add_argument("--blur_radius", type=int, default=1,
                        help="size of the radius of the gaussian filter applied to previous views")
    parser.add_argument("--scale", type=int, default=23, help="factor by which the space is scaled down")
    parser.add_argument("--blur_sigma", type=float, default=0.5, help="guassian blur sigma")
    parser.add_argument("--common_destination", type=list, default=[-200, -200],
                        help="a common location for targets come from and go to")
    parser.add_argument("--cdl", type=float, default=0.0, help="how many targets go to location")
    parser.add_argument("--speed_layer", type=int, default=0, help="if speed is included in state space")
    parser.add_argument("--speed_scale", type=int, default=1,
                        help="how much the reward is scaled for seeing moving objects compared to not moving object")
    args = parser.parse_args()
    transform = T.ToPILImage()
    test = Simulation(args)
    images = []
    for t in range(20):
        test.update_t()
        images.append(transform(torch.stack([test.next_image] * 3, dim=0)))
    # images = [Image.fromarray(jnp.repeat(im,repeats = 3,axis=0)) for im in test.images]
    images[0].save("./images/cartesian.gif", save_all=True, append_images=images, duration=test.t, loop=0)
    # images = [Image.fromarray(jnp.repeat(im,repeats = 3,axis=0)) for im in test.polar_images]
    # images[0].save("./images/polar.gif", save_all=True, append_images=images, duration=test.t, loop=0)
    print('done')


if __name__ == "__main__":
    main()
