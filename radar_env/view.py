# this class makes the tensor views and updates them
# this can either be for a single radar or multiple radars

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
from utils import action_unpack, action_repack

device = torch.device(GPU_NAME if torch.cuda.is_available() else "cpu")
print(device)


class View:

    def __init__(self, radars, bounds, args, num):
        self.mask_val = args.outside_radar_value
        self.args = args
        self.num = num
        self.scale = self.args.scale
        self.bounds = bounds
        self.args = args
        self.blur_radius = self.args.blur_radius
        self.shape = [self.blur_radius * 3 +
                      ceil((self.bounds['y_upper'] - self.bounds['y_lower']) / self.scale),
                      self.blur_radius * 3 +
                      ceil((self.bounds['x_upper'] - self.bounds['x_lower']) / self.scale)]
        self.next_image = (torch.ones((self.shape[0], self.shape[1])) * self.mask_val).to(device)
        self.x = (torch.arange(self.shape[1], dtype=torch.float32).view(1, -1).repeat(self.shape[0], 1)).to(device)
        self.y = (torch.arange(self.shape[0], dtype=torch.float32).view(-1, 1).repeat(1, self.shape[1])).to(device)
        self.transform = T.GaussianBlur(kernel_size=(self.blur_radius * 2 + 1, self.blur_radius * 2 + 1),
                                        sigma=(self.args.blur_sigma, self.args.blur_sigma)).to(device)
        self.radars = radars if type(radars) == list else [radars]
        if args.search_outer_circle == 0:
            self.masks = [
                self.draw_shape(self.x.clone(), self.y.clone(), radar.cartesian_coordinates, radar.start_angle, radar.end_angle, radar.max_distance) for
                radar in self.radars]
        else:
            self.masks = [
                self.draw_shape(self.x.clone(), self.y.clone(), radar.cartesian_coordinates, 0, 360, radar.max_distance) for
                radar in self.radars]
        self.mask_image = reduce(lambda x, y: torch.logical_or(x, y), self.masks)
        self.last_tensor = None
        self.speed_layers = (torch.zeros((self.shape[0], self.shape[1]))).to(device)
        self.indiv_images = None
        self.pair_masks = None # action mask for pair of radars
        self.action_masks,self.pair_masks = self.get_mask()
        self.current_mask=None
        self.current_pair_mask = None

    def draw_shape(self, x, y, center, start_angle, end_angle, radius):
        # Compute distances from the center
        scaled_center = [(center[0] - self.bounds['x_lower']) / self.scale + self.blur_radius,
                         (center[1] - self.bounds['y_lower']) / self.scale + self.blur_radius]
        distances = torch.sqrt((x - scaled_center[0]) ** 2 + (y - scaled_center[1]) ** 2).to(device)
        # Compute angles from the center
        angles = torch.atan2(y - scaled_center[1], x - scaled_center[0]).to(device)
        angles = (angles * 180 / torch.tensor(pi)).int() % 360  # Convert angles to degrees
        # Create a binary mask for the pie slice
        scaled_radius = radius / self.scale
        if start_angle <= end_angle:
            mask = (distances <= scaled_radius) & (angles >= start_angle) & (angles < end_angle)
        else:
            mask = (distances <= scaled_radius) & ((angles >= start_angle) | (angles < end_angle))
        return mask

    def get_mask_function(self, a):
        if len(self.radars) == 2:
            a = action_unpack(a, self.args.action_size)
        else:
            a = [a]
        for i, a in enumerate(a):
            self.radars[i].update_t(0, a, False)

        masks = [self.draw_shape(self.x, self.y, radar.cartesian_coordinates,
                                                            radar.viewing_angle,
                                                            (radar.viewing_angle +
                                                            radar.radians_of_view) % 360,
                                                            radar.max_distance)
                 for radar in self.radars]
        mask = reduce(lambda x, y: torch.logical_or(x, y), masks)
        return mask, masks

    def get_mask(self):
        # transform = T.ToPILImage()
        # images = []
        total_masks = list(map(lambda a: self.get_mask_function(a)[0], range(self.args.action_size)))
        pair_masks = list(map(lambda a: self.get_mask_function(a)[1], range(self.args.action_size)))
        # for mask in total_masks:
        #    images.append(transform(torch.stack([mask.squeeze(0).float()] * 3, dim=0)))
        # images[0].save("./images/mask.gif", save_all=True, append_images=images)
        return total_masks,pair_masks

    def create_image(self, visible_targets):
        # create the tensor image of the world in black and white [0,1] where 1 is nothing and 0 is something
        # blur the last image
        # fill in the seen area as 1
        # fill in the seen targets as 0
        # anything that is outside of the area of view set to 0.5
        self.next_image = self.transform(torch.stack([self.next_image] * 3, dim=0))[0, :, :]  # add blur
        # Create a meshgrid of coordinates
        if len(self.radars)==2:
            action = action_repack([radar.given_dir for radar in self.radars], self.args.action_size)
        else:
            action = self.radars[0].given_dir
        self.current_mask = self.action_masks[action]
        self.current_pair_mask = self.pair_masks[action]
        self.next_image[self.current_mask] = 1
        if self.speed_layers is not None:
            self.speed_layers[self.current_mask] = 0
        for radar in visible_targets:
            if self.args.search_outer_circle == 0:
                for target in visible_targets[radar]:
                    mask = self.draw_shape(self.x.clone(), self.y.clone(), target.tensor_cart_coords(), 0, 360,
                                           max(self.scale / 2 + 1, target.radius))
                    self.next_image[mask] = 0
                    if self.speed_layers is not None:
                        observed_vel = abs(max(target.doppler_velocity.values(), key=abs))if min(target.doppler_velocity.values(), key=abs) == 0 else target.abs_vel
                        vel_mask = abs(observed_vel) > self.speed_layers.squeeze(0).abs()
                        self.speed_layers[mask & vel_mask] = observed_vel

        # add mask of original value to everything outside mask
        self.next_image[~self.mask_image] = self.mask_val
        return self.next_image

    def create_hidden_target_tensor(self, targets):
        # create overall view of the world. this is like create image but everything is included
        # this is for comparison purposes
        # we don't need to set a mask for outside the radar view, if were setting everything to 0.5 outside the view the
        # loss will be the same wheither there is a target(1) or no target(0).  given symmetry of loss
        # either way there is lower threshold of loss.
        world_view = torch.ones((self.shape[0], self.shape[1])).to(device)
        for target in targets:
            mask = self.draw_shape(self.x.clone(), self.y.clone(), target.cartesian_coordinates, 0, 360,
                                   max(self.scale / 2 + 1, target.radius))
            world_view[mask] = 0
        return world_view


    def indiv_lambda(self,next_image, mask):
        radar_im = next_image.clone()
        radar_im[~mask] = self.mask_val
        return radar_im

    def individual_radars(self):
        # make a masked version of each indivual radar
        # individual_radar_masked_images = torch.empty((0,) + self.next_image.shape, dtype=self.next_image.dtype)
        self.indiv_images = list(map(lambda mask: self.indiv_lambda(self.next_image, mask), self.masks))
        return self.indiv_images

    def indiv_action_masks(self, action):
        return self.pair_masks[action]

    def indiv_radar_as_state(self):
        resize = self.masks[0].size()[0]
        resized_states = list((self.indiv_images[0][:, :resize], self.indiv_images[1][:, -resize:]))
        return resized_states

    def set_last(self):
        self.last_tensor = self.next_image
