"""
Claire Phillips
Jan. 26, 2024
"""

from radar_env.radar import Radar
from radar_env.target import Target
from utils import min_max_radar_breadth
from PIL import Image, ImageDraw, ImageFilter
from math import ceil, pi, floor
import numpy as np
import torch


def create_radars(seed=None):
    radar_1 = Radar(peak_power=400, duty_cycle=3,
                    pulsewidth=4, bandwidth=1, frequency=3,
                    pulse_repetition_rate=3, antenna_size=4, cartesian_coordinates=(0, 0), wavelength=3,
                    radians_of_view=45,seed=seed)
    radar_2 = Radar(peak_power=200, duty_cycle=3,
                    pulsewidth=4, bandwidth=1, frequency=3,
                    pulse_repetition_rate=3, antenna_size=4, cartesian_coordinates=(1, 1), wavelength=3,
                    radians_of_view=360 / 12,seed=seed)
    return [radar_1]  # , radar_2 just 1 radar for now


def bounds(radar):
    # create bounds around radars so we have image size
    x_lower, y_lower, x_upper, y_upper = min_max_radar_breadth(radar)  # update to include both radars
    return {'x_lower': x_lower, 'x_upper': x_upper, 'y_lower': y_lower, 'y_upper': y_upper, }


def create_targets(n_ts, bounds,seed=None):
    targets = [Target(radius=1, bounds=bounds, name=n,seed=seed) for n in range(n_ts)]
    return targets


class Simulation:

    def __init__(self, blur_radius: int = 3, scale: int = 50,seed=None):
        self.reward = None
        self.t = 0
        self.radars = create_radars(seed)
        self.overall_bounds = bounds(self.radars[0])  # these are overall bounds for when there are multiple radars
        self.targets = create_targets(10, self.overall_bounds, seed=seed)
        self.scale = scale
        self.blur_radius = blur_radius
        self.base_image = Image.new("RGB", (
            self.blur_radius * 3 + ceil((self.overall_bounds['x_upper'] - self.overall_bounds['x_lower']) / self.scale),
            self.blur_radius * 3 + ceil(
                (self.overall_bounds['y_upper'] - self.overall_bounds['y_lower']) / self.scale)), (150, 150, 150))
        self.last_image = self.base_image.copy()
        self.mask_image = self.create_mask()
        self.images = []
        self.polar_images = []
        self.last_tensor = None
        self.initial_scan()

    def create_mask(self, angle_start=0, angle_stop=360):
        mask_im = Image.new("L", self.base_image.size, 0)
        draw = ImageDraw.Draw(mask_im)
        for radar in self.radars:
            draw.ellipse((140, 50, 260, 170), fill=255)
            x_min, y_min, x_max, y_max = min_max_radar_breadth(radar)
            draw.pieslice((self.blur_radius, self.blur_radius, (x_max - x_min) / self.scale + self.blur_radius,
                           (y_max - y_min) / self.scale + self.blur_radius),
                          angle_start, angle_stop, fill=255)
        return mask_im
    def to_action(self, i):
        # can only be up to 2 actions
        if len(self.radars) == 1:
            return [i]
        else:
            return [i % self.radars[0].num_states, floor(i / self.radars[0].num_states)]

    def initial_scan(self):
        # initial scan - just look in every direction for the max number of looks required
        steps = min([radar.num_states for radar in self.radars])
        integer_array = np.arange(0, steps)
        for step in integer_array:
            self.update_t(dir_list = self.to_action(step),recording = False)

    def update_t(self, dir_list=None, recording=False):
        #radar direction moves
        # targets move
        # compute the reward
        self.t += 1
        if dir_list is None:
            dir_list = [None] * len(self.radars)
        [rad.update_t(self.t, dir_list[i]) for i, rad in enumerate(self.radars)]  # i think this is acutally pointless
        [tar.update_t(self.t) for tar in self.targets]
        visible_targets = self.get_visible_targets_and_update_stats()
        next_tensor = self.create_image(visible_targets)

        if recording == True:
            self.images.append(next_tensor)

        if self.last_tensor is not None:
            self.reward = self.reward_slice_cross_entropy(self.last_tensor, next_tensor)
        self.last_tensor = next_tensor

    def get_visible_targets_and_update_stats(self):
        visible_targets = []
        for radar in self.radars:
            visible_targets = radar.visible_targets(self.targets, visible_targets)
        return visible_targets

    def create_image(self, visible_targets):
        image = self.last_image.copy().filter(ImageFilter.GaussianBlur(radius=self.blur_radius))
        new_image = ImageDraw.Draw(image)
        for radar in self.radars:
            x_min, y_min, x_max, y_max = min_max_radar_breadth(radar)
            new_image.pieslice((self.blur_radius, self.blur_radius, (x_max - x_min) / self.scale + self.blur_radius,
                                (y_max - y_min) / self.scale + self.blur_radius),
                               (radar.viewing_angle),
                               (radar.viewing_angle + radar.radians_of_view), fill='white')
        for target in visible_targets:
            new_image.pieslice(
                (floor((target.cartesian_coordinates[0] - target.radius - x_min) / self.scale) + self.blur_radius,
                 floor((target.cartesian_coordinates[1] - target.radius - y_min) / self.scale) + self.blur_radius,
                 ceil((target.cartesian_coordinates[0] + target.radius - x_min) / self.scale) + self.blur_radius,
                 ceil((target.cartesian_coordinates[1] + target.radius - y_min) / self.scale) + self.blur_radius
                 ), 0,
                360, fill='black')

        self.last_image = self.base_image.copy()
        self.last_image.paste(image, (0, 0), self.mask_image)
        tensor_view = torch.tensor(np.array(self.last_image)[:, :, 0])/255
        return tensor_view

    def reward_slice_cross_entropy(self, last_image, next_image):

        # Using BCE loss to find the binary cross entropy of the the changing images
        # The model is rewarded for a large change == high entropy

        loss = torch.nn.BCELoss(reduction='none')
        loss = loss(last_image, torch.floor(next_image))
        mask = ((next_image<1)&(next_image>0)).nonzero()
        # scaling values to be between 0 and 1.  torch BCELoss values are clipped to be between 0 and 100.
        # this works because if the last pixel was white, and it stayed white (or white), loss is 0
        # if the last pixel was grey, it will only now be white/black if the pixel has been viewed
        # so we only need to mask the grey cells
        loss[mask[:, 0], mask[:, 1]] = 0
        reward = (torch.sum(loss))

        return reward


def main():
    test = Simulation(2, 25)
    images = []
    for t in range(20):
        test.update_t()
        images.append(test.last_image)
    # images = [Image.fromarray(np.repeat(im,repeats = 3,axis=0)) for im in test.images]
    images[0].save("./images/cartesian.gif", save_all=True, append_images=images, duration=test.t, loop=0)
    # images = [Image.fromarray(np.repeat(im,repeats = 3,axis=0)) for im in test.polar_images]
    # images[0].save("./images/polar.gif", save_all=True, append_images=images, duration=test.t, loop=0)
    print('done')


if __name__ == "__main__":
    main()

