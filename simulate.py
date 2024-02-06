"""
Claire Phillips
Jan. 26, 2024
"""

from radar import Radar, look_new_direction
from target import Target
from utils import min_max_radar_breadth, radians_to_degrees
from math import pi
from PIL import Image, ImageDraw, ImageFilter
from create_plots import make_gif
from math import ceil, pi, floor
import numpy as np
import torch


def create_radars():
    radar_1 = Radar(peak_power=400, duty_cycle=3,
                    pulsewidth=4, bandwidth=1, frequency=3,
                    pulse_repetition_rate=3, antenna_size=4, cartesian_coordinates=(0, 0), wavelength=3,
                    radians_of_view=pi / 4)
    radar_2 = Radar(peak_power=200, duty_cycle=3,
                    pulsewidth=4, bandwidth=1, frequency=3,
                    pulse_repetition_rate=3, antenna_size=4, cartesian_coordinates=(1, 1), wavelength=3,
                    radians_of_view=pi / 6)
    return [radar_1]  # , radar_2 just 1 radar for now


def bounds(radar):
    # create bounds around radars so we have image size
    x_lower, y_lower, x_upper, y_upper = min_max_radar_breadth(radar)  # update to include both radars
    return {'x_lower': x_lower, 'x_upper': x_upper, 'y_lower': y_lower, 'y_upper': y_upper, }


def create_targets(n_ts, bounds):
    targets = [Target(radius=1, bounds=bounds, path_eqn=None, name=n) for n in range(n_ts)]
    return targets


class Simulation:

    def __init__(self, blur_radius: int = 3, scale: int = 50):
        self.reward = None
        self.t = 0
        self.radars = create_radars()
        self.overall_bounds = bounds(self.radars[0])  # these are overall bounds for when there are multiple radars
        self.targets = create_targets(10, self.overall_bounds)
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
        self.next_tensor = None
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

    def initial_scan(self):
        # initial scan - just look in every direction for the max number of looks required

        steps = min([radar.num_states for radar in self.radars])
        integer_array = np.arange(0, steps)
        # np.random.shuffle(integer_array) we look in an order actually
        for step in integer_array:
            [rad.update_viewing_angle(step * rad.radians_of_view) for rad in self.radars]
            self.update_t(False)

    def update_t(self, dir_list=None, recording=False):
        # update direction randomly
        # targets move
        # compute the reward
        self.t = self.t + 1
        if dir_list is None:
            dir_list = [None] * len(self.radars)
        [rad.update_t(self.t, dir_list[i]) for i, rad in enumerate(self.radars)]  # i think this is acutally pointless
        [tar.update_t(self.t) for tar in self.targets]
        next_tensor = self.create_image()

        if recording == True:
            self.images.append(next_tensor)

        if self.next_tensor is not None:
            self.reward = self.reward_slice_cross_entropy(self.next_tensor, next_tensor)
        self.next_tensor = next_tensor

    def create_image(self):
        image = self.last_image.filter(ImageFilter.GaussianBlur(radius=self.blur_radius))
        new_image = ImageDraw.Draw(image)
        for radar in self.radars:
            x_min, y_min, x_max, y_max = min_max_radar_breadth(radar)
            new_image.pieslice((self.blur_radius, self.blur_radius, (x_max - x_min) / self.scale + self.blur_radius,
                                (y_max - y_min) / self.scale + self.blur_radius),
                               radians_to_degrees(radar.viewing_angle),
                               radians_to_degrees(radar.viewing_angle + radar.radians_of_view), fill='white')
            visible_targets = radar.visible_targets(self.targets)
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
        tensor_view = torch.tensor(np.array(self.last_image)[:, :, 0])
        return tensor_view

    def reward_slice_cross_entropy(self, last_image, next_image):

        # Using BCE loss to find the binary cross entropy of the the changing images
        # The model is rewarded for a large change == high entropy

        loss = torch.nn.BCELoss(reduction='none')
        loss = loss(last_image / 255, torch.floor(next_image / 255))
        mask = torch.where((next_image < 255) & (next_image > 0), 0.0, 1.0)
        # this works because if the last pixel was white, and it stayed white (or white), loss is 0
        # if the last pixel was grey, it will only now be white/black if the pixel has been viewed
        # so we only need to mask the grey cells
        reward = (torch.sum(torch.matmul(loss, mask)))
        print(reward)
        return reward


def main():
    test = Simulation(2, 50)
    for t in range(20):
        test.update_t()
    # images = [Image.fromarray(np.repeat(im,repeats = 3,axis=0)) for im in test.images]
    # images[0].save("./images/cartesian.gif", save_all=True, append_images=images, duration=test.t, loop=0)
    # images = [Image.fromarray(np.repeat(im,repeats = 3,axis=0)) for im in test.polar_images]
    # images[0].save("./images/polar.gif", save_all=True, append_images=images, duration=test.t, loop=0)
    print('done')


if __name__ == "__main__":
    main()
