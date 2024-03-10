"""
Claire Phillips
Jan. 26, 2024
"""

from radar_env.radar import Radar
from radar_env.target import Target
from utils import min_max_radar_breadth
from PIL import Image, ImageDraw, ImageFilter
from math import ceil, floor
import numpy as np
import torch


def create_radars(seed=None):
    radar_1 = Radar(peak_power=400, duty_cycle=3,
                    pulsewidth=4, bandwidth=1, frequency=3,
                    pulse_repetition_rate=3, antenna_size=4, cartesian_coordinates=(0, 0), wavelength=3,
                    radians_of_view=45,seed=seed)
    # radar_2 = Radar(peak_power=400, duty_cycle=3,
    #                 pulsewidth=4, bandwidth=1, frequency=3,
    #                 pulse_repetition_rate=3, antenna_size=4, cartesian_coordinates=(0, 400), wavelength=3,
    #                 radians_of_view=45,seed=seed)
    return [radar_1]  # , radar_2 just 1 radar for now


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


def create_targets(n_ts, bounds,seed=None):
    targets = [Target(radius=1, bounds=bounds, name=n,seed=seed) for n in range(n_ts)]
    return targets


class Simulation:
    meta_data = {'game_types': ['single_agent','MARL_shared_view', 'MARL_shared_targets']}
    def __init__(self, blur_radius: int = 3, scale: int = 50,seed=None, game_type='single_agent'):
        self.game_type = game_type
        self.reward = None
        self.t = 0
        self.radars = create_radars(seed)
        self.bounds = [bounds(radar) for radar in self.radars]
        self.overall_bounds = overall_bounds(self.bounds)  # these are overall bounds for when there are multiple radars
        self.targets = create_targets(10, self.overall_bounds, seed=seed)
        self.scale = scale
        self.blur_radius = blur_radius
        self.base_image = Image.new("RGB", (
            self.blur_radius * 3 + ceil((self.overall_bounds['x_upper'] - self.overall_bounds['x_lower']) / self.scale),
            self.blur_radius * 3 + ceil(
                (self.overall_bounds['y_upper'] - self.overall_bounds['y_lower']) / self.scale)), (150, 150, 150))
        self.last_image = self.base_image.copy()
        self.mask_image = self.create_mask().copy()
        self.images = []
        self.polar_images = []
        self.last_tensor = None
        self.initial_scan()

    def create_mask(self, angle_start=0, angle_stop=360, radars=None):
        radars = self.radars if radars is None else radars
        mask_im = Image.new("L", self.base_image.size, 0)
        draw = ImageDraw.Draw(mask_im)
        for radar in radars:
            draw.pieslice((radar.cartesian_coordinates[0]/self.scale+self.blur_radius,
                                radar.cartesian_coordinates[1]/self.scale+self.blur_radius,
                                (radar.cartesian_coordinates[0]+2*radar.max_distance) / self.scale + self.blur_radius,
                                (radar.cartesian_coordinates[1]+2*radar.max_distance) / self.scale + self.blur_radius),
                               angle_start,
                               angle_stop, fill='white')
        return mask_im
    def to_action(self, i):
        # can only be up to 2 actions
        if len(self.radars) == 1:
            return [i]
        elif len(self.radars)==2:
            return [i % self.radars[0].num_states, floor(i / self.radars[0].num_states)]
        else:
            raise 'error with number of radars.  we cant handle more than 2 yet'

    def initial_scan(self):
        # initial scan - just look in every direction for the max number of looks required
        steps = max([radar.num_states for radar in self.radars])
        for step in range(int(ceil(steps))):
            step =[step]*len(self.radars)
            self.update_t(dir_list = step, recording = False)

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
        if self.game_type=='single_agent':
            self.step_for_single_agent(visible_targets, recording)
        elif self.game_type=='MARL_shared_view':
            self.step_for_shared_view(visible_targets, recording)
        else:
            self.step_for_shared_targets(visible_targets, recording)

    def step_for_single_agent(self, visible_targets, recording):
        next_tensor = self.create_image(visible_targets)

        if recording:
            self.images.append(next_tensor)

        if self.last_tensor is not None:
            self.reward = self.reward_slice_cross_entropy(self.last_tensor, next_tensor)
        self.last_tensor = next_tensor

    def get_visible_targets_and_update_stats(self, radars=None):
        radars = self.radars if radars is None else radars
        visible_targets = []
        for radar in radars:
            visible_targets = radar.visible_targets(self.targets, visible_targets)
        return visible_targets

    def create_image(self, visible_targets):
        image = self.last_image.copy().filter(ImageFilter.GaussianBlur(radius=self.blur_radius))
        new_image = ImageDraw.Draw(image)
        for radar in self.radars:
            new_image.pieslice((radar.cartesian_coordinates[0]/self.scale+self.blur_radius,
                                radar.cartesian_coordinates[1]/self.scale+self.blur_radius,
                                (radar.cartesian_coordinates[0]+2*radar.max_distance) / self.scale + self.blur_radius,
                                (radar.cartesian_coordinates[1]+2*radar.max_distance) / self.scale + self.blur_radius),
                               radar.viewing_angle,
                               (radar.viewing_angle + radar.radians_of_view), fill='white')
        for target in visible_targets:
            new_image.pieslice(
                (floor((target.cartesian_coordinates[0] - target.radius - self.overall_bounds['x_lower']) / self.scale) + self.blur_radius,
                 floor((target.cartesian_coordinates[1] - target.radius - self.overall_bounds['y_lower']) / self.scale) + self.blur_radius,
                 ceil((target.cartesian_coordinates[0] + target.radius - self.overall_bounds['x_lower']) / self.scale) + self.blur_radius,
                 ceil((target.cartesian_coordinates[1] + target.radius - self.overall_bounds['y_lower']) / self.scale) + self.blur_radius
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
    # images = [Image.fromarray(jnp.repeat(im,repeats = 3,axis=0)) for im in test.images]
    images[0].save("./images/cartesian.gif", save_all=True, append_images=images, duration=test.t, loop=0)
    # images = [Image.fromarray(jnp.repeat(im,repeats = 3,axis=0)) for im in test.polar_images]
    # images[0].save("./images/polar.gif", save_all=True, append_images=images, duration=test.t, loop=0)
    print('done')


if __name__ == "__main__":
    main()

