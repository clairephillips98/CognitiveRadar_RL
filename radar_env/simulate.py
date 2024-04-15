"""
Claire Phillips
Jan. 26, 2024

Pulling together radar and target to create the environment, and define the reward.
"""

from radar_env.radar import Radar
from radar_env.target import Target
from utils import min_max_radar_breadth
from math import ceil, floor
import torch
from functools import reduce
from math import pi
import torchvision.transforms as T
from rl_agents.config import GPU_NAME
import argparse
from radar_env.view import View
device = torch.device(GPU_NAME if torch.cuda.is_available() else "cpu")
print(device)

def create_radars(seed=None):
    radar_1 = Radar(max_distance=184, duty_cycle=3,
                    pulsewidth=4, bandwidth=1, frequency=3,
                    pulse_repetition_rate=3, antenna_size=4, cartesian_coordinates=(0, 0), wavelength=3,
                    radians_of_view=45, seed=seed, radar_num=0)
    radar_2 = Radar(max_distance=184, duty_cycle=3,
                    pulsewidth=4, bandwidth=1, frequency=3,
                    pulse_repetition_rate=3, antenna_size=4, cartesian_coordinates=(234, 0), wavelength=3,
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
        if self.args.radars == 1: self.radars = [self.radars[0]]
        self.bounds = [bounds(radar) for radar in self.radars]
        self.overall_bounds = overall_bounds(self.bounds)  # these are overall bounds for when there are multiple radars
        self.targets = create_targets(15, self.overall_bounds, args, seed=seed)
        if self.args.type_of_MARL in ['single_agent', 'MARL_shared_everything']:
            self.world_view = View(self.radars, self.overall_bounds, self.args, 0)
            self.diff_view = False
            self.diff_reward = False
        elif self.args.type_of_MARL in ['some_shared_info']:
            self.world_view = View(self.radars, self.overall_bounds, self.args)
            self.rewards = None
            self.diff_view = True
            self.diff_reward = True
        elif self.args.type_of_MARL in ['some_shared_info_shared_reward']:
            self.world_view = View(self.radars, self.overall_bounds, self.args)
            self.rewards = None
            self.diff_view = True
            self.diff_reward = True

        else: # no shared info, only shared target location
            self.views = [View(self.radars[x], self.bounds[x], self.args, x) for x in len(self.radars)]
            self.diff_view = True
            self.diff_reward = True
        self.individual_views=None
        self.initial_scan()

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

    def update_t(self, dir_list=None, recording=True, agent_learning=True):
        # radar direction moves
        # targets move
        # compute the reward
        self.t += 1
        [rad.update_t(self.t, dir_list[i], (bool(self.args.relative_change) & agent_learning)) for i, rad in
         enumerate(self.radars)]  # i think this is acutally pointless
        [tar.update_t(self.t) for tar in self.targets]
        visible_targets = self.get_visible_targets_and_update_stats(recording=recording)
        if self.args.type_of_MARL in ['single_agent', 'MARL_shared_everything']:
            self.step_for_single_view(visible_targets)
        elif self.args.type_of_MARL in ['some_shared_info', 'some_shared_info_shared_reward']:
            self.step_for_shared_view_diff_reward(visible_targets)
        else:
            self.step_for_shared_targets(visible_targets)

    def step_for_shared_view_diff_reward(self, visible_targets):
        # same view but masked version of the rewards
        # this should have the agents understand their actions better
        self.world_view.create_image(visible_targets)  # makes next image
        self.individual_views = self.world_view.individual_radars()
        if self.world_view.last_tensor is not None:
            self.reward = self.reward_slice_cross_entropy(self.world_view.last_tensor, self.world_view.next_image, self.world_view.speed_layers)

            self.rewards = list(map(lambda x: self.reward_slice_cross_entropy(self.world_view.last_tensor,x,
                                                                              self.world_view.speed_layers),
                                    self.individual_views))
        self.world_view.last_tensor = self.world_view.next_image

    def step_for_shared_targets(self, visible_targets):

        for i,view in enumerate(self.views):
            view.create_image(visible_targets)
            if view.last_tensor is not None:
                self.rewards[i] = self.reward_slice_cross_entropy(view.last_tensor, view.next_image,
                                                              view.speed_layers)
                view.last_tensor = view.next_image
        self.world_view.create_image(visible_targets)

        if self.world_view.last_tensor is not None:
            self.reward = self.reward_slice_cross_entropy(self.world_view.last_tensor, self.world_view.next_image, self.world_view.speed_layers)
        self.world_view.last_tensor = self.world_view.next_image
        self.individual_views = [view.next_image in self.views]

    def step_for_single_view(self, visible_targets):
        self.world_view.create_image(visible_targets)  # makes next image

        if self.world_view.last_tensor is not None:
            self.reward = self.reward_slice_cross_entropy(self.world_view.last_tensor, self.world_view.next_image, self.world_view.speed_layers)
        self.world_view.last_tensor = self.world_view.next_image

    def get_visible_targets_and_update_stats(self, radars=None, recording = True):
        radars = self.radars if radars is None else radars  # if radars isnt specified use all radars
        visible_targets = {}  # make a list of visible targets
        for radar in radars:
            visible_targets[radar.radar_num] = (radar.visible_targets(self.targets, recording))  # check which targets are visible
        return visible_targets

    def measure_world_loss(self, input, target):
        # make it so there is no loss in the areas we cannot see
        input[~self.world_view.mask_image] = 0
        target[~self.world_view.mask_image] = 0
        loss = torch.nn.BCELoss(reduction='mean').to(device)
        world_loss = loss(input=input, target=target)
        return world_loss

    def reward_slice_cross_entropy(self, last_tensor, next_image, speed_layers, add_mask=True, speed_scale=True, ):
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
            loss = torch.mul(speed_layers.abs() * self.speed_scale + 1, loss)
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
    parser.add_argument("--radars", type=int, default=2)
    parser.add_argument("--relative_change", type=int, default=0)
    args = parser.parse_args()
    transform = T.ToPILImage()
    test = Simulation(args)
    images = []
    for t in range(20):
        test.update_t([t%8,((-t)%8)])
        images.append(transform(torch.stack([test.next_image] * 3, dim=0)))
    # images = [Image.fromarray(jnp.repeat(im,repeats = 3,axis=0)) for im in test.images]
    images[0].save("./images/cartesian.gif", save_all=True, append_images=images, duration=test.t, loop=0)
    # images = [Image.fromarray(jnp.repeat(im,repeats = 3,axis=0)) for im in test.polar_images]
    # images[0].save("./images/polar.gif", save_all=True, append_images=images, duration=test.t, loop=0)
    print('done')


if __name__ == "__main__":
    main()
