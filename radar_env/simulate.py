"""
Claire Phillips
Jan. 26, 2024

Pulling together radar and target to create the environment, and define the reward.
"""
import random
from radar_env.radar import Radar
from radar_env.target import Target
from utils import min_max_radar_breadth
from math import ceil, floor
import torch
from functools import reduce
from math import pi, inf
import torchvision.transforms as T
from rl_agents.config import GPU_NAME
import argparse
from radar_env.view import View
device = torch.device(GPU_NAME if torch.cuda.is_available() else "cpu")
print(device)

transform = T.ToPILImage()
images = []
masks = []
states=[]
lasts = []
def create_radars(seed=None):
    radar_1 = Radar(max_distance=184, duty_cycle=3,
                    pulsewidth=4, bandwidth=1, frequency=3,
                    pulse_repetition_rate=3, antenna_size=4, cartesian_coordinates=(0, 0), wavelength=3,
                    radians_of_view=6, seed=seed, radar_num=0, start_angle = 315, end_angle = 45)
    radar_2 = Radar(max_distance=184, duty_cycle=3,
                    pulsewidth=4, bandwidth=1, frequency=3,
                    pulse_repetition_rate=3, antenna_size=4, cartesian_coordinates=(92, 130.11), wavelength=3,
                    radians_of_view=6, seed=seed, radar_num=1, start_angle = 225, end_angle = 315)
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
    meta_data = {'game_types': ['single_agent', 'MARL_shared_everything',
                                'some_shared_info', 'some_shared_info_shared_reward', 'shared_targets_only']}

    def __init__(self, args, seed=None, game_type='single_agent'):
        self.seed=seed
        self.prob_of_target = 0.01
        self.args = args
        self.game_type = game_type
        self.reward = None
        self.t = 0
        self.speed_scale = self.args.speed_scale
        self.radars = create_radars(seed)
        self.rewards = None
        if self.args.radars == 1: self.radars = [self.radars[0]]
        self.bounds = [bounds(radar) for radar in self.radars]
        self.args.action_size = int(reduce(lambda x, y: x * y, [radar.num_states for radar in self.radars]))
        self.overall_bounds = overall_bounds(self.bounds)  # these are overall bounds for when there are multiple radars
        self.demised_targets = []
        self.targets = create_targets(5, self.overall_bounds, args, seed=seed)
        self.world_view = View(self.radars, self.overall_bounds, self.args, 0)
        if self.args.type_of_MARL in ['single_agent', 'MARL_shared_everything']:
            self.diff_view = False
            self.diff_reward = False
        elif self.args.type_of_MARL in ['some_shared_info']:
            self.views = [View(radars=self.radars[x], bounds=self.bounds[x], args=self.args, num=x) for x in range(len(self.radars))]
            self.diff_view = True
            self.diff_reward = True
        elif self.args.type_of_MARL in ['some_shared_info_shared_reward']:
            self.diff_view = True
            self.diff_reward = False
        elif self.args.type_of_MARL in ['shared_targets_only']: # no shared info, only shared target location radars, bounds, args, num
            self.views = [View(radars=self.radars[x], bounds=self.bounds[x], args=self.args, num=x) for x in range(len(self.radars))]
            self.diff_view = True
            self.diff_reward = True
        self.individual_views = None
        self.individual_states = None
        self.initial_scan()

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
        if self.args.tracking_mode==0:
            self.add_remove_targets()
        visible_targets = self.get_visible_targets_and_update_stats(recording=recording)
        if self.args.type_of_MARL in ['single_agent', 'MARL_shared_everything']:
            self.step_for_single_view(visible_targets)
        elif self.args.type_of_MARL in ['some_shared_info', 'some_shared_info_shared_reward']:
            self.step_for_diff_world_view(visible_targets)
        elif self.args.type_of_MARL in ['shared_targets_only']:
            self.step_for_shared_targets(visible_targets)

    def add_remove_targets(self):
        prob_new = random.random()
        prob_remove = random.random()
        if prob_new < self.prob_of_target:# add a new_target
            new_target = Target(bounds=self.overall_bounds, args=self.args, name=len(self.targets), seed=self.seed)
            new_target.re_init([inf,inf],self.t)
            self.targets.append(new_target)
        if prob_remove < self.prob_of_target:# remove a target
            if self.targets!= []:
                removed_target = self.targets.pop(random.randrange(len(self.targets)))
                removed_target.episode_end()
                self.demised_targets.append(removed_target)



    def step_for_diff_world_view(self, visible_targets):
        # same view but masked version of the rewards
        # this should have the agents understand their actions better
        self.world_view.create_image(visible_targets)  # makes next image
        self.individual_views = self.world_view.individual_radars()
        if self.world_view.last_tensor is not None:
            self.reward = self.reward_slice_cross_entropy(self.world_view.last_tensor,
                                                          self.world_view.next_image,
                                                          self.world_view.speed_layers,
                                                          self.world_view.current_mask)
            if self.diff_reward:
                self.rewards = list(map(lambda i: self.reward_slice_cross_entropy(self.world_view.last_tensor,
                                                                                  self.individual_views[i],
                                                                                  self.world_view.speed_layers,
                                                                                  self.world_view.current_pair_mask[i]
                                                                                  ),
                                    range(len(self.individual_views))))
            else:
                self.rewards = None
        self.world_view.last_tensor = self.world_view.next_image
        self.individual_states = self.world_view.indiv_radar_as_state()

    def step_for_shared_targets(self, visible_targets):
        self.world_view.create_image(visible_targets)
        list(map(lambda x: self.views[x].create_image(visible_targets), range(len(self.views))))
        if self.world_view.last_tensor is not None:
            self.reward = self.reward_slice_cross_entropy(self.world_view.last_tensor, self.world_view.next_image,
                                                          self.world_view.speed_layers, self.world_view.current_mask)
            self.rewards = list(map(lambda x: self.reward_slice_cross_entropy(self.views[x].last_tensor,
                                                                              self.views[x].next_image,
                                                              self.views[x].speed_layers,
                                                                              self.views[x].current_mask),
                                    range(len(self.views))))
        for view in self.views:
            view.set_last()
        list(map(lambda x: self.views[x].set_last(), range(len(self.views))))
        self.world_view.last_tensor = self.world_view.next_image
        self.individual_views = [view.next_image for view in self.views]
        self.individual_states = self.individual_views


    def step_for_single_view(self, visible_targets):
        self.world_view.create_image(visible_targets)  # makes next image

        if self.world_view.last_tensor is not None:
            self.reward = self.reward_slice_cross_entropy(self.world_view.last_tensor, self.world_view.next_image,
                                                          self.world_view.speed_layers, self.world_view.current_mask)
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

    def reward_slice_cross_entropy(self, last_tensor, next_image, speed_layers, action_mask = None, add_mask=True, speed_scale=True):
        # Using BCE loss to find the binary cross entropy of the  changing images
        # The model is rewarded for a large change == high entropy
        loss = torch.nn.BCELoss(reduction='none').to(device)
        next_image[next_image > 1] = 1
        loss = loss(input=last_tensor, target=torch.floor(next_image))
        # lasts.append(transform(torch.stack([last_tensor] * 3, dim=0)))
        # states.append(transform(torch.stack([torch.floor(next_image)] * 3, dim=0)))
        if add_mask:
            mask = action_mask
            # this works because if the last pixel was white, and it stayed white (or white), loss is 0
            # if the last pixel was grey, it will only now be white/black if the pixel has been viewed
            # so we only need to mask the grey cells
            loss[~mask] = 0
            # masks.append(transform(torch.stack([mask.float()] * 3, dim=0)))
        if speed_scale:
            # scale the rewards so something with an absolute
            loss = torch.mul(speed_layers.abs() * self.speed_scale + 1, loss)
        # images.append(transform(torch.stack([loss.float()] * 3, dim=0)))
        # if self.t == 20:
        #     lasts[0].save("./images/lasts.gif", save_all=True, append_images=lasts)
        #     states[0].save("./images/states.gif", save_all=True, append_images=states)
        #     masks[0].save("./images/mask.gif", save_all=True, append_images=masks)
        #     images[0].save("./images/losses.gif", save_all=True, append_images=images)
        #     exit()
        reward = (torch.mean(loss))
        return reward


def main():
    parser = argparse.ArgumentParser("Hyperparameter Setting for DQN")
    parser.add_argument("--blur_radius", type=int, default=1,
                        help="size of the radius of the gaussian filter applied to previous views")
    parser.add_argument("--scale", type=int, default=10, help="factor by which the space is scaled down")
    parser.add_argument("--blur_sigma", type=float, default=0.5, help="guassian blur sigma")
    parser.add_argument("--common_destination", type=list, default=[-200, -200],
                        help="a common location for targets come from and go to")
    parser.add_argument("--cdl", type=float, default=0.0, help="how many targets go to location")
    parser.add_argument("--speed_layer", type=int, default=0, help="if speed is included in state space")
    parser.add_argument("--speed_scale", type=int, default=1,
                        help="how much the reward is scaled for seeing moving objects compared to not moving object")
    parser.add_argument("--radars", type=int, default=1)
    parser.add_argument("--relative_change", type=int, default=0)
    parser.add_argument("--penalize_no_movement", type=int, default =1, help="pnm: if no change in action is taken, and the reward is 0, this action is  penalized with a reward of -1")
    parser.add_argument("--type_of_MARL", type=str, default="single_agent", help="type of shared info in the MARL system")
    parser.add_argument("--outside_radar_value", type=float, default=0.5, help="value outside of radar observation area")
    args = parser.parse_args()

    args = parser.parse_args()
    transform = T.ToPILImage()
    test = Simulation(args)
    images = []
    images_2 =[]
    #images_3=[]


    for t in range(70):
        test.update_t([(t*47)%15,((-t)%15)])
        images.append(transform(torch.stack([test.world_view.last_tensor] * 3, dim=0)))
        #images_2.append(transform(torch.stack([test.individual_states[0]] * 3, dim=0)))
        #images_3.append(transform(torch.stack([test.individual_states[1]] * 3, dim=0)))

    images[0].save("./images/cartesian1.gif", save_all=True, append_images=images, duration=test.t, loop=0)
    #images_2[0].save("./images/cartesian2.gif", save_all=True, append_images=images_2, duration=test.t, loop=0)
    #images_3[0].save("./images/cartesian3.gif", save_all=True, append_images=images_3, duration=test.t, loop=0)
    print('done')


if __name__ == "__main__":
    main()
