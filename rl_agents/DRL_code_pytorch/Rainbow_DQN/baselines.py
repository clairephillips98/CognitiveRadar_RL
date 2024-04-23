import torch
from functools import reduce
import random
from utils import action_unpack, action_repack
import torchvision.transforms as T


def simple_baseline(agent_setup, prev_action):
    if len(prev_action) == 2:
        action = [(x + 1) % int(agent_setup.env.action_size ** (1 / 2)) for x in prev_action]
    else:
        action = [(1 + prev_action[0]) % agent_setup.env.action_size]
    return action


def get_mask_function(agent_setup, a):
    x=agent_setup.env.game.world_view.x
    y=agent_setup.env.game.world_view.y
    if agent_setup.args.radars == 2:
        a=action_unpack(a, agent_setup.args.action_dim)
    else:
        a = [a]
    for i,a in enumerate(a):
        agent_setup.env.game.radars[i].update_t(0, a, False)

    masks = [agent_setup.env.game.world_view.draw_shape(x, y, agent_setup.env.game.radars[i].cartesian_coordinates,
                                                        agent_setup.env.game.radars[i].viewing_angle, agent_setup.env.game.radars[i].viewing_angle+agent_setup.env.game.radars[i].radians_of_view,
                                                        agent_setup.env.game.radars[i].max_distance)
             for i in range(agent_setup.args.radars)]
    mask = reduce(lambda x, y: torch.logical_or(x,y), masks)
    return mask


def get_mask(agent_setup):
    total_masks = list(map(lambda a: get_mask_function(agent_setup,a), range(agent_setup.env.action_size)))

    return total_masks

def get_sum_blur(state, mask):
    state_t = state.clone()
    selected_values = torch.flatten(state_t[mask])
    selected_values -= 0.5
    val = torch.sum((torch.abs(selected_values)))
    return val

def get_variance(state, mask):
    state_t = state.clone()
    selected_values = torch.flatten(state_t[mask])
    val = torch.var(selected_values)
    return val
def variance_blur_baseline(agent_setup, state, var_type, past_action=[0,0]):
    if random.choice([0, 1]) == 1:
        if var_type == 'min_var':
            rels = [get_variance(state,mask.unsqueeze(0)) for mask in agent_setup.masks]
            best = min(rels)
        elif var_type == 'max_var':
            rels = [get_variance(state,mask.unsqueeze(0)) for mask in agent_setup.masks]
            best = max(rels)
        elif var_type == 'sum_blur':
            rels = [get_sum_blur(state,mask.unsqueeze(0)) for mask in agent_setup.masks]
            best = min(rels)
        indexes = [i for i, value in enumerate(rels) if value == best]
        action = random.choice(
            indexes)  # if there are multiple views which have the same best variance then select that one
        action_ = action_unpack(action, agent_setup.env.action_size)
    else:
        action = random.choice(range(agent_setup.env.action_size-1))+1
        action=(action+action_repack(past_action, agent_setup.env.action_size)) % agent_setup.env.action_size
        action_ = action_unpack(action, agent_setup.env.action_size)
    return action, action_


def baselines_next_step(agent_setup, last_action, state):
    if last_action == None:
        rand_act = random.choice(range(agent_setup.env.action_size))
        if agent_setup.args.radars==2:
            last_action=action_unpack(rand_act, agent_setup.env.action_size)
        else:
            last_action=[rand_act]
    if agent_setup.args.baseline == 1:
        action_ = simple_baseline(agent_setup,last_action)
    elif agent_setup.args.baseline == 2:
        action_ = last_action
    elif agent_setup.args.baseline == 3:
        action, action_ = variance_blur_baseline(agent_setup, state, 'max_var', last_action)
    elif agent_setup.args.baseline == 4:
        action, action_ = variance_blur_baseline(agent_setup, state, 'min_var', last_action)
    elif agent_setup.args.baseline == 5:
        action, action_ = variance_blur_baseline(agent_setup, state, 'sum_blur', last_action)
    if agent_setup.args.baseline in [1,2]:
        if agent_setup.args.radars == 2:
            action = action_[0] + action_[1] * (agent_setup.env.action_size ** (1 / 2) - 1)
        else:
            action = action_[0]
    return action_, action
