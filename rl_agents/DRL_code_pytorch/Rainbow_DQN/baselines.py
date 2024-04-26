import torch
from functools import reduce
import random
from utils import action_unpack, action_repack
import torchvision.transforms as T
from math import inf
def simple_baseline(agent_setup, prev_action):
    if len(prev_action) == 2:
        action = [(x + 1) % int(agent_setup.env.action_size ** (1 / 2)) for x in prev_action]
    else:
        action = [(1 + prev_action[0]) % agent_setup.env.action_size]
    return action

def get_sum_blur(state, mask):
    state_t = state.clone()
    selected_values = torch.flatten(state_t[mask])
    selected_values -= 0.5
    selected_values = torch.abs(torch.abs(selected_values-0.5)-0.5)
    val = torch.sum(-torch.log(1-selected_values))
    return val

def get_variance(state, mask):
    state_t = state.clone()
    selected_values = torch.flatten(state_t[mask])
    val = torch.var(selected_values)
    return val
def variance_blur_baseline(agent_setup, state, var_type, past_action):
    if agent_setup.args.radars ==2:
        la = action_repack(past_action, agent_setup.env.action_size)
    else: la = past_action[0]
    if random.choice([0, 1]) == 1:
        if var_type == 'min_var':
            rels = [get_variance(state,mask.unsqueeze(0)) if i != la else inf for i, mask in enumerate(agent_setup.masks)]
            best = min(rels)
        elif var_type == 'max_var':
            rels = [get_variance(state,mask.unsqueeze(0)) if i != la else 0 for i, mask in enumerate(agent_setup.masks)]
            best = max(rels)
        elif var_type == 'sum_blur':
            rels = [get_sum_blur(state,mask.unsqueeze(0)) if i != la else 0 for i, mask in enumerate(agent_setup.masks)]
            best = max(rels)
        indexes = [i for i, value in enumerate(rels) if value == best]
        action = random.choice(
            indexes)  # if there are multiple views which have the same best variance then select that one
    else:
        action = random.choice(range(agent_setup.env.action_size-1))+1
        action = (action+action_repack(past_action, agent_setup.env.action_size)) % agent_setup.env.action_size
    if agent_setup.args.radars == 2:
        action_ = action_unpack(action, agent_setup.env.action_size)
    else:
        action_ = [action]
    return action, action_


def baselines_next_step(agent_setup, last_action, state):
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
