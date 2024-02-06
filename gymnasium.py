"""""
ABANDONNING THIS FILE AS I DONt THINK I NEED IT!
I may be wrong and I may return 
"""""

import random
from simulate import Simulation
import numpy as np
import pygame
from functools import reduce
from math import floor
import gymnasium as gym

class RadarEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self,actions_size, render_mode=None, size=5):
        self.blur_radius = 3
        self.scale = 50
        self.game = Simulation(self.blur_radius,self.scale)
        self.size = size  # The size of the square grid
        self.window_size = 512  # The size of the PyGame window

        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).
        self.observation_space = gym.spaces.Dict(
            {
                "angles" : [radar.viewing_angle for radar in self.game.radars],
                "targets": self.game.targets,
            }
        )

        # 1 radar for now, so the number of actions is the number of states
        self.action_space = gym.spaces.Discrete(reduce(lambda x,y: x*y,
                                                   [radar.num_states for radar in self.game.radars]))



        """
        The following dictionary maps abstract actions from `self.action_space` to
        the direction we will walk in if that action is taken.
        I.e. 0 corresponds to "right", 1 to "up" etc.
        """

        self._action_to_angle = {i: self.to_action(i) for i in range(actions_size)} #only up to 2 radars

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None

    def to_action(self, i):
        #can only be up to 2 actions
        if len(self.game.radars) ==1:
            return [i]
        else:
            return [i % self.game.radars[0].num_states, floor(i/self.game.radars[0].num_states)]

    def _get_obs(self):
        return {"agent": self._agent_angle, "targets": self.game.next_tensor}

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        self.game = Simulation(self.blur_radius,self.scale)

        # Choose the agent's location uniformly at random
        self._agent_angle = [random.randint(0, radar.num_states) for radar in self.game.radars]

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):

        # Map the action to angle of view of all agents
        angles = self._action_to_angle[action]

        self.game.update_t(angles)
        terminated = 1 if self.game.t == 1000 else 0
        reward = self.game.reward
        observation = self._get_obs()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, None