"""""
ABANDONNING THIS FILE AS I DONt THINK I NEED IT!
I may be wrong and I may return 
"""""

import random
random.seed(10)
from radar_env.simulate import Simulation
import numpy as np
np.random.seed(10)
import pygame
from functools import reduce
from math import floor
import gymnasium as gym




class RadarEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self,seed=None, render_mode=None, size=5):
        self.seed = seed
        self.blur_radius = 2
        self.scale = 50
        self.game = Simulation(self.blur_radius, self.scale)
        print(self.game.last_tensor)
        self.size = size  # The size of the square grid
        self.window_size = np.array(self.game.last_tensor.size()) * self.size  # The size of the PyGame window
        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).
        self.observation_space = gym.spaces.Dict(
            {
                # "agent_angles": gym.spaces.Space(np.array([radar.viewing_angle for radar in self.game.radars])),
                "observation": gym.spaces.Box(low=0.0, high=1.0,
                                              shape=tuple(list(self.game.last_tensor.size())), dtype=np.float32),
            }
        )

        self.action_size = int(reduce(lambda x, y: x * y, [radar.num_states for radar in self.game.radars]))
        # 1 radar for now, so the number of actions is the number of states
        self.action_space = gym.spaces.Discrete(self.action_size)

        """
        The following dictionary maps abstract actions from `self.action_space` to
        the direction we will walk in if that action is taken.
        I.e. 0 corresponds to "right", 1 to "up" etc.
        """

        self._action_to_angle = {i: self.to_action(i) for i in range(self.action_size)}  # only up to 2 radars

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
        self._max_episode_steps = 300

    def to_action(self, i):
        # can only be up to 2 actions
        if len(self.game.radars) == 1:
            return [i]
        else:
            return [i % self.game.radars[0].num_states, floor(i / self.game.radars[0].num_states)]

    def _get_obs(self):
        return self.game.last_tensor

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        if self.seed is not None:
            self.seed += 1
        super().reset(seed=self.seed)

        self.game = Simulation(self.blur_radius, self.scale, seed=self.seed)

        # Choose the agent starting angle at random
        self._agent_angle = [random.randint(0, radar.num_states) for radar in self.game.radars]

        observation = self._get_obs()
        info = None  # update this for when I need info

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):

        # Map the action to angle of view of all agents
        self._agent_angle = self._action_to_angle[action]

        self.game.update_t(self._agent_angle)
        terminated = 1 if self.game.t == 1000 else 0
        reward = self.game.reward
        observation = self._get_obs()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, None

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                tuple(self.window_size)
            )
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.display.set_mode(tuple(self.window_size))
        ret = np.empty((*self.window_size, 3), dtype=np.uint8)
        plotted_arr = (self.game.last_tensor.numpy() * 255).astype(int)
        plotted_arr = np.repeat(plotted_arr, 5, axis=0)
        plotted_arr = np.repeat(plotted_arr, 5, axis=1)
        ret[:, :, 2] = ret[:, :, 1] = ret[:, :, 0] = plotted_arr

        surf = pygame.surfarray.make_surface(ret)
        if self.render_mode == "human":
            canvas.blit(surf, (0, 0))
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
