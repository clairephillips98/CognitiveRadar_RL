"""""
ABANDONNING THIS FILE AS I DONt THINK I NEED IT!
I may be wrong and I may return 
"""""

import random
from radar_env.simulate import Simulation
import torch

# if torch.cuda.is_available():
#     import jax.numpy as jnp
#
# else:
import numpy as jnp
if torch.cuda.is_available():
    show = False
    pygame = None
else:
    import pygame
    show = True
from functools import reduce
import gymnasium as gym
from rl_agents.config import GPU_NAME

device = torch.device(GPU_NAME if torch.cuda.is_available() else "cpu")
print(device)


class RadarEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self,args, seed=None,render_mode='human', size=5):
        self.seed = seed
        self.args=args
        self.game = Simulation(self.args)
        self.info = torch.empty(0,4)
        self.size = size  # The size of the square grid
        self.window_size = jnp.array(self.game.next_image.size()) * self.size  # The size of the PyGame window
        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).
        self.observation_space = gym.spaces.Dict(
            {
                # "agent_angles": gym.spaces.Space(np.array([radar.viewing_angle for radar in self.game.radars])),
                "observation": gym.spaces.Box(low=0.0, high=1.0,
                                              shape=tuple(list(self.game.next_image.size())), dtype=jnp.float32),
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

        self._action_to_angle = {i: self.game.to_action(i) for i in range(self.action_size)}  # only up to 2 radars
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


    def _get_obs(self):
        if self.game.speed_layers is not None:
            expanded_image = self.game.next_image.unsqueeze(2)
            joined_tensor = torch.cat((expanded_image, self.game.speed_layers), dim=2).permute(2, 0, 1)
            return joined_tensor
        else:
            return self.game.next_image


    def info_analysis(self):
        info=torch.vstack([target.stats for target in self.game.targets]).to(device)
        world_loss = self.game.measure_world_loss(input=self.game.next_image,
                                                  target=self.game.create_hidden_target_tensor())
        time_til_first_view = info[:,2]
        time_til_first_view[time_til_first_view == -1] = self._max_episode_steps
        return {'time_til_first_view': torch.Tensor(time_til_first_view),'views_vel': info[:,[0,1]],
                'world_loss': torch.Tensor(world_loss).to(device)}

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        if self.seed is not None:
            self.seed += 1
        super().reset(seed=self.seed)

        self.game = Simulation(self.args,seed=self.seed)
        # self.info = torch.empty(0,4)
        # Choose the agent starting angle at random
        self._agent_angle = [random.randint(0, radar.num_states) for radar in self.game.radars]

        observation = self._get_obs()

        if self.render_mode == "human":
            self._render_frame()

        return observation, None

    def step(self, action):
        # Map the action to angle of view of all agents
        self._agent_angle = self._action_to_angle[action]
        self.game.update_t(self._agent_angle)
        terminated = 1 if self.game.t == self._max_episode_steps else 0
        reward = self.game.reward
        observation = self._get_obs()
        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, None

    def render(self):
        if (self.render_mode == "rgb_array") & show:
            return self._render_frame()

    def _render_frame(self):
        if show:
            if self.window is None and self.render_mode == "human":
                pygame.init()
                pygame.display.init()
                self.window = pygame.display.set_mode(
                    tuple(self.window_size)
                )
            if self.clock is None and self.render_mode == "human":
                self.clock = pygame.time.Clock()

            canvas = pygame.display.set_mode(tuple(self.window_size))
            ret = jnp.empty((*self.window_size, 3), dtype=jnp.uint8)
            plotted_arr = (self.game.next_image.numpy() * 255).astype(int)
            plotted_arr = jnp.repeat(plotted_arr, 5, axis=0)
            plotted_arr = jnp.repeat(plotted_arr, 5, axis=1)
            ret[:, :, 2] = ret[:, :, 1] = ret[:, :, 0] = plotted_arr

            surf = pygame.surfarray.make_surface(ret)
            if self.render_mode == "human":
                canvas.blit(surf, (0, 0))
                pygame.display.update()

                # We need to ensure that human-rendering occurs at the predefined framerate.
                # The following line will automatically add a delay to keep the framerate stable.
                self.clock.tick(self.metadata["render_fps"])
            else:  # rgb_array
                return jnp.transpose(
                    jnp.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
                )

    def close(self):
        if (self.window is not None) & show:
            pygame.display.quit()
            pygame.quit()
