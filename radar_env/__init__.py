from gymnasium.envs.registration import register

register(
     id="radarEnv",
     entry_point="radar_env:radar_gymnasium",
     max_episode_steps=300,
)