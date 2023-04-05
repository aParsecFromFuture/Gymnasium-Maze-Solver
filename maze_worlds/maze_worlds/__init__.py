from gymnasium.envs.registration import register

register(
     id="maze_worlds/MyWorld-v0",
     entry_point="maze_worlds.envs:MouseCatEnv",
     max_episode_steps=300,
)