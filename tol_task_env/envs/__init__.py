from gym.envs.registration import register

register(id='TolTask-v0',
         entry_point='envs.custom_tol_env_dir:ToLTaskEnv',
         max_episode_steps=100,
         )

"""
Difference between v0 and v1 is the way how rewards are calculated:
V0 - calculates on every step
v1 - reward is only given when the end of the episode
"""
register(id='TolTask-v1',
         entry_point='envs.custom_tol_env_dir:ToLTaskRewardsAfterCompletedEnv',
         max_episode_steps=100,
         )
