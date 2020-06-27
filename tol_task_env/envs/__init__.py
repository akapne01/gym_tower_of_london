from gym.envs.registration import register

register(id='TolTask-v0',
         entry_point='envs.custom_tol_env_dir:ToLTaskEnv',
         )
