import gym
import envs

# State 1st number is a colour permutation, 2nd number is ball arrangement as per (Berg & Byrd 2002)
env = gym.make('TolTask-v0', start_state=53, goal_state=14)
a_s = env.action_space
print('Action space', a_s)

while not env.is_done:
    env.render()
    env.step(env.get_random_action())
env.close()
