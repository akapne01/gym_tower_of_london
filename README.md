# Custom OpenAi Gym Tower of London Task Environment 
released under CC-BY 2.0 licence

## Install environment
cd into the directory: tol_task_env and execute: 
<code> pip install -e . </code>

## Use environment in your python code:
<code>import gym </code>
<code>import envs </code>
<code>env = gym.make('TolTask-v0') # creates task with start_state=16, goal_state=25</code>
<code>env = gym.make('TolTask-v0', start_state=33, goal_state=54) # Allows to define custom tasks.</code>
<code>action, reward, done, info = env.step(a)</code>
<code>
for _ in range(1000):
    env.render()  
    if env.is_done:
          break 
     env.step(env.get_random_action()) # take a random action
env.close()
</code>
## Due to the specifics how ToL task works, to get a random action, please use: 
<code> env.get_random_action() </code>
## Please note that currently the number of minimum moves is hardcoded in the environment for following problems (start, goal): minimum number of moves
MIN_MOVES = {
    (53, 14): 4,
    (11, 52): 5,
    (34, 56): 5,
    (46, 16): 6,
    (33, 52): 6,
    (13, 32): 6,
    (16, 25): 4,
    (24, 66): 5,
    (33, 54): 4,
    (23, 62): 6,
    (16, 46): 6,
    (51, 32): 5,
}

## To initalize problems it uses numbers as proposed by Berg & Byrd, 2002: https://www.researchgate.net/profile/W_Berg/publication/11201010_The_Tower_of_London_Spatial_Problem-Solving_Task_Enhancing_Clinical_and_Research_Implementation/links/0a85e5321de4fed8c6000000/The-Tower-of-London-Spatial-Problem-Solving-Task-Enhancing-Clinical-and-Research-Implementation.pdf

