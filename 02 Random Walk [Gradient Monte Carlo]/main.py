from numpy.core.fromnumeric import argmax
from agent import agent
from env import env

from tqdm import tqdm

import numpy as np
np.set_printoptions(threshold=np.inf)

import matplotlib.pyplot as plt

EPISODE = 5000

walk_length = 1000
group_num = 20

Random_Walk_Env = env(walk_length=walk_length, group_num=group_num)

walker_Agent = agent(walk_length=walk_length, group_num=group_num)

for episode in tqdm(range(EPISODE)):

    Random_Walk_Env.reset()
    
    while True:

        action = walker_Agent.act()

        state, next_state, reward, done = Random_Walk_Env.step(action)

        walker_Agent.remember(state, next_state, action, reward, done)

        if done == 1: break

    walker_Agent.update()

STATES = np.arange(1, walk_length, 0.01)
weights = [walker_Agent.value(i) for i in STATES]

plt.plot(STATES, weights)
plt.show()