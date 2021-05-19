import random
import numpy as np
from itertools import product

import copy

class agent():

    def __init__(self, walk_length=1000, group_num=20):

        self.walk_length = 1000
        self.group_num = group_num
        self.group_size = self.walk_length // self.group_num

        print(self.group_size)

        self.terminal_states = [0, walk_length + 1]

        self.group_weights = np.zeros(group_num)

        self.action = [-1, 1]

        self.memory = []

        self.learning_rate = 1e-5

    def value(self, state):

        if state <= self.terminal_states[0]:
            state = self.terminal_states[0]

        elif state >= self.terminal_states[1]:
            state = self.terminal_states[1]

        if state in self.terminal_states: return 0

        group_idx = (state - 1) // self.group_size

        return self.group_weights[int(group_idx)]

    def act(self):

        random_stride = np.random.randint(1, 101)

        random_action_idx = np.random.randint(0, 2)

        return random_stride * self.action[random_action_idx]

    def remember(self, state, next_state, action, reward, done):

        self.memory.append([state, next_state, action, reward, done])

    ### State-Action value function update ###
    def update(self):

        for sample in reversed(self.memory):
            state = sample[0]
            next_state = sample[1]
            action = sample[2]
            reward = sample[3]

            MC_error = self.learning_rate * (reward - self.value(state))
            
            group_idx = (state - 1) // self.group_size
                
            if state <= self.terminal_states[0]:
                state = self.terminal_states[0]

            elif state >= self.terminal_states[1]:
                state = self.terminal_states[1]
                
            self.group_weights[int(group_idx)] += MC_error