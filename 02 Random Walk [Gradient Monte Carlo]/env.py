import numpy as np
import copy

class env():

    def __init__(self, walk_length=1000, group_num=20):

        self.walk_length = 1000

        self.group_num = group_num

        self.terminal_states = [0, walk_length + 1]

        self.state = self.walk_length // 2

    def reset(self):

        self.state = self.walk_length // 2

    def step(self, action):

        state = copy.deepcopy(self.state)

        next_state = copy.deepcopy(self.state + action)

        if next_state <= self.terminal_states[0]:
            self.state = self.terminal_states[0]
            next_state = self.terminal_states[0]
            reward = -1
            done = 1

        elif next_state >= self.terminal_states[1]:
            self.state = self.terminal_states[1]
            next_state = self.terminal_states[1]
            reward = 1
            done = 1

        else:
            self.state += action
            reward = 0
            done = 0

        return state, next_state, reward, done