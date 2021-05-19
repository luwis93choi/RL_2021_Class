import random
import numpy as np
from itertools import product

import copy

class agent():

    def __init__(self, player_num=0):

        self.player_num = player_num

        self.discount_factor = 1
        self.learning_rate = 0.01

        self.grid_height = 3
        self.grid_width = 3

        # Action space generation
        self.action_space = [[0, 0], [0, 1], [0, 2], 
                             [1, 0], [1, 1], [1, 2], 
                             [2, 0], [2, 1], [2, 2]]

        # State space generation
        self.state_space = product([2, 1, 0], repeat=self.grid_height * self.grid_width)
        self.state_space = np.reshape(list(self.state_space), (-1, self.grid_height, self.grid_width))
                
        self.value_table = np.zeros((len(self.state_space), len(self.action_space)))     # State (Height), Action Space (Width)
        self.memory = []

    def act(self, board):

        ### Random Policy ###

        # Draw random coordinate in board until it is verified as a valid move
        while True:
            
            rand_action_idx = random.randint(0, len(self.action_space)-1)

            rand_action = self.action_space[rand_action_idx]

            # If there is no more space on the board, skip current iteration and return illegal value
            if np.count_nonzero(board == 0) == 0:

                return [-1, -1]

            # If there is a space at the current coordinate, break the loop and use current action selection
            elif board[rand_action[0], rand_action[1]] == 0: 
                # board[rand_y, rand_x] = self.player_num
                break
        
        return rand_action

    ### State-Action value function update ###
    def update(self, state, next_state, action, reward, done):

        for idx in range(len(self.state_space)):
            if np.array_equal(self.state_space[idx], state):
                state_idx = idx
                break

        for idx in range(len(self.state_space)):
            if np.array_equal(self.state_space[idx], next_state):
                next_state_idx = idx
                break

        for idx in range(len(self.action_space)):
            if np.array_equal(self.action_space[idx], action):
                action_idx = idx
                break

        # Update value table according to TD(0)
        V_t = self.value_table[state_idx, action_idx]

        V_t_1 = np.mean(self.value_table[next_state_idx, :])

        self.value_table[state_idx, action_idx] = V_t + self.learning_rate * (reward + self.discount_factor * V_t_1 - V_t)