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

        # Full state space generation
        self.full_state_space = product([2, 1, 0], repeat=self.grid_height * self.grid_width)
        self.full_state_space = np.reshape(list(self.full_state_space), (-1, self.grid_height, self.grid_width))
                
        ### State space filtering ###
        # Pick only mirror image element
        self.state_space = []
        for i in range(len(self.full_state_space)):
            if(self.full_state_space[i][0][1] == self.full_state_space[i][0][2] == self.full_state_space[i][1][2] == 0):
                self.state_space.append(self.full_state_space[i])
        self.state_space = np.array(self.state_space)
        
        self.value_table = np.zeros((len(self.state_space), len(self.action_space)))     # State (Height), Action Space (Width)
        
        self.memory = []

    def mirror_equal(self, board):

        for rotation_count in range(4):
            for mirror_idx in range(len(self.state_space)):
                mirror = self.state_space[mirror_idx]                
                rotated_board = np.rot90(board, rotation_count)
                
                if (rotated_board[0, 0] == mirror[0, 0]) and (rotated_board[1, 1] == mirror[1, 1]) and (rotated_board[2, 2] == mirror[2, 2]) and \
                   (rotated_board[1, 0] == mirror[1, 0]) and (rotated_board[2, 0] == mirror[2, 0]) and (rotated_board[2, 1] == mirror[2, 1]):

                    return mirror_idx
        return -1

    def act(self, board):

        ### Random Policy ###
        # Draw random coordinate in board until it is verified as a valid move
        zero_idx = np.where(board == 0)

        if len(zero_idx) == 0:
            return [-1, -1]

        rand_action_idx = random.randint(0, len(zero_idx[0])-1)
        
        rand_action = [zero_idx[0][rand_action_idx], zero_idx[1][rand_action_idx]]
        
        return rand_action

    ### State-Action value function update ###
    def update(self, state, next_state, action, reward, done):

        state_idx = self.mirror_equal(state)

        next_state_idx = self.mirror_equal(next_state)

        for idx in range(len(self.action_space)):
            if np.array_equal(self.action_space[idx], action):
                action_idx = idx
                break

        # Update value table according to TD(0)
        V_t = self.value_table[state_idx, action_idx]

        V_t_1 = np.mean(self.value_table[next_state_idx, :])

        self.value_table[state_idx, action_idx] = V_t + self.learning_rate * (reward + self.discount_factor * V_t_1 - V_t)

        return reward + self.discount_factor * V_t_1 - V_t