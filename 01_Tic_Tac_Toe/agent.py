import random
import numpy as np
from itertools import product

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
            
            rand_x = random.randint(0, 2)
            rand_y = random.randint(0, 2)

            # If there is no more space on the board, skip current iteration and return illegal value
            if np.count_nonzero(board == 0) == 0:

                return [-1, -1]

            # If there is a space at the current coordinate, break the loop and use current action selection
            elif board[rand_y, rand_x] == 0: 
                # board[rand_y, rand_x] = self.player_num
                break
        
        return [rand_y, rand_x]

    def remember(self, state, next_state, action, reward, done):

        self.memory.append([state, next_state, action, reward, done])

    ### Update Agent's state-action value function ###
    def update(self):

        G_t = 0
        visit_states = []

        for sample in reversed(self.memory):
            state = sample[0]
            action = sample[2]
            reward = sample[3]

            for idx in range(len(self.state_space)):
                if np.array_equal(self.state_space[idx], state):
                    state_idx = idx
                    break

            for idx in range(len(self.action_space)):
                if np.array_equal(self.action_space[idx], action):
                    action_idx = idx
                    break

            visit_states.append(state)
            G_t = reward + self.discount_factor * G_t
            V_t = self.value_table[state_idx, action_idx]

            self.value_table[state_idx, action_idx] = V_t + self.learning_rate * (G_t - V_t)