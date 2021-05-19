import numpy as np
import copy

class env():

    def __init__(self):

        self.board = np.zeros((3, 3))   # Shape : height, width

    def reset(self):

        self.board = np.zeros((3, 3))   # Shape : height, width

    def detect_win(self):

        height = self.board.shape[0]
        width = self.board.shape[1]

        # Scan for a winner horizontally
        for i in range(height):

            value_types = np.unique(self.board[i, :])

            if (len(value_types) == 1) and (value_types[0] != 0):

                return value_types[0]
                
        # Scan for a winner vertically
        for i in range(width):

            value_types = np.unique(self.board[:, i])

            if (len(value_types) == 1) and (value_types[0] != 0):

                return value_types[0]

        # Scan for a winner diagonally
        if self.board[0, 0] == self.board[1, 1] == self.board[2, 2] != 0:

            return self.board[0, 0]

        return -1

    def detect_draw(self):

        if np.count_nonzero(self.board == 0) == 0: return True
        else: return False

    def step(self, player_num, action):

        reward = 0
        done = 0
        
        state = copy.deepcopy(self.board)

        self.board[action[0], action[1]] = player_num

        next_state = copy.deepcopy(self.board)

        if self.detect_win() != -1:

            done = 1
            reward = 1

        elif self.detect_draw() == False:

            done = 0
            reward = 0

        elif self.detect_draw() == True:

            done = 1
            reward = 0

        return state, next_state, reward, done