import random
import numpy as np

class agent():

    def __init__(self, player_num=0):

        self.player_num = player_num

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

    # def update(self, state, )