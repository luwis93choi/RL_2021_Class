import random
import numpy as np

class agent():

    def __init__(self, player_num=0):

        self.player_num = player_num

    def play_random(self, board):

        ### Random Policy ###

        # Draw random coordinate in board until it is verified as a valid move
        while True:
            
            rand_x = random.randint(0, 2)
            rand_y = random.randint(0, 2)

            # If there is no more space on the board, skip current iteration
            if np.count_nonzero(board == 0) == 0:

                return False

            # If there is a space at the current coordinate, place player's numer on the board
            elif board[rand_x, rand_y] == 0: 
                board[rand_x, rand_y] = self.player_num
                break
        
        return True