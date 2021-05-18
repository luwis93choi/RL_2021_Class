import numpy as np

class env():

    def __init__(self):

        self.board = np.zeros((3, 3))   # Shape : height, width

    def detect_win(self):

        height = self.board.shape[0]
        width = self.board.shape[1]

        # Scan for a winner horizontally
        for i in range(height):

            value_types = np.unique(self.board[i, :])

            if (len(value_types) == 1) and (value_types[0] != 0):

                print('Winner (Horizontal) : Player {}'.format(value_types[0]))

                return value_types[0]
                
        # Scan for a winner vertically
        for i in range(height):

            value_types = np.unique(self.board[:, i])

            if (len(value_types) == 1) and (value_types[0] != 0):

                print('Winner (Vertical) : Player {}'.format(value_types[0]))

                return value_types[0]

        # Scan for a winner diagonally
        if self.board[0, 0] == self.board[1, 1] == self.board[2, 2] != 0:

            print('Winner (Diagonal) : Player {}'.format(self.board[0, 0]))

            return self.board[0, 0]

        return -1

    def detect_draw(self):

        # print(self.board == 0)
        # print(np.count_nonzero(self.board == 0))

        if np.count_nonzero(self.board == 0) == 0: return True
        else: return False