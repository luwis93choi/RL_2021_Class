import numpy as np

class env():

    def __init__(self):

        self.board = np.zeros((3, 3))

    def detect_win(self):

        # Scan for a winner horizontally
        print(self.board.shape)

        # Scan for a winner vertically

        # Scan for a winner diagonally

        pass

    def detect_draw(self):

        # print(self.board == 0)
        # print(np.count_nonzero(self.board == 0))

        if np.count_nonzero(self.board == 0) == 0: return True
        else: return False