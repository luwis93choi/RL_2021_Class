from agent import agent
from env import env

RL_player_1 = agent(player_num=1)
RL_player_2 = agent(player_num=2)

Tic_Tac_Toe_Env = env()

while True:

    RL_player_1.play_random(Tic_Tac_Toe_Env.board)
    RL_player_2.play_random(Tic_Tac_Toe_Env.board)

    if Tic_Tac_Toe_Env.detect_win() != -1:

        print('There is a winner / Game ends')
        print(Tic_Tac_Toe_Env.board)
        print('------------------------------')
        break

    elif Tic_Tac_Toe_Env.detect_draw() == False:

        print(Tic_Tac_Toe_Env.board)
        print('------------------------------')

    elif Tic_Tac_Toe_Env.detect_draw() == True:

        print('Board is full / No Winner / Game ends')
        print(Tic_Tac_Toe_Env.board)
        print('------------------------------')
        break