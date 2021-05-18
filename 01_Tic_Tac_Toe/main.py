from agent import agent
from env import env

RL_player_1 = agent(player_num=1)
RL_player_2 = agent(player_num=2)

Tic_Tac_Toe_Env = env()

while True:

    # Player 1
    RL_player_1_action = RL_player_1.act(Tic_Tac_Toe_Env.board)                                     # Player 1 chooses its action based on current state
    next_state, reward, done = Tic_Tac_Toe_Env.step(RL_player_1.player_num, RL_player_1_action)     # Player 1's action interacts with the environment
                                                                                                    # Due to this interaction, environment returns next state, reward, done for player 1
    print('RL_player_1_action : {}'.format(RL_player_1_action))
    
    if done == 1:
        break

    # Player 2
    RL_player_2_action = RL_player_2.act(Tic_Tac_Toe_Env.board)                                     # Player 2 chooses its action based on current state
    next_state, reward, done = Tic_Tac_Toe_Env.step(RL_player_2.player_num, RL_player_2_action)     # Player 2's action interacts with the environment
                                                                                                    # Due to this interaction, environment returns next state and reward for player 2
    print('RL_player_2_action : {}'.format(RL_player_2_action))

    if done == 1:
        break
    