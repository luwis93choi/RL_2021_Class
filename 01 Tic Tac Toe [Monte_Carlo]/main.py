from agent import agent
from env import env

from tqdm import tqdm

import numpy as np
np.set_printoptions(threshold=np.inf)

import matplotlib.pyplot as plt

RL_player_1 = agent(player_num=1)
RL_player_2 = agent(player_num=2)

Tic_Tac_Toe_Env = env()

EPISODE = 40000

init_state_value_RL_player_1 = np.zeros((EPISODE, len(RL_player_1.action_space)))

MC_errors = []
running_average_length = 1000
running_average = []

for episode in tqdm(range(EPISODE)):

    Tic_Tac_Toe_Env.reset()
    
    while True:

        # Player 1
        RL_player_1_action = RL_player_1.act(Tic_Tac_Toe_Env.board)     # Player 1 chooses its action based on current state
        
        state, next_state, reward, done = Tic_Tac_Toe_Env.step(RL_player_1.player_num, RL_player_1_action)      # Player 1's action interacts with the environment
                                                                                                                # Due to this interaction, environment returns next state, reward, done for player 1
        RL_player_1.remember(state, next_state, RL_player_1_action, reward, done)

        if done == 1: break

        # Player 2
        RL_player_2_action = RL_player_2.act(Tic_Tac_Toe_Env.board)     # Player 2 chooses its action based on current state
        
        state, next_state, reward, done = Tic_Tac_Toe_Env.step(RL_player_2.player_num, RL_player_2_action)      # Player 2's action interacts with the environment
                                                                                                                # Due to this interaction, environment returns next state and reward for player 2
        RL_player_2.remember(state, next_state, RL_player_2_action, reward, done)
        
        if done == 1: break

    MC_error_RL_player_1 = RL_player_1.update()
    RL_player_1.memory.clear()

    init_state_value_RL_player_1[episode] = RL_player_1.value_table[-1]

    MC_error_RL_player_2 = RL_player_2.update()
    RL_player_2.memory.clear()
    
    MC_errors.append(MC_error_RL_player_1)

    if episode > running_average_length:
        running_average.append(np.mean(np.array(MC_errors[-running_average_length:])))

print('RL Player 1 Init State-Action Value : {}'.format(RL_player_1.value_table[-1]))

print('Best First Move at Initial State S0 of Game : {}'.format(RL_player_1.action_space[np.argmax(RL_player_1.value_table[-1])]))

# Plot State-Action value transition at initial state S0
for i in range(len(RL_player_1.action_space)):
    plt.plot(range(EPISODE), init_state_value_RL_player_1[:, i], '*-', label='Action {}'.format(i))

plt.title('Monte Carlo : State-Action Value at Initial State S0')
plt.xlabel('Episode')
plt.ylabel('State-Action Value')
plt.legend()
plt.show()

plt.cla()

plt.plot(range(len(running_average)), running_average)
plt.title('Mean Monte-Carlo Error at Each Episode')
plt.xlabel('Episode')
plt.ylabel('Mean Monte-Carlo Error')
plt.show()