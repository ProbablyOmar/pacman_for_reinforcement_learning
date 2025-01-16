from run import GameController
from constants import *
import numpy as np
import random
import pickle
from queue import PriorityQueue
from q_table_obs import *
import itertools
import os
import matplotlib.pyplot as plt 

def init_q_table ():
    q_table = {}
    bool_val = [0,1]
    action_val = [0,1,2,3]
    combinations = itertools.product(bool_val, bool_val, bool_val , bool_val , action_val , bool_val , bool_val , bool_val , bool_val , bool_val)
    combinations = list(combinations)
    for combination in combinations:
        q_table[combination] = np.array([0,0,0,0])
    return q_table
    

##games
game = GameController(rlTraining=True , mode = SAFE_MODE , move_mode = DISCRETE_STEPS_MODE , clock_tick= 0 , pacman_lives=1 , maze_mode=MAZE1 , pac_pos_mode=NORMAL_PAC_POS)
game_test1 = GameController(rlTraining=True , mode = SAFE_MODE , move_mode = DISCRETE_STEPS_MODE , clock_tick= 10 , pacman_lives=1 , maze_mode=MAZE1 , pac_pos_mode=NORMAL_PAC_POS)
game_test2 = GameController(rlTraining=True , mode = SAFE_MODE , move_mode = DISCRETE_STEPS_MODE , clock_tick= 10 , pacman_lives=1 , maze_mode=MAZE3 , pac_pos_mode=NORMAL_PAC_POS)
game_test3 = GameController(rlTraining=True , mode = SAFE_MODE , move_mode = DISCRETE_STEPS_MODE , clock_tick= 10 , pacman_lives=1 , maze_mode=MAZE4 , pac_pos_mode=NORMAL_PAC_POS)
game_test4 = GameController(rlTraining=True , mode = SAFE_MODE , move_mode = DISCRETE_STEPS_MODE , clock_tick= 10 , pacman_lives=1 , maze_mode=MAZE5 , pac_pos_mode=NORMAL_PAC_POS)

## learning params
EPISODES = 200
LEARNING_RATE = 0.01
DISCOUNT_FACTOR = 0.99
EPSILON = 1
EPSILON_DECAY = 0.97
MIN_EPSILON = 0.01

AVG_EVERY = 10

##observation and action shapes
total_states = 2048 #from our obs space (16 * 4 * 16 * 2)
total_actions = 4

# Q-Table initialization
q_table = init_q_table()

##tracking reward
ep_rewards = []
ep_lengths = []
aggr_ep_rewards = {'ep': [], 'avg': [], 'min': [], 'max': [] , 'ep_length' : []}

MAX_AVG_REWARD = float("-inf")
PLOTS_DIR = 'plots'
q_tables_DIR = "q_tables"

#training loop
for episode in range(EPISODES):
    episode_reward = 0   
    observation = get_observation(game)
    episode_length = 0

    done = False
    while not done:
        episode_length += 1

        if np.random.random() > EPSILON:
            action = np.argmax(q_table[tuple(observation)])  # Exploitation
        else:
            # possible_actions = []
            # for direction in range(4):
            #     if observation[direction] == 0:  # there is no wall in this direction
            #         possible_actions.append(direction)
            # action = random.choice(possible_actions) # Exploration
            action = random.choice([0,1,2,3])
            
        agent_direction = get_direction_value(action)
        game.update(render=False ,agent_direction = agent_direction)
        new_observation = get_observation(game)

        episode_reward += game.RLreward
        done = game.done


        if not done:
            #update Q values
            max_future_q = np.max(q_table[tuple(new_observation)])  # Max Q-value for next state (Value function)
            current_q = q_table[tuple(observation)][action]        # Current Q-value
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (game.RLreward + DISCOUNT_FACTOR * max_future_q)
            q_table[tuple(observation)][action]  = new_q             

        else:
            print(f"episode{episode} , with reward = {episode_reward}, episode length = {episode_length}")
            q_table[tuple(observation)][action] = 0

        observation = new_observation

    ep_rewards.append(episode_reward)
    ep_lengths.append(episode_length)

    ##update the epsilon value
    EPSILON = max(MIN_EPSILON , EPSILON * EPSILON_DECAY)
        

    #dir for saving the plots     
    if not os.path.exists(PLOTS_DIR):
        os.makedirs(PLOTS_DIR)
    if not os.path.exists(q_tables_DIR):
        os.makedirs(q_tables_DIR)
    
    if episode % AVG_EVERY  == 0:
        avg_reward = sum(ep_rewards[-AVG_EVERY:])/len(ep_rewards[-AVG_EVERY:])
        avg_ep_length = sum(ep_lengths[-AVG_EVERY:])/len(ep_lengths[-AVG_EVERY:])

        if avg_reward >= MAX_AVG_REWARD:
            with open(f"{q_tables_DIR}\\q_table_{episode}_episodes.pkl", "wb") as file:
                pickle.dump(q_table, file)

        aggr_ep_rewards['ep'].append(episode)
        aggr_ep_rewards['avg'].append(avg_reward)
        aggr_ep_rewards['min'].append(min(ep_rewards[-AVG_EVERY:]))
        aggr_ep_rewards['max'].append(max(ep_rewards[-AVG_EVERY:]))
        aggr_ep_rewards['ep_length'].append(avg_ep_length)

        #plotting 
        plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['avg'], label ="Average Reward")
        plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['min'], label="Min Reward")
        plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['max'], label="Max Reward")
        plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['ep_length'], label="AVG Episode length")
        plt.legend(loc="best")

        #save plotting 
        plt.title(f"Training Rewards (Episode {episode})")
        plot_filename = os.path.join(PLOTS_DIR, f'training_rewards_plot_episode_{episode}.png')
        plt.savefig(plot_filename)
        plt.close()

        print(f"Episode: {episode} avg: {avg_reward} min: {min(ep_rewards[-AVG_EVERY:])} max: {max(ep_rewards[-AVG_EVERY:])}")

