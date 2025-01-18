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
import torch 

# def init_q_table (q_table_path):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     if os.path.exists(q_table_path):
#         with open(q_table_path, "rb") as file:
#             q_table = pickle.load(file)
#             for key in q_table:
#                 q_table[key] = q_table[key].to(device)

#         bool_val = [0,1]
#         action_val = [4]
#         combinations = itertools.product(bool_val, bool_val, bool_val , bool_val , action_val , bool_val , bool_val , bool_val , bool_val , bool_val)
#         combinations = list(combinations)
#         for combination in combinations:
#             q_table[combination] = np.array([0,0,0,0])
#         return q_table
#     else:
#         print(f"Error: Q-table file '{q_table_path}' not found.")
#         exit()

#####
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

q_table_path = "q_table_300_episodes.pkl"

def init_q_table (q_table_path):
    if os.path.exists(q_table_path):
        with open(q_table_path, "rb") as file:
            q_table = pickle.load(file)
            for key in q_table:
                q_table[key] = q_table[key].to(device)
        # q_table = {}
        bool_val = [0,1]
        action_val = [4]
        combinations = itertools.product(bool_val, bool_val, bool_val , bool_val , action_val , bool_val , bool_val , bool_val , bool_val , bool_val)
        combinations = list(combinations)
        for combination in combinations:
            # q_table[combination] = np.array([0,0,0,0])
            q_table[combination] = torch.zeros(4, device='cuda') ###
        return q_table
    else:
        print(f"Error: Q-table file '{q_table_path}' not found.")
        exit()

##games
game = GameController(rlTraining=True , mode = NORMAL_MODE , move_mode = DISCRETE_STEPS_MODE , clock_tick = 0 , pacman_lives=3 , maze_mode=MAZE1 , pac_pos_mode=NORMAL_PAC_POS)
game_test1 = GameController(rlTraining=True , mode = NORMAL_MODE , move_mode = DISCRETE_STEPS_MODE , clock_tick= 10 , pacman_lives=3 , maze_mode=MAZE1 , pac_pos_mode=NORMAL_PAC_POS)
game_test2 = GameController(rlTraining=True , mode = NORMAL_MODE , move_mode = DISCRETE_STEPS_MODE , clock_tick= 10 , pacman_lives=3 , maze_mode=MAZE3 , pac_pos_mode=NORMAL_PAC_POS)
game_test3 = GameController(rlTraining=True , mode = NORMAL_MODE , move_mode = DISCRETE_STEPS_MODE , clock_tick= 10 , pacman_lives=3 , maze_mode=MAZE4 , pac_pos_mode=NORMAL_PAC_POS)
game_test4 = GameController(rlTraining=True , mode = NORMAL_MODE , move_mode = DISCRETE_STEPS_MODE , clock_tick= 10 , pacman_lives=3 , maze_mode=MAZE5 , pac_pos_mode=NORMAL_PAC_POS)


## learning params for sarsa1 and q_table
# EPISODES = 200
# LEARNING_RATE = 0.01
# DISCOUNT_FACTOR = 0.99
# EPSILON = 1.0
# EPSILON_DECAY = 0.97
# MIN_EPSILON = 0.01
# CHEAT_PROB = 0.5

## learning params for sarsa2
EPISODES = 3000
LEARNING_RATE = 0.01
DISCOUNT_FACTOR = 0.99
EPSILON = 1.0
EPSILON_DECAY = 0.99
MIN_EPSILON = 0.01
CHEAT_PROB = 0.5

AVG_EVERY = 30

##observation and action shapes
total_states = 2048 #from our obs space (16 * 4 * 16 * 2)
total_actions = 4

# Q-Table initialization
q_table = init_q_table(q_table_path)

##tracking reward
ep_rewards = []
ep_lengths = []

level_comp_per = []  #this should store the level_completion_percentage (num of pellets remained in the maze)

NUM_WINS = 0
aggr_ep_rewards = {'ep': [], 'avg_score': [], 'ep_length' : [] , 'num_wins' : [] , 'level_comp_per' : []}

MAX_AVG_REWARD = float("-inf")
PLOTS_DIR_BIG = 'plots_sarsa_ghosts_rewardsAndEpisodeLengths'
PLOTS_DIR_SMALL = 'plots_sarsa_ghosts_WinAndLevelCompletion'
q_tables_DIR = "q_tables_sarsa_ghosts_complete"


def tensor_to_tuple(tensor):
    return tuple(v.item() for v in tensor)

#training loop
for episode in range(EPISODES):
    episode_reward = 0   
    # observation = get_observation(game)
    observation = torch.tensor(get_observation(game), device=device)

    episode_length = 0
    ###
    #num_pellets_remaining = len(game.pellets.pelletList)
    ###
    done = False

    while not done:  #start the episode
        episode_length += 1

        if np.random.random() > EPSILON:
            max_q_value = float("-inf")
            for direction in range(4):
                if observation[direction] == 0 :    #there is no wall in this direction
                    if q_table[tensor_to_tuple(observation)][direction] >= max_q_value:
                        max_q_value = q_table[tensor_to_tuple(observation)][direction]
                        action = direction  # Exploitation

        else:   #when exploreing there is 0.5 probability we are going to cheat
            if np.random.random() > CHEAT_PROB and observation[4] != 4:
                action = observation[4]
            else:
                possible_actions = []
                for direction in range(4):
                    if observation[direction] == 0:  # there is no wall in this direction
                        possible_actions.append(direction)
                action = random.choice(possible_actions) # Exploration

        agent_direction = get_direction_value(action)
        game.update(render=False ,agent_direction = agent_direction)
        # new_observation = get_observation(game)
        new_observation = get_observation(game)  
        new_observation = torch.tensor(new_observation, device=device)
        
        episode_reward += game.RLreward
        done = game.done

        if not done:
            #update Q values
            if np.random.random() > EPSILON:
                max_next_q_value = float("-inf")
                for direction in range(4):
                    if new_observation[direction] == 0 :    #there is no wall in this direction
                        if q_table[tensor_to_tuple(new_observation)][direction] >= max_next_q_value:
                            max_next_q_value = q_table[tensor_to_tuple(new_observation)][direction]
                            next_action = direction  # Exploitation

            else:
                if np.random.random() > CHEAT_PROB and new_observation[4] != 4:
                    next_action = new_observation[4]
                else:
                    possible_actions = []
                    for direction in range(4):
                        if observation[direction] == 0:  # there is no wall in this direction
                            possible_actions.append(direction)
                    next_action = random.choice(possible_actions) # Exploration

            future_q = q_table[tensor_to_tuple(new_observation)][next_action]  # Max Q-value for next state
            current_q = q_table[tensor_to_tuple(observation)][action]        # Current Q-value
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (game.RLreward + DISCOUNT_FACTOR * future_q)
            q_table[tensor_to_tuple(observation)][action]  = new_q 
            ###
            #num_pellets_remaining = len(game.pellets.pelletList)  ## calculate the num_pellets_remaining before done because at this time the number of pellets will be reset
            ###
        else:
            num_pellets_remaining = len(game.pellets.pelletList)
            print(f"episode{episode} , with reward = {episode_reward}, episode length = {episode_length} , won: {game.win} , remaining pellets: {num_pellets_remaining}")
            q_table[tensor_to_tuple(observation)][action] = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (game.RLreward)

        observation = new_observation

    ### 
    if game.win == True:
        NUM_WINS +=1

    comp_per = (NUM_PELLETS - num_pellets_remaining) / NUM_PELLETS

    ep_rewards.append(episode_reward)
    ep_lengths.append(episode_length)
    level_comp_per.append(comp_per)
    ##update the epsilon value
    EPSILON = max(MIN_EPSILON , EPSILON * EPSILON_DECAY)

    #dir for saving the plots     
    if not os.path.exists(PLOTS_DIR_BIG):
        os.makedirs(PLOTS_DIR_BIG)
    if not os.path.exists(PLOTS_DIR_SMALL):
        os.makedirs(PLOTS_DIR_SMALL)
    if not os.path.exists(q_tables_DIR):
        os.makedirs(q_tables_DIR)
    
    if episode % AVG_EVERY  == 0:
        avg_num_wins = NUM_WINS / AVG_EVERY
        NUM_WINS = 0

        avg_reward = sum(ep_rewards[-AVG_EVERY:])/len(ep_rewards[-AVG_EVERY:])
        avg_ep_length = sum(ep_lengths[-AVG_EVERY:])/len(ep_lengths[-AVG_EVERY:])
        avg_level_comp_per = sum(level_comp_per[-AVG_EVERY:])/len(level_comp_per[-AVG_EVERY:])

        if avg_reward >= MAX_AVG_REWARD:
            with open(f"{q_tables_DIR}\\q_table_{episode}_episodes.pkl", "wb") as file:
                pickle.dump(q_table, file)
            MAX_AVG_REWARD = avg_reward

        aggr_ep_rewards['ep'].append(episode)
        aggr_ep_rewards['avg_score'].append(avg_reward)
        aggr_ep_rewards['ep_length'].append(avg_ep_length)
        aggr_ep_rewards['num_wins'].append(avg_num_wins)
        aggr_ep_rewards['level_comp_per'].append(avg_level_comp_per)

        #plotting Big plots (avg reward and avg episode lengths)
        plt.figure(figsize=(10, 6))
        plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['avg_score'], label ="Average Reward")
        plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['ep_length'], label="AVG Episode length")
        plt.legend(loc="best")
        #save plotting 
        plt.title(f"Training Rewards (Episode {episode})")
        plot_filename = os.path.join(PLOTS_DIR_BIG, f'training_rewards_plot_episode_{episode}.png')
        plt.savefig(plot_filename)
        plt.close()

        #plotting small plots (in percentage and level completion percentage)
        plt.figure(figsize=(10, 6))
        plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['num_wins'], label="AVG number of wins")
        plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['level_comp_per'], label="percentage of eaten pellets relative to all pellets")
        plt.legend(loc="best")

        #save plotting 
        plt.title(f"Training Rewards (Episode {episode})")
        plot_filename = os.path.join(PLOTS_DIR_SMALL, f'training_rewards_plot_episode_{episode}.png')
        plt.savefig(plot_filename)
        plt.close()

        print(f"Episode: {episode}, avg_score: {avg_reward}, level_comp_per: {avg_level_comp_per} , win percentage: {avg_num_wins}")