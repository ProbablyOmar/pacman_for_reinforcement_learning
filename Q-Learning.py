import gymnasium as gym
import numpy as np
import random
import matplotlib.pyplot as plt 
import os
import time
from pacman_env import PacmanEnv

env = PacmanEnv()
env_render = gym.make("pacman-v0", render_mode="human")
env_no_render = gym.make("pacman-v0", render_mode=None)
    
    
LEARNING_RATE = 0.1
DISCOUNT = 0.95
EPISODES = 25000

epsilon = 0.5

START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = EPISODES // 2
 
epsilon_decay_value = epsilon/(END_EPSILON_DECAYING - START_EPSILON_DECAYING) #decay value

SHOW_EVERY = 5000
SAVE_EVERY = 5

#discretization parameters 

"""
##dynamically
DISCRETE_OS_SIZE = [20] *len(env.observation_space.spaces.keys())
dicrete_os_win_size = (env.observation_space.high - env.observation_space.low) / DISCRETE_OS_SIZE
q_table = np.random.uniform(low=-2, high=0, size=(total_states, total_actions))

def get_discrete_state(state):
    discrecte_state = (state - env.observation_space.low) / discrete_os_win_size
    return tuple(discrecte_state.astyp(np.int))
"""
total_states = 2560 #from our obs space (16 * 5 * 16 * 2)
total_actions = env.action_space.n

# Q-Table initialization
q_table = np.random.uniform(low=-2, high=0, size=(total_states, total_actions))



ep_rewards = []
aggr_ep_rewards = {'ep': [], 'avg': [], 'min': [], 'max': []}


def get_discrete_state(observation):
    try:
        #Extract values from obs
        walls_position = observation["walls_position"]
        best_direction = observation["best_direction"]
        ghosts_position = observation["ghosts_position"]
        trapped = observation["trapped"]

        
        walls_position_int = int("".join(map(str, walls_position.astype(int))), 2)  
        ghosts_position_int = int("".join(map(str, ghosts_position.astype(int))), 2)  

        
        best_direction_index = best_direction + 2  

        if trapped.ndim > 0:
            trapped_index = int(trapped.item())  
        else:
            trapped_index = int(trapped)  


        #combine all into a single discrete state 
        # Each part scaled based on  its size
        discrete_state = (
            (walls_position_int * 5 * 16 * 2) +
            (best_direction_index * 16 * 2) +
            (ghosts_position_int * 2) +
            trapped_index
        )

        return discrete_state

    except KeyError as e:
        raise KeyError(f"Missing key in observation: {e}")
    except Exception as e:
        raise RuntimeError(f"Unexpected error during discrete state calculation: {e}")

#training loop
for episode in range(EPISODES):
    episode_reward = 0
    
    if episode % SHOW_EVERY == 0:
        render = True
        env = env_render
    else:
        render = False
        env = env_no_render
    
    
    observation, info = env.reset()
    discrete_state = get_discrete_state(observation)

    #print(np.argmax(q_table[discrete_state]))
    done = False

    while not done:
        if np.random.random() > epsilon:
            action = np.argmax(q_table[discrete_state])  # Exploitation
        else:
            action = np.random.randint(0, env.action_space.n) # Exploration
            
        
        obs , reward, terminated, truncated, info = env.step(int(action))
        episode_reward += reward
        done = terminated or truncated
        
        new_discrete_state = get_discrete_state(obs)
        
        if render:
            env.render()
            
            
        if not done:
            #update Q values
            max_future_q = np.max(q_table[new_discrete_state])  # Max Q-value for next state
            current_q = q_table[discrete_state, action]         # Current Q-value
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
            q_table[discrete_state, action] = new_q             

        else:
            #print(f"We made it on epsiode {episode} ")
            print(f"Completed episode {episode} with reward {episode_reward}")
            q_table[discrete_state, action ] = 0

        discrete_state = new_discrete_state
        
    if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
        epsilon -= epsilon_decay_value
        
    ep_rewards.append(episode_reward)
    
    #dir for saing
    if not os.path.exists('qtables'):
        os.makedirs('qtables')
        
    save_dir = 'plots'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    if episode % SAVE_EVERY  == 0:

        #Save Q-table for this episode
        np.save(f"qtables/{episode}-qtable.npy", q_table)

        avg_reward = sum(ep_rewards[-SHOW_EVERY:])/len(ep_rewards[-SHOW_EVERY:])
        aggr_ep_rewards['ep'].append(episode)
        aggr_ep_rewards['avg'].append(avg_reward)
        aggr_ep_rewards['min'].append(min(ep_rewards[-SHOW_EVERY:]))
        aggr_ep_rewards['max'].append(max(ep_rewards[-SHOW_EVERY:]))
        
    
        
        
         
        #plotting 
        plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['avg'], label ="Average Reward")
        plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['min'], label="Min Reward")
        plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['max'], label="Max Reward")
        plt.legend(loc="best")
        #save lotting 
        plt.title(f"Training Rewards (Episode {episode})")
        plot_filename = os.path.join(save_dir, f'training_rewards_plot_episode_{episode}.png')
        plt.savefig(plot_filename)
        plt.close()

        
        print(f"Episode: {episode} avg: {avg_reward} min: {min(ep_rewards[-SHOW_EVERY:])} max: {max(ep_rewards[-SHOW_EVERY:])}")
                
env.close()


