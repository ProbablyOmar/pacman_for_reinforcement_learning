import numpy as np

import matplotlib
import matplotlib.pyplot as plt

import random
import torch
from torch import nn
import yaml

from experience_replay import ReplayMemory
from dqn import DQN

from datetime import datetime, timedelta
import argparse
import itertools

from run import GameController
from constants import *
from observation import *
import os

# For printing date and time
DATE_FORMAT = "%m-%d %H:%M:%S"

# Directory for saving run info
RUNS_DIR = "runs_safe2"
os.makedirs(RUNS_DIR, exist_ok=True)

# 'Agg': used to generate plots as images and save them to a file instead of rendering to screen
matplotlib.use('Agg')

device = 'cuda'
#device = 'cpu' # force cpu, sometimes GPU not always faster than CPU due to overhead of moving data to GPU

# Deep Q-Learning Agent
class Agent():

    def __init__(self, hyperparameter_set):
        with open('hyperparameters.yml', 'r') as file:
            all_hyperparameter_sets = yaml.safe_load(file)
            hyperparameters = all_hyperparameter_sets[hyperparameter_set]
            # print(hyperparameters)

        self.hyperparameter_set = hyperparameter_set

        # Hyperparameters (adjustable)
        self.learning_rate_a    = hyperparameters['learning_rate_a']        # learning rate (alpha)
        self.discount_factor_g  = hyperparameters['discount_factor_g']      # discount rate (gamma)
        self.network_sync_rate  = hyperparameters['network_sync_rate']      # number of steps the agent takes before syncing the policy and target network
        self.replay_memory_size = hyperparameters['replay_memory_size']     # size of replay memory
        self.mini_batch_size    = hyperparameters['mini_batch_size']        # size of the training data set sampled from the replay memory
        self.epsilon_init       = hyperparameters['epsilon_init']           # 1 = 100% random actions
        self.epsilon_decay      = hyperparameters['epsilon_decay']          # epsilon decay rate
        self.epsilon_min        = hyperparameters['epsilon_min']            # minimum epsilon value
        self.stop_on_reward     = hyperparameters['stop_on_reward']         # stop training after reaching this number of rewards
        self.fc1_nodes          = hyperparameters['fc1_nodes']
        self.enable_double_dqn  = hyperparameters['enable_double_dqn']      # double dqn on/off flag
        self.enable_dueling_dqn = hyperparameters['enable_dueling_dqn']     # dueling dqn on/off flag
        self.cheat_prob         = hyperparameters['cheat_prob']
        # Neural Network
        self.loss_fn = nn.MSELoss()          # NN Loss function. MSE=Mean Squared Error can be swapped to something else.
        self.optimizer = None                # NN Optimizer. Initialize later.

        # Path to Run info
        self.LOG_FILE   = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}.log')
        self.MODEL_FILE = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}.pt')
        self.GRAPH_FILE = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}.png')

    def run(self, is_training=True, render=False):
        if is_training:
            start_time = datetime.now()
            last_graph_update_time = start_time

            log_message = f"{start_time.strftime(DATE_FORMAT)}: Training starting..."
            print(log_message)
            with open(self.LOG_FILE, 'w') as file:
                file.write(log_message + '\n')

        # Create instance of the environment.
        # Use "**self.env_make_params" to pass in environment-specific parameters from hyperparameters.yml.
        env = GameController(rlTraining=True , mode = SAFE_MODE , move_mode = DISCRETE_STEPS_MODE , clock_tick= 0 , pacman_lives=1 , maze_mode=MAZE1 , pac_pos_mode=NORMAL_PAC_POS)
        state = smart_observation(env)
        # Number of possible actions and state shape
        num_actions = 4
        num_states = len(state) 

        # List to keep track of rewards collected per episode.
        rewards_per_episode = []
        episodes_lengths = []

        # Create policy and target network. Number of nodes in the hidden layer can be adjusted.
        policy_dqn = DQN(num_states, num_actions, self.fc1_nodes, self.enable_dueling_dqn).to(device)

        if is_training:   
            epsilon = self.epsilon_init   # Initialize epsilon
            memory = ReplayMemory(self.replay_memory_size)  # Initialize replay memory
            
            target_dqn = DQN(num_states, num_actions, self.fc1_nodes, self.enable_dueling_dqn).to(device)  # Create the target network and make it identical to the policy network
            target_dqn.load_state_dict(policy_dqn.state_dict())

            self.optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=self.learning_rate_a)   # Policy network optimizer. "Adam" optimizer can be swapped to something else.
            epsilon_history = []  # List to keep track of epsilon decay
            step_count=0   # Track number of steps taken. Used for syncing policy => target network.          
            best_reward = -9999999  # Track best reward
        else:
            policy_dqn.load_state_dict(torch.load(self.MODEL_FILE))    # Load learned policy   
            policy_dqn.eval()    # switch model to evaluation mode

        # Train INDEFINITELY, manually stop the run when you are satisfied (or unsatisfied) with the results
        for episode in itertools.count():
            state = smart_observation(env)  # get initial state.
            cheat_obs = get_observation(env)

            state = torch.tensor(state, dtype=torch.float, device=device) # Convert state to tensor directly on device

            terminated = False      # True when agent reaches goal or fails
            episode_reward = 0.0    # Used to accumulate rewards per episode
            episode_length = 0
            # Perform actions until episode terminates or reaches max rewards
            # (on some envs, it is possible for the agent to train to a point where it NEVER terminates, so stop on reward is necessary)
            while(not terminated and episode_reward < self.stop_on_reward):  ## episode starts
                # Select action based on epsilon-greedy
                if is_training and random.random() < epsilon:  ## exploration (select random action or cheat) 
                    pref_dir = cheat_obs[4]
                    if np.random.random() > self.cheat_prob and pref_dir != STOP:
                        action = pref_dir
                    else:
                        possible_actions = []
                        for direction in range(4):
                            if cheat_obs[direction] == 0:  # there is no wall in this direction
                                p_action = get_direction_value(direction)
                                possible_actions.append(p_action)
                        action = random.choice(possible_actions) # Exploration
                    action = torch.tensor(action, dtype=torch.int64, device=device)

                else:            ### exploitation # select best action
                    with torch.no_grad():
                        actions = policy_dqn(state.unsqueeze(dim=0)).squeeze()
                        best_q = torch.tensor(float("-inf"), dtype=torch.float, device=device)
                        best_action = torch.tensor(0, dtype=torch.int64, device=device)

                        for action , q_value in enumerate(actions):
                            if q_value.item() > best_q.item() and cheat_obs[action] == 0:  #there is no wall and this is the biggest q value so far
                                best_q.fill_(q_value.item())
                                best_action.fill_(action)
                        action = torch.tensor(best_action.item(), dtype=torch.int64, device=device)
                        action.fill_(get_direction_value(action.item()))
                # Execute action. Truncated and info is not used.
                env.update(agent_direction = action.item() , render=render)
                action.fill_(get_direction_idx(action.item()))

                new_state = smart_observation(env)
                new_cheat_obs = get_observation(env)
                reward = env.RLreward
                terminated = env.done
                if reward !=0 and reward !=1.2 and reward != -0.5:
                    print("******************************")
                    print("cheat: " , cheat_obs)
                    print("state: " , state)
                    print("step reward: " ,reward)
                    print("num of pellets: " , len(env.pellets.pelletList))
                # Accumulate rewards
                episode_reward += reward
                episode_length += 1

                # Convert new state and reward to tensors on device
                new_state = torch.tensor(new_state, dtype=torch.float, device=device)
                reward = torch.tensor(reward, dtype=torch.float, device=device)

                if is_training:                   
                    memory.append((state, action, new_state, reward, terminated))  # Save experience into memory       
                    step_count+=1   # Increment step counter

                # Move to the next state
                state = new_state
                cheat_obs = new_cheat_obs

            env.update(agent_direction = STOP , render=render)  ## make another update after fininshing the episode to restart 
            ############# Here we finished the episode
            print(f"finished episode with reward: {episode_reward} and episode length: {episode_length} , epsilon = {epsilon}")
            # Keep track of the rewards collected per episode.
            rewards_per_episode.append(episode_reward)
            episodes_lengths.append(episode_length)

            # Save model when new best reward is obtained.
            if is_training:
                if episode_reward > best_reward:
                    log_message = f"{datetime.now().strftime(DATE_FORMAT)}: New best reward {episode_reward:0.1f} ({(episode_reward-best_reward)/best_reward*100:+.1f}%) at episode {episode}, saving model..."
                    print(log_message)
                    with open(self.LOG_FILE, 'a') as file:
                        file.write(log_message + '\n')

                    torch.save(policy_dqn.state_dict(), self.MODEL_FILE)
                    best_reward = episode_reward


                # Update graph every x seconds
                current_time = datetime.now()
                if current_time - last_graph_update_time > timedelta(seconds=10):
                    self.save_graph(rewards_per_episode , episodes_lengths)
                    last_graph_update_time = current_time

                # If enough experience has been collected
                if len(memory)>self.mini_batch_size:
                    mini_batch = memory.sample(self.mini_batch_size)
                    self.optimize(mini_batch, policy_dqn, target_dqn)

                    # Decay epsilon
                    epsilon = max(epsilon * self.epsilon_decay, self.epsilon_min)
                    epsilon_history.append(epsilon)

                    # Copy policy network to target network after a certain number of steps
                    if step_count > self.network_sync_rate:
                        target_dqn.load_state_dict(policy_dqn.state_dict())
                        step_count=0


    def save_graph(self, rewards_per_episode, episodes_lengths):
        # Save plots
        fig = plt.figure(1)

        # Plot average rewards (Y-axis) vs episodes (X-axis)
        mean_rewards = np.zeros(len(rewards_per_episode))
        for x in range(len(mean_rewards)):
            mean_rewards[x] = np.mean(rewards_per_episode[max(0, x-30):(x+1)])
        plt.subplot(121) 
        plt.xlabel('Episodes')
        plt.ylabel('Mean Rewards')
        plt.plot(mean_rewards)

        # Plot average num_pellets_remaining (Y-axis) vs episodes (X-axis)
        mean_episodes_lengths = np.zeros(len(episodes_lengths))
        for x in range(len(mean_episodes_lengths)):
            mean_episodes_lengths[x] = np.mean(episodes_lengths[max(0, x-30):(x+1)])
        plt.subplot(122) 
        plt.xlabel('Episodes')
        plt.ylabel('AVG episodes length')
        plt.plot(mean_episodes_lengths)

        plt.subplots_adjust(wspace=1.0, hspace=1.0)

        # Save plots
        fig.savefig(self.GRAPH_FILE)
        plt.close(fig)


    # Optimize policy network
    def optimize(self, mini_batch, policy_dqn, target_dqn):

        # Transpose the list of experiences and separate each element
        states, actions, new_states, rewards, terminations = zip(*mini_batch)

        # Stack tensors to create batch tensors
        # tensor([[1,2,3]])
        states = torch.stack(states)

        actions = torch.stack(actions)

        new_states = torch.stack(new_states)

        rewards = torch.stack(rewards)
        terminations = torch.tensor(terminations).float().to(device)

        with torch.no_grad():
            if self.enable_double_dqn:
                best_actions_from_policy = policy_dqn(new_states).argmax(dim=1)

                target_q = rewards + (1-terminations) * self.discount_factor_g * \
                                target_dqn(new_states).gather(dim=1, index=best_actions_from_policy.unsqueeze(dim=1)).squeeze()
            else:
                # Calculate target Q values (expected returns)
                target_q = rewards + (1-terminations) * self.discount_factor_g * target_dqn(new_states).max(dim=1)[0]
                '''
                    target_dqn(new_states)  ==> tensor([[1,2,3],[4,5,6]])
                        .max(dim=1)         ==> torch.return_types.max(values=tensor([3,6]), indices=tensor([3, 0, 0, 1]))
                            [0]             ==> tensor([3,6])
                '''

        # Calcuate Q values from current policy
        current_q = policy_dqn(states).gather(dim=1, index=actions.unsqueeze(dim=1)).squeeze()
        '''
            policy_dqn(states)  ==> tensor([[1,2,3],[4,5,6]])
                actions.unsqueeze(dim=1)
                .gather(1, actions.unsqueeze(dim=1))  ==>
                    .squeeze()                    ==>
        '''

        # Compute loss
        loss = self.loss_fn(current_q, target_q)

        # Optimize the model (backpropagation)
        self.optimizer.zero_grad()  # Clear gradients
        loss.backward()             # Compute gradients
        self.optimizer.step()       # Update network parameters i.e. weights and biases

if __name__ == '__main__':
    # Parse command line inputs
    # parser = argparse.ArgumentParser(description='Train or test model.')
    # parser.add_argument('hyperparameters', help='')
    # parser.add_argument('--train', help='Training mode', action='store_true')
    # args = parser.parse_args()

    #dql = Agent(hyperparameter_set=args.hyperparameters)
    dql = Agent("pacman")
    dql.run(is_training=True, render=False)
    # if args.train:
    #     dql.run(is_training=True)
    # else:
    #     dql.run(is_training=False, render=False)