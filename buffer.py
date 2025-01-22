import os
import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
from collections import deque

class MultiAgentReplayBuffer:
    def __init__(self, max_size, critic_dims, actor_dims, n_agents, n_actions, batch_size):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.n_agents = n_agents
        self.batch_size = batch_size
        self.n_actions = n_actions
        self.actor_dims = actor_dims
        self.critic_dims = critic_dims
        
        self.state_memory = np.zeros((self.mem_size, critic_dims))
        self.new_state_memory = np.zeros((self.mem_size, critic_dims))
        self.reward_memory = np.zeros((self.mem_size, n_agents))
        self.terminal_memory = np.zeros((self.mem_size, n_agents), dtype=bool)
        
        self.init_actor_memory(actor_dims)
    
    
    def init_actor_memory(self, actor_dims):
        
        # Actor memory // local states,and actions
        self.actor_state_memory = []
        self.actor_new_state_memory = []
        self.actor_action_memory = []
    
        for i in range(self.n_agents):
            self.actor_state_memory.append(
                            np.zeros((self.mem_size, self.actor_dims[i])))
            self.actor_new_state_memory.append(
                            np.zeros((self.mem_size, self.actor_dims[i])))
            self.actor_action_memory.append(
                            np.zeros((self.mem_size, self.n_actions)))

    def store_transition(self, raw_obs, state, action, reward, raw_obs_, state_, done):
        """
        Store transitions into replay memory.
        :param raw_obs: List of current observations for all agents (local for actors).
        :param state: Centralized joint state for the critic.
        :param action: List of actions taken by each agent.
        :param reward: List of rewards received by each agent.
        :param raw_obs_: List of next observations for all agents (local for actors).
        :param state_: Centralized next joint state for the critic.
        :param done: List of terminal flags for each agent.
        """
        
        index = self.mem_cntr % self.mem_size
        #actor memory
        for agent in range(self.n_agents):
            self.actor_state_memory[agent][index] = raw_obs[agent]
            self.actor_new_state_memory[agent][index] = raw_obs_[agent]
            self.actor_action_memory[agent][index] = action[agent]

        #critic memory
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done
        self.mem_cntr += 1

    def sample_buffer(self):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, self.batch_size, replace=False)

        
        states = self.state_memory[batch]
        new_states = self.new_state_memory[batch]
        rewards = self.reward_memory[batch]
        dones = self.terminal_memory[batch]
        
        #states and action for actor
        actor_states = []
        actor_new_states = []
        actions = []
        for agent in range(self.n_agents):
            actor_states.append(self.actor_state_memory[agent][batch])
            actor_new_states.append(self.actor_new_state_memory[agent][batch])
            actions.append(self.actor_action_memory[agent][batch])

        return actor_states, states, actions, rewards, actor_new_states, states_, terminal

    def ready(self):
        if self.mem_cntr >= self.batch_size:
            return True