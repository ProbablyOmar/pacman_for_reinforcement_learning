import os
import numpy as np
import torch as T
import random

class MultiAgentReplayBuffer:
    def __init__(self, max_size, critic_dims, actor_dims, possible_agents, n_actions, batch_size):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.batch_size = batch_size
        self.n_actions = n_actions
        self.actor_dims = actor_dims
        self.critic_dims = critic_dims
        
        # Ensure possible_agents is a list of agent names (e.g., ["pacman", "ghost"])
        if isinstance(possible_agents, int):
            # Generate agent names like "agent0", "agent1", ...
            self.possible_agents = [f"agent{i}" for i in range(possible_agents)]
        else:
            self.possible_agents = possible_agents

        # Create memory buffers for each agent
        self.state_memory = np.zeros((self.mem_size, critic_dims))
        self.new_state_memory = np.zeros((self.mem_size, critic_dims))
        self.reward_memory = {agent: np.zeros(self.mem_size) for agent in self.possible_agents}
        self.terminal_memory = {agent: np.zeros(self.mem_size, dtype=bool) for agent in self.possible_agents}
        
        self.actor_state_memory = {agent: np.zeros((self.mem_size, actor_dims[agent])) for agent in self.possible_agents}
        self.actor_new_state_memory = {agent: np.zeros((self.mem_size, actor_dims[agent])) for agent in self.possible_agents}
        self.actor_action_memory = {agent: np.zeros((self.mem_size, n_actions)) for agent in self.possible_agents}


    def store_transition(self, raw_obs, state, action, reward, raw_obs_, state_, done):
        """
        Store transitions into replay memory.
        :param raw_obs: Dict of current observations for all agents (local for actors).
        :param state: Centralized joint state for the critic.
        :param action: Dict of actions taken by each agent.
        :param reward: Dict of rewards received by each agent.
        :param raw_obs_: Dict of next observations for all agents (local for actors).
        :param state_: Centralized next joint state for the critic.
        :param done: Dict of terminal flags for each agent.
        """
        index = self.mem_cntr % self.mem_size
        
        for agent in self.possible_agents:
            self.actor_state_memory[agent][index] = raw_obs[agent]
            self.actor_new_state_memory[agent][index] = raw_obs_[agent]
            self.actor_action_memory[agent][index] = action[agent]
            self.reward_memory[agent][index] = reward[agent]
            self.terminal_memory[agent][index] = done[agent]

        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.mem_cntr += 1

    def sample_buffer(self):
        """
        Sample a batch of experiences from the replay buffer.
        """
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, self.batch_size, replace=False)

        states = self.state_memory[batch]
        new_states = self.new_state_memory[batch]
        rewards = {agent: self.reward_memory[agent][batch] for agent in self.possible_agents}
        dones = {agent: self.terminal_memory[agent][batch] for agent in self.possible_agents}
        
        actor_states = {agent: self.actor_state_memory[agent][batch] for agent in self.possible_agents}
        actor_new_states = {agent: self.actor_new_state_memory[agent][batch] for agent in self.possible_agents}
        actions = {agent: self.actor_action_memory[agent][batch] for agent in self.possible_agents}

        return actor_states, states, actions, rewards, actor_new_states, new_states, dones

    def ready(self):
        """
        Check if the replay buffer has enough samples for training.
        """
        return self.mem_cntr >= self.batch_size
