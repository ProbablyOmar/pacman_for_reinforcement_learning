import os
import numpy as np
import torch as T
import random
from pygame.math import Vector2


class MultiAgentReplayBuffer:
    def __init__(self, max_size, critic_dims, actor_dims, n_agents, possible_agents, n_actions, batch_size, env):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.batch_size = batch_size
        self.n_actions = n_actions
        self.actor_dims = actor_dims
        self.critic_dims = critic_dims  
        self.n_agents = n_agents
        self.env = env

        # Initializing memory buffers for each agent
        self.actor_memory = {agent: {"state": [], "action": [], "reward": [], "next_state": [], "done": []}
                             for agent in possible_agents}
        self.critic_memory = {"state": [], "action": [], "reward": [], "next_state": [], "done": []}

    def init_actor_memory(self):
        """
        Initializes memory for each agent in the environment.
        """
        for agent in self.actor_memory:
            self.actor_memory[agent] = {"state": [], "action": [], "reward": [], "next_state": [], "done": []}

    def store_transition(self, raw_obs, state, action, reward, raw_obs_, state_, done):
        """
        Store transitions into replay memory.
        :param raw_obs: Dict of current observations for all agents (local for actors).
        :param state: Centralized joint state for the critic.
        :param action: Dict of actions taken by each agent.
        :param reward: Dict of rewards received by each agent.
        :param raw_obs_: Dict of next observations for all agents (local for actors).
        :param state_: Centralized next joint state for the critic.
        :param done: List of terminal flags for each agent (one flag per agent).
        """
        for agent, agent_done in zip(raw_obs.keys(), done):  # Iterate through agents and their corresponding done flags
            self.actor_memory[agent]["state"].append(raw_obs[agent])  # Storing the observations
            self.actor_memory[agent]["action"].append(action[agent])  # Storing the actions
            self.actor_memory[agent]["reward"].append(reward[agent])  # Storing the rewards
            self.actor_memory[agent]["next_state"].append(raw_obs_[agent])  # Storing the next state (next observations)
            self.actor_memory[agent]["done"].append(agent_done)  # Storing the done flag for each agent

        # Storing centralized data for critics
        self.critic_memory["state"].append(state)  # Storing the joint state for critic
        self.critic_memory["action"].append(action)  # Storing the joint actions
        self.critic_memory["reward"].append(reward)  # Storing the joint rewards
        self.critic_memory["next_state"].append(state_)  # Storing the next joint state for critic
        self.critic_memory["done"].append(done)  # Storing the done flags for critics

        # Increment memory counter
        self.mem_cntr += 1

        # If memory exceeds max size, remove the oldest transition
        if self.mem_cntr > self.mem_size:
            self.remove_oldest_transition()


    def remove_oldest_transition(self):
        """
        Removes the oldest transition from the replay buffer when the memory size exceeds the maximum.
        """
        for agent in self.actor_memory:
            self.actor_memory[agent]["state"].pop(0)
            self.actor_memory[agent]["action"].pop(0)
            self.actor_memory[agent]["reward"].pop(0)
            self.actor_memory[agent]["next_state"].pop(0)
            self.actor_memory[agent]["done"].pop(0)

        self.critic_memory["state"].pop(0)
        self.critic_memory["action"].pop(0)
        self.critic_memory["reward"].pop(0)
        self.critic_memory["next_state"].pop(0)
        self.critic_memory["done"].pop(0)

    def sample_buffer(self):
        """
        Sample a batch of experiences from the replay buffer.
        """
        # Sampling batch size from the memory buffer
        batch_idx = np.random.choice(len(self.critic_memory["state"]), self.batch_size, replace=False)

        # Create batch dictionaries
        batch = {}
        for key in self.critic_memory:
            batch[key] = [self.critic_memory[key][idx] for idx in batch_idx]

        # Sample actor memory for each agent
        actor_batch = {}
        for agent in self.actor_memory:
            actor_batch[agent] = {
                "state": [self.actor_memory[agent]["state"][idx] for idx in batch_idx],
                "action": [self.actor_memory[agent]["action"][idx] for idx in batch_idx],
                "reward": [self.actor_memory[agent]["reward"][idx] for idx in batch_idx],
                "next_state": [self.actor_memory[agent]["next_state"][idx] for idx in batch_idx],
                "done": [self.actor_memory[agent]["done"][idx] for idx in batch_idx],
            }

        return batch, actor_batch

    def ready(self):
        """
        Check if the replay buffer has enough samples for training.
        """
        return self.mem_cntr >= self.batch_size
