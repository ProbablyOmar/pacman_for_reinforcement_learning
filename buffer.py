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
        # print("\n==== Storing Transition Debugging ====")
        
        for agent in raw_obs.keys():
            # print(f"Agent: {agent}")
            # print(f"  - State shape before flattening: {np.array(raw_obs[agent]).shape}")  
            # print(f"  - Next State shape before flattening: {np.array(raw_obs_[agent]).shape}")
            
            # Flatten state
            raw_obs[agent] = np.array(raw_obs[agent]).flatten()
            raw_obs_[agent] = np.array(raw_obs_[agent]).flatten()
            
            # print(f"  - State shape after flattening: {raw_obs[agent].shape}")  
            # print(f"  - Next State shape after flattening: {raw_obs_[agent].shape}")
            # print(f"  - Action stored: {action[agent]}")
            # print(f"  - Reward stored: {reward[agent]}")
        
        # print(f"Centralized State shape: {np.array(state).shape}")
        # print(f"Centralized Next State shape: {np.array(state_).shape}")

        # Store transition
        for agent, agent_done in zip(raw_obs.keys(), done):  
            self.actor_memory[agent]["state"].append(raw_obs[agent])  
            self.actor_memory[agent]["action"].append(action[agent])  
            self.actor_memory[agent]["reward"].append(reward[agent])  
            self.actor_memory[agent]["next_state"].append(raw_obs_[agent])  
            self.actor_memory[agent]["done"].append(agent_done)  

        self.critic_memory["state"].append(state)  
        self.critic_memory["action"].append(action)  
        self.critic_memory["reward"].append(reward)  
        self.critic_memory["next_state"].append(state_)  
        self.critic_memory["done"].append(done)  

        self.mem_cntr += 1

        if self.mem_cntr > self.mem_size:
            self.remove_oldest_transition()
        
        # print(f"Memory size after storing: {len(self.critic_memory['state'])}")



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
        batch_idx = np.random.choice(len(self.critic_memory["state"]), self.batch_size, replace=False)

        batch = {key: [self.critic_memory[key][idx] for idx in batch_idx] for key in self.critic_memory}
        actor_batch = {agent: {key: [self.actor_memory[agent][key][idx] for idx in batch_idx] 
                            for key in self.actor_memory[agent]} 
                    for agent in self.actor_memory}


        return batch, actor_batch


    def ready(self):
        """
        Check if the replay buffer has enough samples for training.
        """
        return self.mem_cntr >= self.batch_size
