import torch as T
import torch.nn.functional as F
from networks import ActorNetwork, CriticNetwork
# from constants import *
import torch.optim as optim
import numpy as np
import pygame
import vector
from buffer import MultiAgentReplayBuffer

class Agent:
    def __init__(self, actor_dims, critic_dims, n_actions, n_agents, alpha=0.01 , beta=0.01 , gamma=0.95 , tau=0.01 , device= 'cuda' ,chkpt_dir="tmp/maddpg"):
        self.gamma = gamma
        self.tau = tau
        self.n_actions = n_actions
        # self.n_agents = n_agents
        self.agent_name = 'agent_' + str(n_agents)
        
        
        
        self.actor = ActorNetwork(alpha, actor_dims, n_actions, name=self.agent_name+'_actor').to(device)
        self.critic = CriticNetwork(beta, critic_dims, n_agents, n_actions, name=self.agent_name+'_critic').to(device)
        self.target_actor = ActorNetwork(alpha, actor_dims, n_actions, name=self.agent_name+'_target_actor').to(device)
        self.target_critic = CriticNetwork(beta, critic_dims, n_agents, n_actions, name=self.agent_name+'_target_critic', chkpt_dir=chkpt_dir).to(device)

                
        self.update_network_paramters(tau=1)
        
        self.actor_loss = 0
        self.critic_loss = 0

        
    
    def update_network_paramters(self, tau=None):
        if tau is None:
            tau = self.tau
            
        target_actor_params = self.target_actor.named_parameters()    
        actor_params = self.actor.named_parameters()
        
        target_actor_state_dict = self.target_actor.state_dict()
        actor_state_dict = dict(actor_params)
        
        #####
        for name in actor_state_dict:  
            actor_state_dict[name] = tau*actor_state_dict[name].clone() + (1-tau)*target_actor_state_dict[name].clone()    
        
        self.target_actor.load_state_dict(actor_state_dict)
        
        
        #####
        target_critic_params = self.target_critic.named_parameters()    
        critic_params = self.critic.named_parameters()
        
        target_critic_state_dict = self.target_critic.state_dict()
        critic_state_dict = dict(critic_params)
        
        
        for name in critic_state_dict:  
            critic_state_dict[name] = tau*critic_state_dict[name].clone() + (1-tau)*target_critic_state_dict[name].clone()    
        
        self.target_critic.load_state_dict(critic_state_dict)
        

    def choose_action(self, observations, noise_scale=0.1):
        """
        Choose an action based on the current policy and add exploration noise.
        Args:
            observation (array-like): The current observation/state of the agent.
            noise_scale (float): The scale of the noise to be added for exploration.
        Returns:
            np.array: The chosen action.
        """

        state = T.tensor(observations, dtype=T.float).to(self.actor.device)
        actions = self.actor.forward(state)
        noise = T.normal(mean=0, std=noise_scale, size=actions.shape).to(self.actor.device)
        action = actions + noise

        # Clip the action to be within valid bounds (assuming [-1, 1])
        action = T.clamp(action, -1, 1)
        device = 'cuda'
        actions = action.to(self.actor.device)
        return actions


        
    def save_models(self):
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.target_critic.save_checkpoint()

    def load_models(self):
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.target_critic.load_checkpoint()        
        
        
class MADDPG:
    def __init__(self, actor_dims, critic_dims, n_agents, n_actions, possible_agents, env, alpha=0.01, beta=0.01, gamma=0.99, tau=0.01, chkpt_dir="tmp/maddpg/", device='cuda'):
        self.agents = []
        self.env = env
        self.possible_agents = possible_agents
        self.n_agents = n_agents
        self.n_actions = n_actions
        
        
        for agent_id, agent in enumerate(self.possible_agents):  # Use enumerate to get the integer index
            self.agents.append(Agent(actor_dims[agent_id], critic_dims, n_actions, agent_id, alpha=alpha, beta=beta, chkpt_dir=chkpt_dir))
          
            
    def save_checkpoint(self):
        for agent in self.agents:
            agent.save_models()
            
    def load_checkpoint(self):
        for agent in self.agents:
            agent.load_models()


    # def choose_action(self, observation):
    #     actions = []
    #     for agent_id , agent in enumerate(self.agents):
    #         action = agent.choose_action(observation[agent_id])
    #         actions.append(action)
    #     return actions
    
    def choose_action(self, raw_obs, noise_scale=0.1):
        """
        Choose actions for all agents based on their respective observations and add exploration noise.

        Parameters:
        - raw_obs (2D array): A 2D array where each row corresponds to the observation of an agent.
        - noise_scale (float): The scale of the noise to be added for exploration.

        Returns:
        - actions (dict): A dictionary containing actions for all agents, where keys are agent names.
        """
        actions = {}

        # Ensure raw_obs is a 2D array with enough rows for all agents
        if len(raw_obs) != len(self.agents):  # len(raw_obs) gives the number of rows (agents)
            raise IndexError(f"Expected {len(self.agents)} observations, but raw_obs has {len(raw_obs)} rows.")

        # Loop over the agents and choose actions based on their observations
        for idx, agent in enumerate(self.agents):
            # Get the observation for the agent (raw_obs[idx] gives the observation for the agent)
            observation = raw_obs[idx]  # This is a 1D array representing the observation of the agent

            # Choose action for the agent
            action = agent.choose_action(observation, noise_scale)
            agent_name = self.possible_agents[idx]  # Get the agent's name (e.g., 'pacman', 'ghost')
            actions[agent_name] = action.to(agent.actor.device)

        return actions


            
        

    def learn(self, memory):
        if not memory.ready():
            return
        
        # Fetch data from replay buffer
        batch, actor_batch = memory.sample_buffer()


        # Extract critic data
        states = T.tensor(batch["state"], dtype=T.float).to(self.agents[0].actor.device)
        actions = batch["action"]
        # rewards = T.tensor(batch["reward"], dtype=T.float).to(self.agents[0].actor.device)
        # Assuming `batch["reward"]` is a dictionary with rewards for each agent.
        # Convert the rewards for each agent into a list or array
        # Debugging: print the structure of the batch to see if 'reward' is correct
        print("Batch structure before learning:")
        print(batch)  # Inspect the entire batch structure
        print("Reward field in batch:", batch.get("reward"))  # Print out rewards field specifically
        
        # Debugging: Ensure rewards are a dictionary with the correct agent names
        for agent in self.possible_agents:
            print(f"Reward for {agent}: {batch['reward'].get(agent, 'No reward for this agent')}")
            
        rewards = T.tensor([batch["reward"][agent] for agent in self.possible_agents], dtype=T.float).to(self.agents[0].actor.device)
        
        states_ = T.tensor(batch["next_state"], dtype=T.float).to(self.agents[0].actor.device)
        dones = T.tensor(batch["done"], dtype=T.float).to(self.agents[0].actor.device)

        
        
        # Extract actor data
        actor_states = {agent: T.tensor(actor_batch[agent]["state"], dtype=T.float).to(self.agents[0].actor.device) for agent in actor_batch}
        actor_new_states = {agent: T.tensor(actor_batch[agent]["next_state"], dtype=T.float).to(self.agents[0].actor.device) for agent in actor_batch}
        
        
        device = self.agents[0].actor.device
        
        states = T.tensor(states, dtype=T.float).to(device)
        actions = T.tensor(actions, dtype=T.float).to(device)
        rewards = T.tensor(rewards).to(device)
        states_ = T.tensor(states_, dtype=T.float).to(device)
        dones = T.tensor(dones).to(device)
        

        
        all_agents_new_actions = []
        all_agents_new_mu_actions = []
        old_agents_actions = []

        for agent_name, agent in zip(self.possible_agents, self.agents):
            if agent_name in actor_new_states:
                new_states = T.tensor(actor_new_states[agent_name], dtype=T.float).unsqueeze(1).to(device)
                new_pi = agent.target_actor.forward(new_states)
                all_agents_new_mu_actions.append(new_pi)

                mu_states = T.tensor(actor_states[agent_name], dtype=T.float).unsqueeze(1).to(device)
                pi = agent.actor.forward(mu_states)
                all_agents_new_actions.append(pi)

                old_agents_actions.append(actions[agent_name].unsqueeze(1))


            
        new_actions = T.cat([acts for acts in all_agents_new_actions], dim=1)
        mu = T.cat([acts for acts in all_agents_new_mu_actions], dim=1)
        old_actions = T.cat([acts for acts in old_agents_actions], dim=1)
        
        # Now, for each agent, compute the critic loss
        for i, agent in enumerate(self.agents):
            critic_value_ = agent.target_critic.forward(T.cat([states_, new_actions], dim=-1)).flatten()
            critic_value = agent.critic.forward(T.cat([states, old_actions], dim=-1)).flatten()

            critic_value_[dones[:, 0]] = 0.0
            
            target = rewards[:, i] + agent.gamma * critic_value_
            critic_loss = F.mse_loss(target, critic_value)

            agent.critic.optimizer.zero_grad()
            critic_loss.backward(retain_graph=True)
            agent.critic.optimizer.step()

            print(f"critic_value_: {critic_value_}")
            print(f"critic_value: {critic_value}")
            print(f"critic_loss: {critic_loss.item()}")

            # Now calculate the actor loss
            actor_loss = agent.critic.forward(states, mu).flatten()
            actor_loss = -T.mean(actor_loss)
            agent.actor.optimizer.zero_grad()
            actor_loss.backward(retain_graph=True)
            agent.actor.optimizer.step()

            # Store actor loss in the agent
            agent.actor_loss = actor_loss.item()  # Store the actor loss for later use or tracking
            print(f"Actor loss: {actor_loss.item()}")

            agent.update_network_paramters()
