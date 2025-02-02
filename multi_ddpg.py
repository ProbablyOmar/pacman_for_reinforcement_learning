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
        
        self.gamma = gamma
        
        
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

        # Extract critic data from the batch
        states = [
            T.tensor(batch["state"][i], dtype=T.float).to(self.agents[0].actor.device)
            for i in range(len(self.possible_agents))
        ]
        # print(f"States shape: {[state.shape for state in states]}")
        
        actions = []
        for i, agent_name in enumerate(self.possible_agents):
            action = batch["action"][i]  # action is a dictionary {agent_name: action_value}
            # print(f"Actions: {action}")
            action_for_agent = action[agent_name]  # Extract the action for the current agent
            actions.append(T.tensor(action_for_agent, device=self.agents[0].actor.device))
        
        

        rewards = []
        for sample in batch["reward"]:  # Iterate over batch elements (each is a dict)
            reward_values = [sample[agent_name] for agent_name in self.possible_agents]  # Extract rewards
            rewards.append(T.tensor(reward_values, dtype=T.float).unsqueeze(0).to(self.agents[0].actor.device))  # Shape [1, num_agents]

        rewards = T.cat(rewards, dim=0)  # Stack into tensor of shape [batch_size, num_agents]
        # print(f"Rewards shape: {rewards.shape}")
        
        states_ = [
            T.tensor(batch["next_state"][i], dtype=T.float).to(self.agents[0].actor.device)
            for i in range(len(self.possible_agents))
        ]
        
        # print(f"Next States shape: {[state.shape for state in states_]}")

        dones = []
        for i, agent_name in enumerate(self.possible_agents):
            done = batch["done"][i]  # Done flag for the agent
            done = T.tensor(done, dtype=T.float).unsqueeze(0).to(self.agents[0].actor.device)  # Ensure it's a 1D tensor for each agent
            dones.append(done)

        states = T.stack(states)
        # print(f"Stacked states shape: {states.shape}")
        
        actions = T.stack(actions)
        # print("&&&&&&&&&&&&&&")
        # print(f"Actions shape: {actions.shape}")
        states_ = T.stack(states_)
        dones = T.stack(dones).view(-1, len(self.possible_agents))  # Ensure shape (1024, 2)
        dones = dones.squeeze(1)

        actions = actions.view(actions.size(0), -1) 
        states_with_actions = T.cat([states, actions], dim=-1)
        # print(f"States with actions shape: {states_with_actions.shape}")
        
        # Ensure actor states and new states are correctly extracted from the actor_batch
        actor_states = []
        actor_new_states = []

        for agent_name in self.possible_agents:
            actor_states.append(actor_batch[agent_name]["state"])
            actor_new_states.append(actor_batch[agent_name]["next_state"])

        device = self.agents[0].actor.device

        all_agents_new_actions = []
        all_agents_new_mu_actions = []
        old_agents_actions = []

        for agent_idx, agent in enumerate(self.agents):
            new_states = T.tensor(actor_new_states[agent_idx], dtype=T.float).to(device)

            # Ensure that new_pi is not empty or None
            new_pi = agent.target_actor.forward(new_states)
            # print(f"Agent {agent_idx}: new_pi shape: {new_pi.shape if new_pi is not None else 'None'}")

            if new_pi is not None:  # If new_pi is valid
                all_agents_new_actions.append(new_pi.detach())
            # else:
            #     print(f"Warning: new_pi is None for agent {agent_idx}")

            mu_states = T.tensor(actor_states[agent_idx], dtype=T.float).to(device)
            pi = agent.actor.forward(mu_states)
            all_agents_new_mu_actions.append(pi)

            if len(actions) > agent_idx:
                old_agents_actions.append(actions[agent_idx].unsqueeze(0).clone())
            else:
                old_agents_actions.append(T.zeros(1, dtype=T.long, device=self.agents[0].actor.device))

        # Now check if the list has anything to concatenate
        if all_agents_new_actions:
            new_actions = T.cat(all_agents_new_actions).clone()
        # else:
        #     print("Warning: all_agents_new_actions is empty!")

        # Proceed with the rest of your code...

        mu = T.cat(all_agents_new_mu_actions).clone()
        old_actions = T.cat(old_agents_actions).clone()

        # Update Critic and Actor for each agent
        for agent_idx, agent in enumerate(self.agents):
            # Compute target Q-values (Bellman equation)
            with T.no_grad():
                next_Q = agent.target_critic.forward(states_, new_actions).flatten()
                # print("******************************")
                # print("states_.shape:", states_.shape)
                # print("new_actions.shape:", new_actions.shape)


            target_Q = rewards[:, agent_idx].unsqueeze(-1).to(self.agents[0].actor.device)
            target_Q = target_Q + (self.gamma * next_Q * (1 - dones[:, agent_idx])) 
            target_Q = target_Q.clone()  # Detach target_Q to avoid backpropagation issues

            # Ensure current_Q and target_Q have matching shapes
            state_for_critic = states  # States need to be passed separately
            action_for_critic = actions  # Actions need to be passed separately

            # Pass state and action separately to the critic
            current_Q = agent.critic.forward(state_for_critic, action_for_critic).flatten()
            current_Q = current_Q.unsqueeze(0).expand_as(target_Q).clone()

            # Compute critic loss
            critic_loss = F.mse_loss(current_Q, target_Q)

            # Update critic
            agent.critic.optimizer.zero_grad()
            critic_loss.backward(retain_graph=True)
            agent.critic.optimizer.step()

            # Compute actor loss
            actor_loss = -agent.critic.forward(state_for_critic, action_for_critic).mean()  # Correctly calculate the actor loss

            agent.actor.optimizer.zero_grad()
            actor_loss.backward(retain_graph=True)
            T.autograd.set_detect_anomaly(True)
            agent.actor.optimizer.step()

            # Soft update of target networks
            agent.update_network_paramters()
