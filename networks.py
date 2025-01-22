import torch as T
import torch.nn as nn
import torch.nn.functional as F
import os
import torch.optim as optim
import numpy as np
from constants import *


class CriticNetwork(nn.Module):
    """
    Critic Network for MADDPG (Multi-Agent Deep Deterministic Policy Gradient).

    Parameters:
        beta: learning rate
        input_dims: tuple (height, width) of the input dimensions
        n_agents: number of agents
        n_actions: number of actions
        name: name of the agent network (used for saving the model)
        chkpt_dir: directory to save/load the model
    """
    def __init__(self, beta, input_dims, n_agents, n_actions, name, chkpt_dir="tmp/maddpg"):
        super(CriticNetwork, self).__init__()
        
        input_dims = (GAME_ROWS, GAME_COLS)
        self.height, self.width = input_dims
        
        self.n_agents = n_agents
        self.n_actions = n_actions
        ###our network layers
        #####################
        self.fc1 = nn.Linear(self.height * self.width + n_agents * n_actions, 256)  
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)
        self.relu = nn.ReLU()
        
        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        
        self.device = T.device('cuda' if T.cuda.is_available() else 'cpu')
        self.to(self.device)
        
        
        self.chkpt_dir = chkpt_dir
        self.chkpt_file = os.path.join(self.chkpt_dir, f"{name}_critic.pth")
        

    def forward(self, states, actions):
        """
        Forward propagation through the network.

        Parameters:
            state: input state tensor
            actions: input actions tensor

        Returns:
            Q-value: scalar value representing the Q-value for the given state-action pair
        """
        state = state.view(state.size(0), -1)  # Flatten the state input
        x = T.cat([state, actions], dim=1)  # Concatenate state and actions
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        q = self.fc3(x)
        return q
    

    def save_checkpoint(self):
        print(f"Saving checkpoint to {self.chkpt_file}...")
        T.save(self.state_dict(), self.chkpt_file)

    def load_checkpoint(self):
        print(f"Loading checkpoint from {self.chkpt_file}...")
        self.load_state_dict(T.load(self.chkpt_file))


class ActorNetwork(nn.Module):
    """
    Actor Network for MADDPG (Multi-Agent Deep Deterministic Policy Gradient).

    Parameters:
        alpha: learning rate
        input_dims: tuple (height, width) of the input dimensions
        n_actions: number of actions
        name: name of the agent network (used for saving the model)
        chkpt_dir: directory to save/load the model
    """
    def __init__(self, alpha, input_dims, n_actions, name, chkpt_dir="tmp/maddpg", device='cuda'):
        super(ActorNetwork, self).__init__()
        
        input_dims = (GAME_ROWS, GAME_COLS)
        self.height, self.width = input_dims
        
        self.device = T.device(device if T.cuda.is_available() else 'cpu')
        self.to(self.device)
        
        self.fc1 = nn.Linear(self.height * self.width, 256)  
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, n_actions)
        self.relu = nn.ReLU()
        
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        
        self.chkpt_dir = chkpt_dir
        self.chkpt_file = os.path.join(self.chkpt_dir, f"{name}_actor.pth")
        
        self.device = T.device('cuda' if T.cuda.is_available() else 'cpu')
        self.to(self.device)
        

    def forward(self, state):
        """
        Forward propagation through the network.

        Parameters:
            state: input state tensor

        Returns:
            actions: tensor representing the probabilities of actions
        """
        state = state.view(state.size(0), -1)  # Flatten the state input
        x = self.relu(self.fc1(state))
        x = self.relu(self.fc2(x))
        actions = self.softmax(self.fc3(x))  
        return actions
    
    
    def save_checkpoint(self):
        print(f"Saving checkpoint to {self.chkpt_file}...")
        T.save(self.state_dict(), self.chkpt_file)

    def load_checkpoint(self):
        print(f"Loading checkpoint from {self.chkpt_file}...")
        self.load_state_dict(T.load(self.chkpt_file))

