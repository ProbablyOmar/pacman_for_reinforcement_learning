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

        input_dims = 31*28
        self.n_agents = n_agents
        self.n_actions = n_actions
        
        self.fc1 = nn.Linear(870, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)
        self.relu = nn.ReLU()

        self.optimizer = optim.Adam(self.parameters(), lr=beta)

        # Device setup
        self.device = T.device('cuda' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

        # Save/load path
        self.chkpt_dir = chkpt_dir
        self.chkpt_file = os.path.join(self.chkpt_dir, f"{name}_critic.pth")

    def forward(self, states, actions):
        # Debugging: Print the shapes before concatenation
        print(f"State shape: {states.shape}")
        print(f"Action shape: {actions.shape}")

        # Concatenate states and actions
        x = T.cat([states, actions], dim=1)

        # Debugging: Print the shape after concatenation
        print(f"Concatenated shape: {x.shape}")

        # Pass through the network layers

        x = self.relu(self.fc1(x))
        print(f"Shape before fc1: {x.shape}")
        x = self.relu(self.fc2(x))
        q_value = self.fc3(x)

        return q_value

    

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
        
        input_dims = 32*28
        
        self.device = T.device(device if T.cuda.is_available() else 'cpu')
        self.to(self.device)
        
        self.fc1 = nn.Linear(868, 256)  
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
         # Debugging: Print the shape of the state
        # print(f"State shape before flattening: {state.shape}")

        # Reshape state to match input size for the fully connected layer
        state = state.view(state.size(0), -1)  # Flatten the state to (batch_size, 868)

        # Debugging: Print the shape after flattening
        # print(f"State shape after flattening: {state.shape}")
        x = self.relu(self.fc1(state))
        x = self.relu(self.fc2(x))
        actions = F.softmax(self.fc3(x), dim=-1) 
        return actions
    
    
    def save_checkpoint(self):
        # print(f"Saving checkpoint to {self.chkpt_file}...")
        T.save(self.state_dict(), self.chkpt_file)

    def load_checkpoint(self):
        # print(f"Loading checkpoint from {self.chkpt_file}...")
        self.load_state_dict(T.load(self.chkpt_file))
