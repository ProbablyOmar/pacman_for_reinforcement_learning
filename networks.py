import torch as T
import torch.nn as nn
import torch.nn.functional as F
import os
import torch.optim as optim

class CriticNetwork(nn.Module):
    def __init__(self, beta, input_dims, n_agents, n_actions, name, chkpt_dir="tmp/maddpg"):
        super(CriticNetwork, self).__init__()
        
        self.fc1 = nn.Linear(1741, 256)  # Concatenated state-action size
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)  # Output Q-value
        
        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = T.device('cuda' if T.cuda.is_available() else 'cpu')
        self.to(self.device)
        self.chkpt_dir = chkpt_dir
        self.chkpt_file = os.path.join(self.chkpt_dir, f"{name}_critic.pth")

    def forward(self, states, actions):
        # Ensure actions match the state batch size
        actions = actions[:states.shape[0]].clone()  # Ensure actions match state batch size
        
        # Concatenate states and actions along the last dimension
        x = T.cat([states, actions], dim=-1)  # Concatenate state and action
        # print(f"Shape of input to critic (states + actions): {x.shape}")  # Debugging print
        
        # # Ensure the total size of input is 1746
        # assert x.shape[1] == 1746, f"Expected input size 1746, but got {x.shape[1]}"
        
        # Forward pass through the network
        x = F.relu(self.fc1(x))  # First hidden layer
        x = F.relu(self.fc2(x))  # Second hidden layer
        q_value = self.fc3(x)  # Output Q-value
        
        return q_value
    
    def save_checkpoint(self):
        T.save(self.state_dict(), self.chkpt_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.chkpt_file))


class ActorNetwork(nn.Module):
    def __init__(self, alpha, input_dims, n_actions, name, chkpt_dir="tmp/maddpg", device='cuda'):
        super(ActorNetwork, self).__init__()
        
        self.fc1 = nn.Linear(868, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, n_actions)
        
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device(device if T.cuda.is_available() else 'cpu')
        self.to(self.device)
        self.chkpt_dir = chkpt_dir
        self.chkpt_file = os.path.join(self.chkpt_dir, f"{name}_actor.pth")

    def forward(self, state):
        if len(state.shape) == 1:  # Ensure state has batch dimension
            state = state.unsqueeze(0)
        state = state.view(state.size(0), -1)  # Flatten if needed
        x = F.relu(self.fc1(state))  
        x = F.relu(self.fc2(x)) 
        actions = F.softmax(self.fc3(x), dim=-1)  
        return actions
    
    def save_checkpoint(self):
        T.save(self.state_dict(), self.chkpt_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.chkpt_file))
