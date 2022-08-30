import torch 
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, seed):
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        
        self.fc1 = torch.nn.Linear(state_size, 64) 
        self.fc2 = torch.nn.Linear(64, 64)
        self.fc3 = torch.nn.Linear(64, action_size)        
    
    def forward(self, state):
        """Pass the given state through the network."""
        x = self.fc1(state)
        x = F.relu(x) 
        
        x = self.fc2(x)
        x = F.relu(x)  
        
        return self.fc3(x) 