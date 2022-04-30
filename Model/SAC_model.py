import torch
import torch.nn as nn
from torch.distributions import Normal
import random
class Actor(nn.Module):
    def __init__(self, config):
        super(Actor, self).__init__()
        self.seed = config.get('seed', random.randint(10000000))
        # ---------- log_std_min, log_std_max分别表示变量方差的最小最大的log值 -------
        self.log_std_min = config.get('log_std_min', -20)
        self.log_std_max = config.get('log_std_max', 2)
        self.state_size = config['state_size']
        self.action_size = config['action_size']
        self.hidden_size = config.get('hidden_size', 256)
        self.fc1 = nn.Linear(self.state_size, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size)

        self.mu = nn.Linear(self.hidden_size, self.action_size)
        self.log_std_linear = nn.Linear(self.hidden_size, self.action_size)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        mu = self.mu(x)
        log_std = self.log_std_linear(x)
        clamped_log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mu, clamped_log_std

    