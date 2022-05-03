import torch
import torch.nn as nn
from torch.distributions import Normal
import random
import numpy as np

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Actor(nn.Module):
    def __init__(self, config):
        super(Actor, self).__init__()
        self.seed = config.get('seed', random.randint(1,10000000))
        # ---------- log_std_min, log_std_max分别表示变量方差的最小最大的log值 -------
        self.log_std_min = config.get('log_std_min', -20)
        self.log_std_max = config.get('log_std_max', 2)
        self.state_size = config['state_dim']
        self.action_size = config['action_dim']
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

    def get_action(self, state):
        mu, log_std = self.forward(state)
        std = log_std.exp()
        dist = Normal(mu, std)
        # -------- rsample表示先从标准0,1正态上采样,然后将采样值作mean + std * 采样值的操作 
        e = dist.rsample()
        # -------- 这个tanh主要是因为动作是-1~1之间 -------
        action = torch.tanh(e)
        return action

    def get_det_action(self, state):
        mu, log_std = self.forward(state)
        return mu

    def evaluate(self, state, epsilon=1e-6):
        # ---------- 这个函数在训练时候启用,用来计算下一个状态的动作,以及出现的概率 --------
        mu, log_std = self.forward(state)
        std = log_std.exp()
        dist = Normal(mu, std)
        e = dist.rsample()
        action = torch.tanh(e)
        log_prob = (dist.log_prob(e) - torch.log(1 - action.pow(2) + epsilon)).sum(1, keepdim=True)

        return action, log_prob

class Critic(nn.Module):
    def __init__(self, config):
        super(Critic, self).__init__()
        self.state_size = config['state_dim']
        self.action_size = config['action_dim']
        self.hidden_size = config.get('hidden_size', 256)
        self.fc1 = nn.Linear(self.state_size + self.action_size, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc3 = nn.Linear(self.hidden_size, 1)
    
    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

    
def init_policy_net(config):
    model = Actor(config)
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.xavier_uniform_(m.weight)
    return model


def init_critic_net(config):
    model = Critic(config)
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.xavier_uniform_(m.weight)
    return model