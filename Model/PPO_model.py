import torch
import torch.nn as nn
from torch.distributions import Normal


class Actor(nn.Module):
    def __init__(self, config):
        super(Actor, self).__init__()
        # self.seed = config.get('seed', random.randint(1,10000000))
        # torch.manual_seed(self.seed)
        # torch.cuda.manual_seed(self.seed)
        # ---------- log_std_min, log_std_max分别表示变量方差的最小最大的log值 -------
        self.log_std_min = config.get('log_std_min', -20)
        self.log_std_max = config.get('log_std_max', 2)
        self.state_size = config['state_dim']
        self.action_size = config['action_dim']
        # self.hidden_size = config.get('hidden_size', 256)
        self.hidden_size  = 128
        self.fc1 = nn.Linear(self.state_size, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, 2* self.hidden_size)
        self.mu = nn.Linear(2* self.hidden_size, self.action_size)
        self.log_std_linear = nn.Linear(2 * self.hidden_size, self.action_size)
        

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        mu = self.mu(x)
        log_std = self.log_std_linear(x)
        clamped_log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mu, clamped_log_std

    def get_action(self, state, epsilon=1e-6):
        mu, log_std = self.forward(state)
        std = log_std.exp()
        dist = Normal(mu, std)
        # -------- rsample表示先从标准0,1正态上采样,然后将采样值作mean + std * 采样值的操作 
        e = dist.rsample()
        # -------- 这个sigmoid主要是因为动作是0~1之间 -------
        action = torch.clamp(e, -1, 1)
        log_prob = torch.sum(dist.log_prob(e),-1).unsqueeze(-1)
        # action = torch.tanh(e)
        # log_prob = (dist.log_prob(e) - torch.log(1 - action.pow(2) + epsilon)).sum(1, keepdim=True)
        return action, log_prob

    def get_det_action(self, state):
        mu, log_std = self.forward(state)
        action = torch.clamp(mu, -1, 1)
        # action = torch.tanh(mu)
        return action

    def evaluate(self, state, action, epsilon=1e-6):
        # ---------- 这个函数在训练时候启用,用来计算在给定状态，动作的条件下，出现的概率 --------
        mu, log_std = self.forward(state)
        std = log_std.exp()
        dist = Normal(mu, std)
        # -------- 需要通过计算反sigmoid函数，得到采样的值 -------
        # sample_data = torch.atanh(action)
        # # -------- 计算采用的噪声向量 --------
        # noise_vector = (sample_data-mu) / std
        # # -------- 重新计算e，action，这次是要带梯度的 --------
        # e = mu + std * noise_vector.data
        # action_with_graph = torch.tanh(e)
        # # -------- 然后再计算出条件概率 ----------
        # log_prob = (dist.log_prob(e) - torch.log(1 - action_with_graph.pow(2) + epsilon)).sum(1, keepdim=True)
        log_prob = dist.log_prob(action)
        # entropy = dist.entropy()
        return torch.sum(log_prob, -1).unsqueeze(-1)


class Critic(nn.Module):
    def __init__(self, config):
        super(Critic, self).__init__()
        self.state_size = config['state_dim']
        self.hidden_size = config.get('hidden_size', 256)
        self.fc1 = nn.Linear(self.state_size, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc3 = nn.Linear(self.hidden_size, 1)
    
    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

    
def init_policy_net(config):
    model = Actor(config)
    # for m in model.modules():
    #     if isinstance(m, (nn.Conv2d, nn.Linear)):
    #         nn.init.xavier_uniform_(m.weight)
    return model


def init_critic_net(config):
    model = Critic(config)
    # for m in model.modules():
    #     if isinstance(m, (nn.Conv2d, nn.Linear)):
    #         nn.init.xavier_uniform_(m.weight)
    return model