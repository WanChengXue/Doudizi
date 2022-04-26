import ptan
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
HID_SIZE = 128


class ModelA2C(nn.Module):
    def __init__(self, obs_size, act_size):
        super(ModelA2C, self).__init__()

        self.base = nn.Sequential(
            nn.Linear(obs_size, HID_SIZE),
            nn.ReLU(),
        )
        self.mu = nn.Sequential(
            nn.Linear(HID_SIZE, act_size),
            nn.Tanh(),
        )
        self.var = nn.Sequential(
            nn.Linear(HID_SIZE, act_size),
            nn.Softplus(),
        )
        self.value = nn.Linear(HID_SIZE, 1)

    def forward(self, x):
        base_out = self.base(x)
        return self.mu(base_out), self.var(base_out), \
               self.value(base_out)


class DDPGActor(nn.Module):
    def __init__(self, obs_size, act_size):
        super(DDPGActor, self).__init__()
        self.observation_dim = obs_size
        self.action_dim = act_size
        self.linear_layer_one = nn.Linear(self.observation_dim, 400)
        self.linear_layer_two = nn.Linear(400, 300)
        self.linear_layer_three = nn.Linear(300, self.action_dim)
    
    def forward(self, input_data):
        output_one = torch.relu(self.linear_layer_one(input_data))
        output_two = torch.relu(self.linear_layer_two(output_one))
        output_three = torch.tanh(self.linear_layer_three(output_two))
        return output_three


class DDPGCritic(nn.Module):
    def __init__(self, obs_size, act_size):
        super(DDPGCritic, self).__init__()

        self.obs_net = nn.Sequential(
            nn.Linear(obs_size, 400),
            nn.ReLU(),
        )

        self.out_net = nn.Sequential(
            nn.Linear(400 + act_size, 300),
            nn.ReLU(),
            nn.Linear(300, 1)
        )

    def forward(self, x, a):
        obs = self.obs_net(x)
        return self.out_net(torch.cat([obs, a], dim=1))


class D4PGCritic(nn.Module):
    def __init__(self, obs_size, act_size,
                 n_atoms, v_min, v_max):
        super(D4PGCritic, self).__init__()

        self.action_dim = act_size
        self.state_dim = obs_size
        self.n_atoms = n_atoms
        self.value_min = v_min
        self.value_max = v_max
        self.atoms_array = Parameter(torch.linspace(self.value_min, self.value_max, self.n_atoms), requires_grad=False)
        self.state_affine_layer = nn.Linear(self.state_dim, 400)
        # self.action_affine_layer = nn.Linear(self.action_dim, self.hidden_dim)
        self.hidden_affine_layer = nn.Linear(400 + self.action_dim, 300)
        self.output_affine_layer = nn.Linear(300, self.n_atoms)

    def distribution_to_value(self, prob_distribution):
        batch_size = prob_distribution.shape[0]
        extend_atoms_array = self.atoms_array.unsqueeze(0).repeat(batch_size, 1)
        state_value = torch.sum(prob_distribution * extend_atoms_array, -1).unsqueeze(-1)
        return state_value

    def forward(self, state, action):
        affine_state = torch.relu(self.state_affine_layer(state))
        # affine_action = torch.relu(self.action_affine_layer(action))
        affine_hidden = torch.relu(self.hidden_affine_layer(torch.cat([affine_state, action], -1)))
        # atom_weight = self.output_affine_layer(torch.cat([affine_state, affine_action], -1))
        atom_weight = self.output_affine_layer(affine_hidden)
        return torch.softmax(atom_weight, -1)


class AgentA2C(ptan.agent.BaseAgent):
    def __init__(self, net, device="cpu"):
        self.net = net
        self.device = device

    def __call__(self, states, agent_states):
        states_v = ptan.agent.float32_preprocessor(states)
        states_v = states_v.to(self.device)

        mu_v, var_v, _ = self.net(states_v)
        mu = mu_v.data.cpu().numpy()
        sigma = torch.sqrt(var_v).data.cpu().numpy()
        actions = np.random.normal(mu, sigma)
        actions = np.clip(actions, -1, 1)
        return actions, agent_states


class AgentDDPG(ptan.agent.BaseAgent):
    """
    Agent implementing Orstein-Uhlenbeck exploration process
    """
    def __init__(self, net, device="cpu", ou_enabled=True,
                 ou_mu=0.0, ou_teta=0.15, ou_sigma=0.2,
                 ou_epsilon=1.0):
        self.net = net
        self.device = device
        self.ou_enabled = ou_enabled
        self.ou_mu = ou_mu
        self.ou_teta = ou_teta
        self.ou_sigma = ou_sigma
        self.ou_epsilon = ou_epsilon

    def initial_state(self):
        return None

    def __call__(self, states, agent_states):
        states_v = ptan.agent.float32_preprocessor(states)
        states_v = states_v.to(self.device)
        mu_v = self.net(states_v)
        actions = mu_v.data.cpu().numpy()
        # actions的维度为 batch_size * action_dim
        if self.ou_enabled and self.ou_epsilon > 0:
            new_a_states = []
            for a_state, action in zip(agent_states, actions):
                if a_state is None: # a_state用0初始化 
                    a_state = np.zeros(
                        shape=action.shape, dtype=np.float32)
                a_state += self.ou_teta * (self.ou_mu - a_state) # ou_teta为0.15
                a_state += self.ou_sigma * np.random.normal( # ou_sigma为0.2
                    size=action.shape)
                # print(a_state)
                # a_state的修改为 ou_tehta * (mu - a_state) + ou_sigma * (长度为action_dim的正太噪声)
                action += self.ou_epsilon * a_state
                new_a_states.append(a_state)
        else:
            new_a_states = agent_states

        actions = np.clip(actions, -1, 1)
        return actions, new_a_states


class AgentD4PG(ptan.agent.BaseAgent):
    """
    Agent implementing noisy agent
    """
    def __init__(self, net, device="cpu", epsilon=0.3):
        self.net = net
        self.device = device
        self.epsilon = epsilon

    def __call__(self, states, agent_states):
        states_v = ptan.agent.float32_preprocessor(states)
        states_v = states_v.to(self.device)
        mu_v = self.net(states_v)
        actions = mu_v.data.cpu().numpy()
        actions += self.epsilon * np.random.normal(
            size=actions.shape)
        actions = np.clip(actions, -1, 1)
        return actions, agent_states
