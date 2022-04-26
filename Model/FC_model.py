import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

class Toy_model(nn.Module):
    def __init__(self, config):
        super(Toy_model, self).__init__()
        self.observation_dim = config['state_dim']
        self.action_dim = config['action_dim']
        self.linear_layer_one = nn.Linear(self.observation_dim, 400)
        self.linear_layer_two = nn.Linear(400, 300)
        self.linear_layer_three = nn.Linear(300, self.action_dim)
    
    def forward(self, input_data):
        output_one = torch.relu(self.linear_layer_one(input_data))
        output_two = torch.relu(self.linear_layer_two(output_one))
        output_three = torch.tanh(self.linear_layer_three(output_two))
        return output_three


class Toy_central_critic(nn.Module):
    def __init__(self, config):
        super(Toy_central_critic, self).__init__()
        self.concatenate_action_dim = config['action_dim']
        self.concatenate_state_dim = config['state_dim']
        self.hidden_dim = config['hidden_dim']
        self.concatenate_state_affine_layer = nn.Linear(self.concatenate_state_dim, self.hidden_dim)
        self.concatenate_action_affine_layer = nn.Linear(self.concatenate_action_dim, self.hidden_dim)
        self.output_layer = nn.Linear(2*self.hidden_dim, 1)

    def forward(self, centralized_state, action):
        state_output = torch.relu(self.concatenate_state_affine_layer(centralized_state))
        action_output = torch.relu(self.concatenate_action_affine_layer(action))
        concatenate_feature_map = torch.cat([state_output, action_output], -1)
        Q_value = self.output_layer(concatenate_feature_map)
        return Q_value

class Toy_critic_categorical(nn.Module):
    def __init__(self, config):
        # --------- 这个类是用来实现categorical critic ----------
        super(Toy_critic_categorical, self).__init__()
        self.action_dim = config['action_dim']
        self.state_dim = config['state_dim']
        self.hidden_dim = config['hidden_dim']
        self.n_atoms = config['n_atoms']
        self.value_min = config['value_min']
        self.value_max = config['value_max']
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



def init_policy_net(config):
    model = Toy_model(config)
    # for m in model.modules():
    #     if isinstance(m, (nn.Conv2d, nn.Linear)):
    #         nn.init.xavier_uniform_(m.weight)
    return model


def init_critic_net(config):
    model = Toy_central_critic(config)
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.xavier_uniform_(m.weight)
    return model

def init_categorical_critic_net(config):
    model = Toy_critic_categorical(config)
    # for m in model.modules():
    #     if isinstance(m, (nn.Conv2d, nn.Linear)):
    #         nn.init.xavier_uniform_(m.weight)
    return model