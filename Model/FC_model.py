"""
This file includes the torch models. We wrap the three
models into one class for convenience.
"""

import random
import torch
from torch import nn

# 模型传入的维度为动作 * 5 * 138
class LandlordLstmModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(138, 128, batch_first=True)
        self.dense1 = nn.Linear(260 + 128, 512)  # 388 * 512
        self.dense2 = nn.Linear(512, 512)
        self.dense3 = nn.Linear(512, 512)
        self.dense4 = nn.Linear(512, 512)
        self.dense5 = nn.Linear(512, 512)
        self.dense6 = nn.Linear(512, 1)

    # 这里输入的z，其实是action的数量作为batch size送入到网络内
    def forward(self, z, x, return_value=False, exp_epsilon=0.05):
        lstm_out, (h_n, _) = self.lstm(z)
        lstm_out = lstm_out[:, -1, :]  # 这个地方得到的是action_dim * 128的tensor
        x = torch.cat([lstm_out, x], dim=-1)
        x = self.dense1(x)
        x = torch.relu(x)
        x = self.dense2(x)
        x = torch.relu(x)
        x = self.dense3(x)
        x = torch.relu(x)
        x = self.dense4(x)
        x = torch.relu(x)
        x = self.dense5(x)
        x = torch.relu(x)
        x = self.dense6(x)
        if return_value:
            return x
        else:
            if random.random() < exp_epsilon:
                action = torch.randint(x.shape[0], (1,))[0]
            else:
                action = torch.argmax(x, dim=0)[0]
            mask = torch.zeros(x.shape[0])
            mask[action] = 1
            return action, mask


class FarmerLstmModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(138, 128, batch_first=True)
        self.dense1 = nn.Linear(309 + 128, 512)  # 437 * 512
        self.dense2 = nn.Linear(512, 512)
        self.dense3 = nn.Linear(512, 512)
        self.dense4 = nn.Linear(512, 512)
        self.dense5 = nn.Linear(512, 512)
        self.dense6 = nn.Linear(512, 1)

    def forward(self, z, x, return_value=False, exp_epsilon=0.05):
        lstm_out, (h_n, _) = self.lstm(z)
        lstm_out = lstm_out[:, -1, :]
        x = torch.cat([lstm_out, x], dim=-1)
        x = self.dense1(x)
        x = torch.relu(x)
        x = self.dense2(x)
        x = torch.relu(x)
        x = self.dense3(x)
        x = torch.relu(x)
        x = self.dense4(x)
        x = torch.relu(x)
        x = self.dense5(x)
        x = torch.relu(x)
        x = self.dense6(x)
        if return_value:
            return x
        else:
            if random.random() < exp_epsilon:
                action = torch.randint(x.shape[0], (1,))[0]
            else:
                action = torch.argmax(x, dim=0)[0]
            mask = torch.zeros(x.shape[0])
            mask[action] = 1
            return action, mask


def init_policy_landlord_net(config):
    model = LandlordLstmModel()
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.xavier_uniform_(m.weight)
    return model


def init_policy_farmer_net(config):
    model = FarmerLstmModel()
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.xavier_uniform_(m.weight)
    return model


# # Model dict is only used in evaluation but not training
# model_dict = {}
# model_dict['landlord'] = LandlordLstmModel
# model_dict['landlord_up'] = FarmerLstmModel
# model_dict['landlord_down'] = FarmerLstmModel

# class Model:
#     """
#     The wrapper for the three models. We also wrap several
#     interfaces such as share_memory, eval, etc.
#     """
#     def __init__(self, device=0):
#         self.models = {}
#         if not device == "cpu":
#             device = 'cuda:' + str(device)
#         self.models['landlord'] = LandlordLstmModel().to(torch.device(device))
#         self.models['landlord_up'] = FarmerLstmModel().to(torch.device(device))
#         self.models['landlord_down'] = FarmerLstmModel().to(torch.device(device))

#     def forward(self, position, z, x, training=False, flags=None):
#         model = self.models[position]
#         return model.forward(z, x, training, flags)

#     def share_memory(self):
#         self.models['landlord'].share_memory()
#         self.models['landlord_up'].share_memory()
#         self.models['landlord_down'].share_memory()

#     def eval(self):
#         self.models['landlord'].eval()
#         self.models['landlord_up'].eval()
#         self.models['landlord_down'].eval()

#     def parameters(self, position):
#         return self.models[position].parameters()

#     def get_model(self, position):
#         return self.models[position]

#     def get_models(self):
#         return self.models
