'''
Author: error: git config user.name && git config user.email & please set dead value or install git
Date: 2022-08-15 20:59:32
LastEditors: error: git config user.name && git config user.email & please set dead value or install git
LastEditTime: 2022-08-19 19:15:24
FilePath: /RLFramework/Algorithm/DQN.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import torch
import torch.nn as nn


def get_cls():
    return DQNTrainer

def soft_update(current_network, target_network, tau):
    for target_param, param in zip(
        target_network.parameters(), current_network.parameters()
    ):
        target_param.data.copy_(target_param.data * (1 - tau) + param.data * tau)

class DQNTrainer():
    def __init__(self,model, target_model, optimizer, scheduler, policy_config):
        self.policy_config = policy_config
        # --------- 使用的是policy based的算法，并且是multiagent scenario，因此有一个中心的critic --------
        self.max_grad_norm = self.policy_config['max_grad_norm']
        self.agent_name = list(model.keys())[0]
        self.model = model[self.agent_name]['policy']
        self.target_model = target_model[self.agent_name]['policy']
        self.optimizer = optimizer[self.agent_name]['policy']
        self.scheduler = scheduler[self.agent_name]['policy']
        # [landlord, farmer]
        self.gamma = self.policy_config['gamma']
        self.tau = float(self.policy_config['tau'])
        self.n_step = self.policy_config['n_step']
        self.critic_loss = nn.MSELoss()
    

    def step(self, training_data, PRB = None):
        current_state = training_data[self.agent_name]['current_agent_obs']
        # next_state = training_data[self.agent_name]['next_agent_obs']
        instant_reward = training_data[self.agent_name]['instant_reward']
        # actions = training_data[self.agent_name]['actions'].bool()
        # done = training_data[self.agent_name]['done']
        # next_state_action_length = training_data[self.agent_name]['next_state_action_length'].long()
        # next_q_list = []
        # start_value = 0
        current_state_Q_value = self.model(**current_state, return_value=True)
        # with torch.no_grad():
        #     next_state_Q_value = self.target_model(**next_state, return_value=True)
        #     for batch_length in next_state_action_length:
        #         next_q_list.append(torch.max(next_state_Q_value[start_value: start_value+batch_length]))
        #         start_value += batch_length
        #     effective_q_value = torch.stack(next_q_list, 0).unsqueeze(-1)
        # target_value = instant_reward + effective_q_value * self.gamma ** self.n_step *(1-done)
        target_value = instant_reward
        mse_loss = self.critic_loss(target_value, current_state_Q_value) 
        info_dict = dict()
        info_dict['mse_loss'] = mse_loss.item()
        self.optimizer.zero_grad()
        mse_loss.backward()
        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.max_grad_norm
            )
        _net_max_grads = {}
        for _name, _value in self.model.named_parameters():
            _net_max_grads[_name] = torch.max(_value.grad).item()
        info_dict['layer_max_grad'] = _net_max_grads
        self.optimizer.step()
        self.scheduler.step()
        soft_update(self.model, self.target_model, self.tau)
        return info_dict