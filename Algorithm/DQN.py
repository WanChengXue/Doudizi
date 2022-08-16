'''
Author: error: git config user.name && git config user.email & please set dead value or install git
Date: 2022-08-15 20:59:32
LastEditors: error: git config user.name && git config user.email & please set dead value or install git
LastEditTime: 2022-08-16 21:32:08
FilePath: /RLFramework/Algorithm/DQN.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import torch
import torch.nn as nn


def get_cls():
    return DQNTrainer

class DQNTrainer():
    def __init__(self,model, target_model, optimizer, scheduler, policy_config):
        self.policy_config = policy_config
        # --------- 使用的是policy based的算法，并且是multiagent scenario，因此有一个中心的critic --------
        self.max_grad_norm = self.policy_config['max_grad_norm']
        self.model = model
        self.target_model = target_model
        self.optimizer = optimizer
        self.scheduler = scheduler
        # [landlord, farmer]
        self.agent_name = list(self.model.keys())[0]
        self.gamma = self.policy_config['gamma']
        self.critic_loss = nn.MSELoss()
    

    def step(self, training_data):
        current_state = training_data['current_state']
        next_state = training_data['next_state']
        instant_reward = training_data['instant_reward']
        mask = training_data['action_mask']
        done = training_data['done']
        current_state_Q_value = self.model[self.agent_name](current_state)
        next_state_Q_value = self.target_model[self.agent_name](next_state)
        mse_loss = 0.5 *  torch.mean((instant_reward + next_state_Q_value[mask] * self.gamma *done) - current_state_Q_value[mask]) **2
        info_dict = dict()
        info_dict['mse_loss'] = mse_loss.item()
        return info_dict