from re import L
import torch
import random
from torch.distributions import MultivariateNormal, Normal
import numpy as np

import os
import sys
current_path = os.path.abspath(__file__)
root_path = '/'.join(current_path.split('/')[:-3])
sys.path.append(root_path)



from Utils.model_utils import deserialize_model, create_model

class Agent():
    def __init__(self, config):
        # ------------- 创建网络，包括actor网络和critic网络 --------------------
        self.agent_config = config
        # ----------- 定义两个变量,动作的最大值,动作的最小值 ---------
        self.action_high = config.get('action_high', 1)
        self.action_low = config.get('action_low', -1)
        # ----------- 构建智能体 ----------
        self.construct_model()

    def construct_model(self):
        self.agent = dict()
        # ------ DDPG系列算法,actor和critic都有,并且一定是分开的 --------
        self.policy_net = create_model(self.agent_config['policy'])
        self.critic_net = create_model(self.agent_config['critic'])
        self.double_critic = create_model(self.agent_config['double_critic'])
        
    
    def synchronize_model(self, model_path):
        # ------------ 这个函数用来同步本地策略模型 ——-----------
        for key in model_path.keys():
            if key == 'policy':
                deserialize_model(self.policy_net, model_path['policy'])
            elif key == 'critic' or key == 'double_critic':
                deserialize_model(self.critic_net, model_path['critic'])
            else:
                pass
    
    def compute_action_training_mode(self, obs):
        with torch.no_grad():
            action = self.policy_net.get_action(obs)
        return action

    def compute_action_eval_mode(self, obs):
        with torch.no_grad():
            action = self.policy_net.get_det_action(obs)
        return action