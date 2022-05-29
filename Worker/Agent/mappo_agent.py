import torch

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
        self.independent_critic = 'critic' in self.agent_config.keys()
        # ----------- 定义两个变量,动作的最大值,动作的最小值 ---------
        self.action_high = config.get('action_high', 1)
        self.action_low = config.get('action_low', -1)
        # ----------- 构建智能体 ----------
        self.construct_model()

    def construct_model(self):
        # --------- 如果所使用的是centralized critic
        if 'policy' in self.agent_config:
            self.policy_net = create_model(self.agent_config['policy'])
            # ------ PPO算法，可以考虑IPPO和MAPPO两种，前者是要有独立的critic网络的 --------
            if not self.agent_config.get('eval_mode', False) and self.independent_critic:
                self.critic_net = create_model(self.agent_config['critic'])
        else:
            self.centralized_critic_net = create_model(self.agent_config)
        
    def synchronize_model(self, model_path):
        # ------------ 这个函数用来同步本地策略模型 ——-----------
        for key in model_path.keys():
            if key == 'policy':
                deserialize_model(self.policy_net, model_path['policy'])
            elif key == 'critic':
                deserialize_model(self.critic_net, model_path['critic'])
            elif key == 'centralized_critic':
                deserialize_model(self.centralized_critic_net, model_path['centralized_critic'])
            else:
                pass
    
    def compute_action_training_mode(self, obs):
        with torch.no_grad():
            action = self.policy_net.get_action(obs)
            # ----- 对动作进行clip操作 ------
            clip_action = torch.clamp(action, min=self.action_low, max=self.action_high)
        return clip_action

    def compute_action_eval_mode(self, obs):
        with torch.no_grad():
            action = self.policy_net.get_det_action(obs)
            clipped_action = torch.clamp(action, self.action_low, self.action_high)
        return clipped_action
    
    def compute_state_value(self, obs):
        with torch.no_grad():
            if self.independent_critic:
                old_state_value = self.critic_net(obs)
            else:
                old_state_value = self.centralized_critic_net(obs)
        return old_state_value

    def reset(self):
        pass