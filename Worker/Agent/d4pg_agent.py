import torch
import os
import sys
current_path = os.path.abspath(__file__)
root_path = '/'.join(current_path.split('/')[:-3])
sys.path.append(root_path)



from Utils.model_utils import deserialize_model, create_model
from Utils.data_utils import OUNoise




class Agent:
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
        if not self.agent_config.get('eval_mode', False):
            self.critic_net = create_model(self.agent_config['critic'])
            self.ou_enabled = self.agent_config.get('ou_enabled', False)
            if self.ou_enabled:
                self._construct_ou_noise_explorator()


    def synchronize_model(self, model_path):
        # ------------ 这个函数用来同步本地策略模型 ——-----------
        for key in model_path.keys():
            if key == 'policy':
                deserialize_model(self.policy_net, model_path['policy'])
            elif key == 'critic':
                deserialize_model(self.critic_net, model_path['critic'])
            else:
                pass


    def _construct_ou_noise_explorator(self):
        ou_config = dict()
        ou_config['action_low'] = self.agent_config['action_low']
        ou_config['action_high'] = self.agent_config['action_high']
        ou_config['action_dim'] = self.agent_config['action_dim']
        self.ou_explorator = OUNoise(ou_config)
        self.ou_explorator.reset()
        

    def compute_action_training_mode(self, obs):
        with torch.no_grad():
            output = self.policy_net(obs)
        # -------- 使用探索策略 ---------
        if self.ou_enabled:
            explore_action = self.ou_explorator.step(output)
        else:
            # -------- 如果不用ou探索,就使用高斯探索策略 ---------
            explore_action = self.explore_by_gaussian(output)
        return explore_action


    def compute_action_eval_mode(self, obs):
        with torch.no_grad():
            output = self.policy_net(obs)
        return output

    def explore_by_gaussian(self, current_action):
        random_noise = torch.randn(1, self.model_config['action_dim'])
        return torch.clamp(0.3 * random_noise + current_action, self.action_low, self.action_high)

