import torch
import torch.nn as nn
from Algorithm.utils import huber_loss, mean_info_dict


def get_cls():
    return MAPPOTrainer

class MAPPOTrainer():
    def __init__(self,model, optimizer, scheduler, policy_config):
        self.policy_config = policy_config
        # --------- 使用的是policy based的算法，并且是multiagent scenario，因此有一个中心的critic --------
        try:
            self.agent_name_list = list(model.keys())
            self.agent_name_list.remove('centralized_critic')
            self.use_centralized_critic = True
        except:
            self.agent_name_list  = list(model.keys())
            self.use_centralized_critic = False
        self.max_grad_norm = self.policy_config['max_grad_norm']
        # ------- clip_epsilon是PPO里面的参数 --------
        self.clip_epsilon = self.policy_config['clip_epsilon']
        # ------ 这个变量是entropy约束的系数 ----------
        self.entropy_coef = self.policy_config['entropy_coef']
        #  -------------- 这个地方进行value loss的clip操作 -----------
        self.clip_value = self.policy_config.get('clip_value', False)      
        self.use_popart = self.policy_config.get('use_popart', False)
        # ----------- dual_clip表示要不要进行双梯度剪裁 ---------
        self.dual_clip = self.policy_config.get('dual_clip', None)
        # --------------- 同一段样本，默认使用PPO更新5次 --------------
        self.ppo_update_epoch = self.policy_config.get('ppo_update_epoch', 5)
        # ----------- 构建优化器，调度器，model -------------
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.critic_loss = nn.MSELoss()
    

    def update_centralized_critic(self, target_state_value, centralized_state, old_state_value):
        # ------------- 使用了GAE对advantage进行估计，使用了V网络，需要对其进行更新 ---------
        # ------------- 更新的loss是使Q-V的MSE达到最小，其实就是让A值为0 ----------
        critic_info_dict = dict()
        predict_state_value = self.model['centralized_critic'](centralized_state)
        if self.clip_value:
            value_clamp_range = 0.2
            value_pred_clipped = old_state_value + (predict_state_value - old_state_value).clamp(-value_clamp_range, value_clamp_range)
            if self.use_popart:
                pass
            else:
                clipped_state_value_loss = self.critic_loss(target_state_value, value_pred_clipped)
                unclipped_state_value_loss = self.critic_loss(target_state_value, predict_state_value)
                value_loss_vecotr = torch.max(clipped_state_value_loss, unclipped_state_value_loss)

        else:
            if self.use_popart:
                pass
            else:
                value_loss_vecotr = self.critic_loss(target_state_value, predict_state_value)
        value_loss = torch.mean(value_loss_vecotr)
        # ------- 计算critic loss -----
        self.optimizer['centralized_critic'].zero_grad()
        value_loss.backward()
        centralized_critic_grad_norm = torch.nn.utils.clip_grad_norm_(self.model['centralized_critic'].parameters(), self.max_grad_norm)
        centralized_critic_max_grad_dcit = {}
        for name, value in self.model['centralized_critic'].named_parameters():
            if value.requires_grad:
                centralized_critic_max_grad_dcit[name] = torch.max(value.grad).item()
        critic_info_dict['Layer_max_grad'] = centralized_critic_max_grad_dcit
        critic_info_dict['critic_grad_norm'] = centralized_critic_grad_norm.item()
        critic_info_dict['critic_loss'] = value_loss.item()
        self.optimizer['centralized_critic'].step()
        self.scheduler['centralized_critic'].step()
        self.epoch_info_dict['Model_centralized_critic'] = critic_info_dict
    
    def update_policy(self, agent_name, agent_obs, agent_action, agent_old_log_prob, agent_advantage):
        policy_info_dict = dict()
        # -------- 使用当前的policy网络计算一下action的log值，以及这个动作的entropy -------
        try:
            action_log_probs = self.model[agent_name]['policy'].module.evaluate(agent_obs, agent_action)
        except:
            action_log_probs = self.model[agent_name]['policy'].evaluate(agent_obs, agent_action)
        # ------ 计算重要性因子 -------
        importance_ratio = torch.exp(agent_old_log_prob - action_log_probs)
        surr1 = importance_ratio * agent_advantage
        policy_info_dict['Surr1'] = surr1.mean().item()
        # ----------- 使用了PPO算法，这里需要进行clip操作 ---------
        surr2 = torch.clamp(importance_ratio, 1.0-self.clip_epsilon, 1.0+self.clip_epsilon) * agent_advantage
        policy_info_dict['Surr2'] = surr2.mean().item()
        surr = torch.min(surr1, surr2)
        if self.dual_clip is not None:
            # ---------- 使用双梯度剪裁，默认开启 -----------
            surr3 =  torch.min(self.dual_clip * agent_advantage, torch.zeros_like(agent_advantage))
            policy_info_dict['Surr3'] = surr3.mean().item()
            surr = torch.max(surr,surr3)
        agent_policy_loss = torch.mean(-surr)
        # entropy_loss = -torch.mean(dist_entropy)
        self.optimizer[agent_name]['policy'].zero_grad()
        total_policy_loss = agent_policy_loss
        total_policy_loss.backward()
        agent_policy_grad_norm = nn.utils.clip_grad_norm_(self.model[agent_name]['policy'].parameters(), self.max_grad_norm)
        policy_grad_dict = dict()
        for name, value in self.model[agent_name]['policy'].named_parameters():
            if value.requires_grad:
                policy_grad_dict[name] = torch.max(value.grad).item()
        policy_info_dict['Policy_loss'] = agent_policy_loss.item()
        # policy_info_dict['Entropy_loss'] = entropy_loss.item()
        policy_info_dict['Total_loss'] = total_policy_loss.item()
        policy_info_dict['agent_grad_norm'] = agent_policy_grad_norm.item()
        policy_info_dict['Layer_max_norm'] = policy_grad_dict
        self.epoch_info_dict['Model_policy_{}'.format(agent_name)] = policy_info_dict
    
    def update_critic(self, agent_name, agent_obs, old_state_value, agent_advantage):
        # --------- 这个位置是使用ippo的时候，每一个智能体除了policy还有critic ---------
        critic_info_dict = dict()
        predict_state_value = self.model[agent_name]['critic'](agent_obs)
        target_state_value = agent_advantage + old_state_value
        if self.clip_value:
            value_clamp_range = 0.2
            value_pred_clipped = old_state_value + (predict_state_value - old_state_value).clamp(-value_clamp_range, value_clamp_range)
            if self.use_popart:
                pass
            else:
                clipped_state_value_loss = self.critic_loss(target_state_value, value_pred_clipped)
                unclipped_state_value_loss = self.critic_loss(target_state_value, predict_state_value)
                value_loss = torch.max(clipped_state_value_loss, unclipped_state_value_loss)
        else:
            if self.use_popart:
                pass
            else:
                value_loss_vecotr = self.critic_loss(target_state_value, predict_state_value)
                value_loss = torch.mean(value_loss_vecotr)
        # ------- 计算critic loss -----
        self.optimizer[agent_name]['critic'].zero_grad()
        value_loss.backward()
        critic_grad_norm = torch.nn.utils.clip_grad_norm_(self.model[agent_name]['critic'].parameters(), self.max_grad_norm)
        critic_max_grad_dict = {}
        for name, value in self.model[agent_name]['critic'].named_parameters():
            if value.requires_grad:
                critic_max_grad_dict[name] = torch.max(value.grad).item()
        critic_info_dict['Layer_max_grad'] = critic_max_grad_dict
        critic_info_dict['critic_grad_norm'] = critic_grad_norm.item()
        critic_info_dict['critic_loss'] = value_loss.item()
        self.optimizer[agent_name]['critic'].step()
        self.scheduler[agent_name]['critic'].step()
        # ----------- 使用ippo算法更新，每一个智能体都能够算一个TDError，最后的更新权重为对所有的权重计算一个min值 ----------
        self.epoch_info_dict['Model_critic_{}'.format(agent_name)] = critic_info_dict

    def step(self, training_data):
        info_list = []
        for ep in range(self.ppo_update_epoch):
            self.epoch_info_dict = dict()
            if self.use_centralized_critic:
                # ------------- update centralized_critic ----------   
                current_centralized_state = training_data['current_centralized_state']
                old_state_value = training_data['old_state_value']
                centralized_advantage_value = training_data['advantages']
                target_state_value = old_state_value + centralized_advantage_value
                self.update_centralized_critic(target_state_value, current_centralized_state, old_state_value)

            for agent_name in self.agent_name_list:
                if self.use_centralized_critic:
                    agent_obs = training_data['current_agent_obs'][agent_name]
                    agent_action = training_data['actions'][agent_name]
                    agent_old_log_prob = training_data['old_log_prob'][agent_name]
                    agent_advantage = centralized_advantage_value
                else:
                    agent_obs = training_data[agent_name]['current_agent_obs']
                    agent_action = training_data[agent_name]['actions']  
                    agent_old_log_prob = training_data[agent_name]['old_log_prob']
                    # ------------ critic update ----------
                    agent_old_state_value = training_data[agent_name]['old_obs_value']
                    agent_advantage = training_data[agent_name]['advantages']
                    self.update_critic(agent_name, agent_obs, agent_old_state_value, agent_advantage)
                    
                # ------------- actor update ---------------
                self.update_policy(agent_name, agent_obs, agent_action, agent_old_log_prob, agent_advantage)
            info_list.append(self.epoch_info_dict)
        # ---------- 将所有epoch的loss平均一下 ----------
        mean_dict = mean_info_dict(info_list)
        return mean_dict