import torch
import torch.nn as nn
import numpy as np
from Algorithm.utils import huber_loss, soft_update
from Utils.data_utils import merge_dict, mean_dict
# ----------- 这个地方是用来实现DDPG算法 ----------
'''
    这个算法使用的是D4PG,相比DDPG做了四个改进:
        1.使用n_step TD Error
        2.使用分布式worker采样(Apex系列)
        3.使用value distribution算法 C51
        4.使用prioritized replay buffer (segment tree)
    ----- 推荐阅读paper:
        1. Continuous Control With Deep Reinforcement Learning
        2. Distributional Perspective on Reinforcement Learning
        3. Prioritized Replay Buffer
        4. Distributed Prioritized Experience Replay (Ape-X)
        5. Rainbow: Combining Improvements in Deep Reinforcement Learning
        6. Distributed Distributional Deep Deterministic Policy Gradient(实现的算法文章)
'''
def get_cls():
    return Independent_D4PGTrainer

    
class Independent_D4PGTrainer:
    def __init__(self, model, target_model, optimizer, scheduler, policy_config):
        self.policy_config = policy_config
        # --------- 由于采用了C51算法，所以Q网络的输出是一个分布结果 ------
        self.n_atoms = self.policy_config['n_atoms']
        self.value_max = self.policy_config['value_max']
        self.value_min = self.policy_config['value_min']
        self.value_delta = (self.value_max - self.value_min) / (self.n_atoms-1)
        # --------- 使用的是独立的D4PG算法,因此是没有中心化的critic这个东西 --------
        self.agent_name_list = list(model.keys())
        self.model = model
        self.target_model = target_model
        self.optimizer = optimizer
        self.scheduler = scheduler
        # ------------ 如果不采用参数共享的话，显然是一个智能体一个策略网络了 ----------
        self.parameter_sharing = self.policy_config.get('parameter_sharing', False)
        self.gamma = self.policy_config['gamma']
        # ------- 这个n_step表示的是DQN拓展中，n-step Q ------
        self.n_step = self.policy_config['n_step']
        self.critic_loss_fn = nn.BCELoss(reduction='none')
        self.max_grad_norm = self.policy_config['max_grad_norm']
        self.tau = float(self.policy_config['tau'])

    def distr_projection(self, next_distr, rewards, dones_mask):
        proj_distr = torch.zeros_like(next_distr)
        for atom in range(self.n_atoms):
            target_value = rewards + (self.value_min + atom * self.value_delta) * (self.gamma ** self.n_step)* (1- dones_mask.float())
            # tz_j = torch.clamp_min(torch.clamp_max(target_value, self.value_min), self.value_max)  
            tz_j = torch.clamp(target_value, min=self.value_min, max=self.value_max)
            # tz_J的操作就是让rewards + Atom_value * gamma 映射到atom上, 维度为batch_size * 1
            b_j = (tz_j - self.value_min) / self.value_delta  # 这个是往回映射了，然后进行对齐操作  
            l = torch.floor(b_j).long() #向下取整
            u = torch.ceil(b_j).long()  #向上取整
            # ----------- 这个地方是在进行平摊操作，atom之间进行对齐 -----------
            eq_mask = (u == l).squeeze()  # (batch_size,)
            # -------- 如果说，恰好，这个b_j就是一个整数，正好对其，就直接把目标概率分布在这个atom上面的概率值拿过来 -----
            proj_distr[eq_mask, l[eq_mask]] += next_distr[eq_mask, atom]
            ne_mask = (u != l).squeeze() # (batch_size, )
            # ---------- 根据距离左右两边分一点。举个例子，如果说这个u是4.3,概率为0.4，那么平摊到4，5上，离4近，因此概率平摊到4上面为
            # (5-4.3) * 0.4 = 0.28, 离5比较远，平摊到5上面的概率为（4.3-4） * 0.4 = 0.12 -------
            proj_distr[ne_mask, l[ne_mask].squeeze()] += next_distr[ne_mask, atom] * ((u - b_j)[ne_mask]).squeeze()
            proj_distr[ne_mask, u[ne_mask].squeeze()] += next_distr[ne_mask, atom] * ((b_j - l)[ne_mask]).squeeze()
        return proj_distr


    def step(self, training_data, priority_weights = None):
        info_dict = dict()
        critic_info = dict()
        agent_policy_info = dict()
        batch_td_error = dict()
        for agent_name in training_data.keys():
            current_agent_obs = training_data[agent_name]['current_agent_obs']
            agent_current_action = training_data[agent_name]['actions']
            next_agent_obs = training_data[agent_name]['next_agent_obs']
            done = training_data[agent_name]['done']
            instant_reward = training_data[agent_name]['instant_reward']
            with torch.no_grad():
            # ----------- 计算next_action ------------------ 
                if self.parameter_sharing:
                    pass 
                else:
                    agent_next_action = self.target_model[agent_name]['policy'](next_agent_obs)        
                    next_q_distribution = self.target_model[agent_name]['critic'](next_agent_obs, agent_next_action)
                    projection_distribution_value = self.distr_projection(next_q_distribution, instant_reward, done)
            # ------- 计算current state distribution -----
            current_q_distribution = self.model[agent_name]['critic'](current_agent_obs, agent_current_action)
            critic_loss_vector = torch.sum(-torch.log(current_q_distribution) * projection_distribution_value, 1)
            if priority_weights is None:
                agent_critic_loss = torch.mean(critic_loss_vector)
            else:
                batch_td_error[agent_name] = torch.abs(critic_loss_vector).data.cpu().numpy() + 1e-4
                agent_critic_loss = torch.mean(critic_loss_vector * priority_weights[agent_name])
            self.optimizer[agent_name]['critic'].zero_grad()
            agent_critic_loss.backward()
            critic_grad_norm = torch.nn.utils.clip_grad_norm_(self.model[agent_name]['critic'].parameters(), self.max_grad_norm)
            critic_max_grads = {}
            for name, value in self.model[agent_name]['critic'].named_parameters():
                if value.requires_grad:
                    critic_max_grads[name] = torch.max(value.grad).item()
            critic_info['critic_grad_norm'] = critic_grad_norm.item()
            critic_info['Layer_max_grad'] = critic_max_grads
            critic_info['critic_loss'] = agent_critic_loss.item()
            self.optimizer[agent_name]['critic'].step()
            self.scheduler[agent_name]['critic'].step()
            # --------------- 开始更新policy网络 -------------
            current_action_value = self.model[agent_name]['policy'](current_agent_obs)
            current_q_distribution = self.model[agent_name]['critic'](current_agent_obs, current_action_value)
            try:
                # ---------- DDP包裹了模型之后的调用方式 ------------
                current_q_value = - self.model[agent_name]['critic'].module.distribution_to_value(current_q_distribution)
            except:
                current_q_value = - self.model[agent_name]['critic'].distribution_to_value(current_q_distribution)
            agent_policy_loss = torch.mean(current_q_value)
            self.optimizer[agent_name]['policy'].zero_grad()
            agent_policy_loss.backward()
            agent_policy_max_grads = dict()
            for name, value in self.model[agent_name]['policy'].named_parameters():
                if value.requires_grad:
                    agent_policy_max_grads[name] = torch.max(value.grad).item()
            agent_policy_grad_norm = torch.nn.utils.clip_grad_norm_(self.model[agent_name]['policy'].parameters(), self.max_grad_norm)
            agent_policy_info['Layer_max_grads'] = agent_policy_max_grads
            agent_policy_info['Policy_loss'] = agent_policy_loss.item()
            agent_policy_info['Policy_grad_norm'] = agent_policy_grad_norm.item()
            agent_policy_info['Q_value_std'] = torch.std(current_q_value).item()
            self.scheduler[agent_name]['policy'].step()
            self.optimizer[agent_name]['policy'].step()
            # ----------- 字典合并 ------------    
            if 'Model_policy_{}'.format(agent_name) not in info_dict.keys():
                # --------- 如果使用同构网络，那么critic和policy都是只有一个，不同构，那都是有多个 ------
                info_dict['Model_policy_{}'.format(agent_name)] = agent_policy_info
                info_dict['Model_critic_{}'.format(agent_name)] = critic_info
            else:
                # ----------- 字典合并 ------------
                info_dict['Model_policy_{}'.format(agent_name)] = merge_dict(info_dict['Model_policy_{}'.format(agent_name)], agent_policy_info)
                info_dict['Model_critic_{}'.format(agent_name)] = merge_dict(info_dict['Model_critic_{}'.format(agent_name)], critic_info)
        # # --------- 最后，如果说使用参数共享用策略，就需要对这个info_dict中的策略部分的key进行平均化处理 ------------
        # if self.parameter_sharing:
        #     info_dict['Model_policy_{}'.format(self.agent_name_list[0])] = mean_dict(info_dict['Model_policy_{}'.format(self.agent_name_list[0])])
        #     info_dict['Model_critic_{}'.format(self.critic_name_list[0])] = mean_dict(info_dict['Model_critic_{}'.format(self.critic_name_list[0])])
        # ------------- 将active model和target model进行同步操作 ----------------
        for agent_name in self.agent_name_list:
            for model_type in self.model[agent_name].keys():
                soft_update(self.model[agent_name][model_type], self.target_model[agent_name][model_type], self.tau)

        if priority_weights is None:
            return info_dict
        else:
            return info_dict, batch_td_error
    

