import torch
from Algorithm.utils import huber_loss, freeze_model, unfreeze_model, reversed_dict, soft_update
from Utils.data_utils import merge_dict, mean_dict
# ----------- 这个地方是用来实现MADDPG算法 ----------
'''
    核心更新逻辑:
        更新中心化的critic网络，这个使用TD Loss更新，建立target critic网络，target actor网络，使用target actor网络计算出next obs的动作，再调用target critic网络计算Q next
            计算出来了Q next，使用公式gamma * Q_next + instant_reward - Q_curent计算TD Error，然后反向梯度更新critic网络
        更新异构policy网络，有联合的动作存放，因此假设更新第i个智能体，就需要将第i个智能体的观测通过策略网络，得到带计算图的动作，而后和其余智能体的动作拼接得到联合动作送入中心critic
            网络中，通过中心critic计算Q值，使用loss函数 -sum Q 作为loss即可。在更新的时候需要将中心criti网络的梯度设置成为False
        Target网络的参数需要进行soft-update
'''
def get_cls():
    return MADDPGTrainer

    
class MADDPGTrainer:
    def __init__(self, model, target_model, optimizer, scheduler, policy_config):
        self.policy_config = policy_config
        self.policy_net = model['policy']
        self.critic_net = model['critic']
        self.target_policy_net = target_model['policy']
        self.target_critic_net = target_model['critic']
        self.policy_optimizer = optimizer['policy']
        self.critic_optimizer = optimizer['critic']
        self.policy_scheduler = scheduler['policy']
        self.critic_scheduler = scheduler['critic']
        self.parameter_sharing = self.policy_config.get('parameter_sharing', False)
        self.agent_name_list = list(self.policy_optimizer.keys())
        # ---------- 这个critic_name_list具体来说，如果说是中心化训练，就是centralized_critic ------
        self.critic_name = list(self.critic_optimizer.keys())[0]
        self.gamma = self.policy_config['gamma']
        self.critic_loss_fn = huber_loss 
        self.max_grad_norm = self.policy_config['max_grad_norm']
        self.tau = float(self.policy_config['tau'])

    def step(self, training_data):
        curret_agent_obs = training_data['current_agent_obs']
        current_centralized_state = training_data['current_centralized_state']
        actions = training_data['actions']
        next_agent_obs = training_data['next_agent_obs']
        next_centralized_state = training_data['next_centralized_state']
        done = training_data['done']
        instant_reward = training_data['instant_reward']
        next_actions = []
        info_dict = dict()
        with torch.no_grad():
            # ----------- 计算next_action ------------------ 
            for agent_index, key in enumerate(next_agent_obs.keys()):
                if self.parameter_sharing:
                    agent_name = self.agent_name_list[0]
                else:
                    agent_name = self.agent_name_list[agent_index]
                agent_next_action = self.target_policy_net[agent_name](next_agent_obs[key])
                next_actions.append(agent_next_action)
            # --------------- 计算Target Q value ----------------
            next_Q_value = self.target_critic_net[self.critic_name](next_centralized_state, torch.cat(next_actions, -1))
        # --------------- 计算TD Error -----------------
        # ----------- 构建current_action ------------
        critic_info = dict()
        reversed_actions_dcit = reversed_dict(actions)
        current_actions = []
        for key in reversed_actions_dcit.keys():
            current_actions.append(reversed_actions_dcit[key])
        current_Q_value = self.critic_net[self.critic_name](current_centralized_state, torch.cat(current_actions, -1))
        target_Q_value = instant_reward + self.gamma * next_Q_value * (1- done)
        critic_loss = self.critic_loss_fn(current_Q_value, target_Q_value).mean()
        self.critic_optimizer[self.critic_name].zero_grad()
        critic_loss.backward()
        critic_grad_norm = torch.nn.utils.clip_grad_norm_(self.critic_net[self.critic_name].parameters(), self.max_grad_norm)
        critic_max_grads = {}
        for name, value in self.critic_net[self.critic_name].named_parameters():
            critic_max_grads[name] = torch.max(value.grad).item()
        critic_info['critic_grad_norm'] = critic_grad_norm.item()
        critic_info['Layer_max_grad'] = critic_max_grads
        critic_info['critic_loss'] = critic_loss.item()
        self.critic_optimizer[self.critic_name].step()
        self.critic_scheduler[self.critic_name].step()
        info_dict['Model_critic'] = critic_info
        # ------------------ 更新策略网络, 先固定critic网络的参数 ----------------------
        freeze_model(self.critic_net[self.critic_name])
        for agent_index, key in enumerate(curret_agent_obs.keys()):
            agent_policy_info = dict()
            if self.parameter_sharing:
                agent_name = self.agent_name_list[0]
            else:
                agent_name = self.agent_name_list[agent_index]
            network_output_action = self.policy_net[agent_name](curret_agent_obs[key])
            # -------------- 构建critic网络的action部分 ------------
            composed_actions = []
            for agent_key in reversed_actions_dcit.keys():
                if agent_key == key:
                    composed_actions.append(network_output_action)
                else:
                    composed_actions.append(reversed_actions_dcit[agent_key])
            # ------------- 放入current critic网络中计算Q值 -------------
            Q_value = self.critic_net[self.critic_name](current_centralized_state, torch.cat(composed_actions, -1))
            agent_policy_loss = -Q_value.mean()
            self.policy_optimizer[agent_name].zero_grad()
            agent_policy_loss.backward()
            agent_policy_grad_norm = torch.nn.utils.clip_grad_norm_(self.policy_net[agent_name].parameters(), self.max_grad_norm)
            agent_policy_max_grads = dict()
            for name, value in self.policy_net[agent_name].named_parameters():
                agent_policy_max_grads[name] = torch.max(value.grad).item()
            self.policy_optimizer[agent_name].step()
            self.policy_scheduler[agent_name].step()
            agent_policy_info['Layer_max_grads'] = agent_policy_max_grads
            agent_policy_info['Policy_loss'] = agent_policy_loss.item()
            agent_policy_info['Policy_grad_norm'] = agent_policy_grad_norm.item()
            agent_policy_info['Q_value_std'] = torch.std(Q_value).item()
            if 'Model_policy_{}'.format(agent_name) not in info_dict.keys():
                info_dict['Model_policy_{}'.format(agent_name)] = agent_policy_info
            else:
                # ----------- 字典合并 ------------
                info_dict['Model_policy_{}'.format(agent_name)] = merge_dict(info_dict['Model_policy_{}'.format(agent_name)], agent_policy_info)
        # --------- 最后，如果说使用参数共享用策略，就需要对这个info_dict中的策略部分的key进行平均化处理 ------------
        if self.parameter_sharing:
            info_dict['Model_policy_{}'.format(self.agent_name_list[0])] = mean_dict(info_dict['Model_policy_{}'.format(self.agent_name_list[0])])
        # ------------ 将critic网络的参数变得可以训练 -----------
        unfreeze_model(self.critic_net[self.critic_name])
        # ------------- 将active model和target model进行同步操作 ----------------
        for agent_name in self.policy_net.keys():
            soft_update(self.policy_net[agent_name], self.target_policy_net[agent_name], self.tau)
        for critic_name in self.critic_net.keys():
            soft_update(self.critic_net[critic_name], self.target_critic_net[critic_name], self.tau)
        return info_dict



