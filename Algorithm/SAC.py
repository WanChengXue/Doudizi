import torch
import torch.nn as nn
from wandb import agent
from Algorithm.utils import huber_loss, soft_update


def get_cls():
    return SACTrainer

    
class SACTrainer:
    def __init__(self, model, target_model, optimizer, scheduler, policy_config):
        self.policy_config = policy_config
        # --------- 这个步骤是使用了C51算法的时候才会生效 ------
        self.categoraical_q_value =  self.policy_config.get('categorical_q_value', False)
        if self.categoraical_q_value:
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
        self.critic_loss_1 = nn.MSELoss()
        self.critic_loss_2 = nn.MSELoss()
        self.max_grad_norm = self.policy_config['max_grad_norm']
        self.tau = float(self.policy_config['tau'])
        # 如果说alpha是自适应学习,那么就不传入值
        if 'entropy_coefficient' not in self.policy_config:
            self.log_alpha = torch.tensor([0.0], requires_grad=True).to(self.rank)
            self.alpha = self.log_alpha.exp().detach()
            self.alpha_optimizer = torch.nn.optim.Adam(params = [self.log_alpha], lr=3e-4)
        else:
            self.alpha = self.policy_config.get('entropy_coefficient', 0.4)


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


    def update_critic_when_using_categorical_critic(self, agent_name, current_agent_obs, agent_current_action, next_agent_obs, instant_reward, done):
        pass

    def update_critic(self, agent_name, current_agent_obs, agent_current_action, next_agent_obs, instant_reward, done, priority_weight):
        critic_info_dict = dict()
        critic_info_dict['critic_1'] = dict()
        critic_info_dict['critic_2'] = dict()
        with torch.no_grad():
        # ----------- 计算next_action ------------------ 
            if self.parameter_sharing:
                pass 
            else:
                try:
                    next_action, log_action_prob = self.model[agent_name]['policy'].module.evaluate(next_agent_obs)
                except:
                    next_action, log_action_prob = self.model[agent_name]['policy'].evaluate(next_agent_obs)
                Q_target_1 = self.target_model[agent_name]['critic'](next_agent_obs, next_action)
                Q_target_2 = self.target_model[agent_name]['double_critic'](next_agent_obs, next_action)
                # ---------------- 计算两个target Q的min ------------------
                Q_target_next = torch.min(Q_target_1, Q_target_2)
                Q_target = instant_reward + self.gamma ** self.n_step * (1-done) *(Q_target_next - self.alpha * log_action_prob)
        # ------- 计算critic loss -----
        current_q_1 = self.model[agent_name]['critic'](current_agent_obs, agent_current_action)
        current_q_2 = self.model[agent_name]['double_critic'](current_agent_obs, agent_current_action)
        assert current_q_1.shape == Q_target.shape
        agent_critic_loss_1 = self.critic_loss_1(current_q_1, Q_target)
        self.optimizer[agent_name]['critic'].zero_grad()
        agent_critic_loss_1.backward()
        critic_1_grad_norm = torch.nn.utils.clip_grad_norm_(self.model[agent_name]['critic'].parameters(), self.max_grad_norm)
        critic_1_max_grads = {}
        for name, value in self.model[agent_name]['critic'].named_parameters():
            if value.requires_grad:
                critic_1_max_grads[name] = torch.max(value.grad).item()
        critic_info_dict['critic_1']['critic_grad_norm'] = critic_1_grad_norm.item()
        critic_info_dict['critic_1']['Layer_max_grad'] = critic_1_max_grads
        critic_info_dict['critic_1']['critic_loss'] = agent_critic_loss_1.item()
        self.optimizer[agent_name]['critic'].step()
        self.scheduler[agent_name]['critic'].step()

        # ------------ 更新double q网络的参数 ----------------
        agent_critic_loss_2 = self.critic_loss_2(current_q_2, Q_target)
        self.optimizer[agent_name]['double_critic'].zero_grad()
        agent_critic_loss_2.backward()
        critic_2_grad_norm = torch.nn.utils.clip_grad_norm_(self.model[agent_name]['double_critic'].parameters(), self.max_grad_norm)
        critic_2_max_grads = {}
        for name, value in self.model[agent_name]['double_critic'].named_parameters():
            if value.requires_grad:
                critic_2_max_grads[name] = torch.max(value.grad).item()
        critic_info_dict['critic_2']['critic_grad_norm'] = critic_2_grad_norm.item()
        critic_info_dict['critic_2']['Layer_max_grad'] = critic_2_max_grads
        critic_info_dict['critic_2']['critic_loss'] = agent_critic_loss_2.item()
        self.optimizer[agent_name]['double_critic'].step()
        self.scheduler[agent_name]['double_critic'].step()
        self.info_dict[agent_name]['critic'] = critic_info_dict
        
        # ----------- 更新优先级 ----------
        if priority_weight is not None:
            with torch.no_grad():
                td1 = (current_q_1 - Q_target_1).data
                td2 = (current_q_2 - Q_target_2).data
                batch_td_error = torch.abs(torch.min(td1, td2)+1e-5)
                self.batch_td_error = batch_td_error

        # --------- 采用soft update更新两个target critic网络 -----------
        soft_update(self.model[agent_name]['critic'], self.target_model[agent_name]['double_critic'], self.tau)
        soft_update(self.model[agent_name]['double_critic'], self.target_model[agent_name]['double_critic'], self.tau)


    def update_policy(self, agent_name, current_agent_obs, weights=1):
        # ---------- 这个函数用来更新策略网络中的参数,首先计算在给定输入下,得到动作以及动作出现的概率 ---------
        policy_info_dict = dict()
        try:
            actions_pred, log_prob = self.model[agent_name]['policy'].module.evaluate(current_agent_obs)
        except:
            actions_pred, log_prob = self.model[agent_name]['policy'].evaluate(current_agent_obs)
        with torch.no_grad():
            q1 = self.model[agent_name]['critic'](current_agent_obs, actions_pred)
            q2 = self.model[agent_name]['double_critic'](current_agent_obs, actions_pred)
            min_q_value = torch.min(q1, q2)
        agent_actor_loss = torch.mean(weights * (self.alpha * log_prob - min_q_value))
        self.optimizer[agent_name]['policy'].zero_grad()
        agent_actor_loss.backward()
        agent_policy_grad_norm = torch.nn.utils.clip_grad_norm_(self.model[agent_name]['policy'].parameters(), self.max_grad_norm)
        agent_policy_max_grads = dict()
        for name, value in self.model[agent_name]['policy'].named_parameters():
            if value.requires_grad:
                agent_policy_max_grads[name] = torch.max(value.grad).item()
        policy_info_dict['Layer_max_grads'] = agent_policy_max_grads
        policy_info_dict['Policy_loss'] = agent_actor_loss.item()
        policy_info_dict['Policy_grad_norm'] = agent_policy_grad_norm.item()
        policy_info_dict['Q_value_std'] = torch.std(min_q_value).item()
        self.optimizer[agent_name]['policy'].step()
        self.scheduler[agent_name]['policy'].step()
        # ------------- 如果不采用固定的alpha,则需要进行
        if 'entropy_coefficient' not in self.policy_config:
            pass
        self.info_dict[agent_name]['policy'] = policy_info_dict

    def step(self, training_data, priority_weights = None):
        self.info_dict = dict()
        for agent_name in training_data.keys():
            current_agent_obs = training_data[agent_name]['current_agent_obs']
            agent_current_action = training_data[agent_name]['actions']
            next_agent_obs = training_data[agent_name]['next_agent_obs']
            done = training_data[agent_name]['done']
            instant_reward = training_data[agent_name]['instant_reward']
            self.info_dict[agent_name] = dict()
            # --------------- 更新critic网络 -------------
            if self.categoraical_q_value:
                self.update_critic_when_using_categorical_critic(agent_name, current_agent_obs, agent_current_action, next_agent_obs, instant_reward, done, priority_weights)
            else:
                self.update_critic(agent_name, current_agent_obs, agent_current_action, next_agent_obs, instant_reward, done, priority_weights) 
            # -------------- 更新策略网路 ---------------
        if priority_weights is None:
            self.update_policy(agent_name, current_agent_obs)
            return self.info_dict
        else:
            self.update_policy(agent_name, current_agent_obs, priority_weights)
            return self.info_dict, self.batch_td_error


            