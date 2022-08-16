import pickle
import torch
import random
import numpy as np
from threading import Lock

import os
import sys
current_path = os.path.abspath(__file__)
root_path = '/'.join(current_path.split('/')[:-2])
sys.path.append(root_path)

from Utils.SegmentTree import SumSegmentTree, MinSegmentTree

class data_buffer:
    def __init__(self, config):
        self.capacity = config['capacity']
        self.batch_size = config['batch_size']
        # --------- 这个参数决定，采样了的数据要不要从buffer里面删掉 -------
        self._lock = Lock()
        self.init_buffer()

    def init_buffer(self):
        self.current_buffer_length = 0
        self.buffer = []


    def sample_data(self):
        # ------- 这个函数将从这个buffer中采样一个batch的数据出去 -------
        random_batch = random.sample(self.buffer, self.batch_size)
        # ------- 将这个random batch列表缝合在一起，构成一个字典 ---------
        dict_output = convert_list_to_dict(random_batch)
        '''
            这个地方返回的结果是
            {
                'state': 
                    'sub_state': [np.array, np.array]或者[np.array]
            }
            因为没有办法区分状态是由一矩阵表示还是两个，就全部采用列表包起来
        '''
        return dict_output

    
    def append_data(self, recv_data):
        # --------- 这个函数是将数据添加到replaybuffer里面 ------------
        with self._lock:
            for data_point in recv_data:
                if self.current_buffer_length < self.capacity:
                    self.buffer.append(data_point)
                else:
                    self.buffer[self.current_buffer_length % self.capacity] = data_point
                self.current_buffer_length += 1


    @property
    def full_buffer(self):
        if self.current_buffer_length >= 0.2 * self.capacity:
            return True
        else:
            return False

    @property
    def buffer_size(self):
        return self.current_buffer_length


class prioritized_replay_buffer():
    def __init__(self, config):
        '''
            最开始的时候,设置p都是一样的,为1,使用均匀采样,而后根据TDError修改权重,具体计算为:
                p(i)^{\alpha} / (\sum_j p(j)^{\alpha})

            每一个点的采样权重的计算和TD Error有关, 具体的值为
                p(i) = |\delta_i| + \epsilon
            
            如果直接用上面的权重会出现一个问题,就是TD error大的样本点梯度大,导致学习不稳定,因此使用了重要性采样:
                w(i) = 1/((Np(i))^{\beta})
            
            N表示这个buffer的大小, 之后会进行正则化操作:
                w(i) = w(i)/(max_jw(j))
        '''
        self._alpha = config['alpha']
        # --------- beta是一个时变参数，随着更新，值发生改变，初始为0 ------
        self._beta = 0.4
        self._beta_increase_rate = config['beta_increase_rate']
        self._capacity = config['capacity']
        self._batch_size = config['batch_size']
        it_capacity = 1
        # ------- 如果说capacity不是2的整数，就扩充成2的整数 ------
        while it_capacity < self._capacity:
            it_capacity *= 2
        # -------- 这个地方建立了两个线段树，一个用来计算sum，一个用来计算最小值 --------
        self._it_sum = SumSegmentTree(it_capacity)
        self._it_min = MinSegmentTree(it_capacity)
        # -------- 在没有计算TD Error的时候，所有样本的权重都用1表示 ---------
        self._max_priority = 1.0
        self.buffer = [] # 定义一个空buffer
        self.current_buffer_length = 0
        self._lock = Lock()

    def append_data(self, recv_data):
        # --------- 这个函数是将数据添加到replaybuffer里面 ------------
        with self._lock:
            for data_point in recv_data:
                if self.current_buffer_length < self._capacity:
                    self.buffer.append(data_point)
                else:
                    self.buffer[self.current_buffer_length % self._capacity] = data_point
                buffer_index_pointer = (self.current_buffer_length) % self._capacity
                # --------- 修改这个样本的weight ------------
                self._add(buffer_index_pointer)
                self.current_buffer_length += 1

    def _update_beta(self):
        # --------- 这个函数周期性的来增加beta的值 ---------
        self._beta = min(1.0,self._beta + self._beta_increase_rate)

    def _add(self, idx):
        # --------- 这个函数的效果就是根据idx设置这个样本的权重 -----------
        # --------- 由于segment Tree中实现了__setitem__方法，因此下面的两个赋值都是调用了这个方法
        self._it_sum[idx] = self._max_priority ** self._alpha
        self._it_min[idx] = self._max_priority ** self._alpha

    def _sample_proportional(self):
        res = []
        for _ in range(self._batch_size):
            mass = random.random() * self._it_sum.sum(0, self.current_buffer_capcity - 1)
            idx = self._it_sum.find_prefixsum_idx(mass)
            res.append(idx)
        return res

    def sample_data(self):
        # --------- 采样出来一个index list -------
        self.current_buffer_capcity = min(self.current_buffer_length, self._capacity)
        idxes = self._sample_proportional()
        weights = []
        # ------- 这个地方计算 min_i[p_i^\alpha / (sum_j p_j^{\alpha})]
        p_min = self._it_min.min() / self._it_sum.sum()
        # ------- 正则化，原文中是 max_i 1/(Np_i^{beta}),等价于找到最小的p_i,然后计算 -----
        max_weight = (p_min * self.current_buffer_capcity) ** (-self._beta)
        for idx in idxes:
            p_sample = self._it_sum[idx] / self._it_sum.sum()
            weight = (p_sample * self.current_buffer_capcity) ** (-self._beta)
            weights.append(weight / max_weight)
        weights = np.array(weights, dtype=np.float32)
        samples = [self.buffer[idx] for idx in idxes]
        # -------- 将样本变成字典形式 ---------
        dict_output = convert_list_to_dict(samples)
        return dict_output, idxes, weights

    def update_priorities(self, idxes, priorities):
        # --------- 这个函数用来更新线段树的优先级 ------
        assert len(idxes) == len(priorities)
        for idx, priority in zip(idxes, priorities):
            assert priority > 0
            assert 0 <= idx < self.current_buffer_capcity
            self._it_sum[idx] = priority ** self._alpha
            self._it_min[idx] = priority ** self._alpha
            self._max_priority = max(self._max_priority, priority)
        # -------- 修改一下beta的值 ------
        self._update_beta()

    @property
    def full_buffer(self):
        if self.current_buffer_length >= 0.2 * self._capacity:
            return True
        else:
            return False

    @property
    def buffer_size(self):
        return self.current_buffer_length


def convert_data_format_to_torch_interference(obs_dict):
    # ------- 这个函数将除了legal action之外的值变成tensor ------
    torch_format_dict = dict()
    for key,value in obs_dict.items():
        if isinstance(value, dict):
            torch_format_dict[key] = convert_data_format_to_torch_interference(value)
        else:
            torch_format_dict[key] = torch.FloatTensor(value)
            # --------- 如果说这个value变成了一个tensor之后，维度是H，则在第0维度进行unsqueeze操作变成1×H的形式 --------
            if len(torch_format_dict[key].shape) == 1:
                torch_format_dict[key] = torch_format_dict[key].unsqueeze(0)
    return torch_format_dict


def convert_data_format_to_torch_trainig(training_batch, local_rank):
    # ----------- training_batch是numpy类型的数据，local rank表示的是使用这台电脑的第几张卡 -------------
    '''
    training_batch:
        state:
            state_1: R^{bs*c*w*h}
            state_2: R^{bs*h1}
            state_3: R^{bs*h2}
        action: ...
        reward:
        done:
        ...
    '''
    for key,value in training_batch.items():
        if isinstance(value, dict):
            training_batch[key] = convert_data_format_to_torch_trainig(value, local_rank)
        else:
            training_batch[key] = torch.FloatTensor(value).to(local_rank)
    return training_batch

def convert_list_to_dict(obs_list):
    # --------- 这个函数是将一个数据列表变成一个字典 -------
    '''
    比如说：
        obs_list = [{
            'state': [[],[]]
            'action': np.array(),
            'instant_reward': np.array(),
            'done': np.array
        }]
    '''
    '''
    如果说是这样子的:
        obs_list = [
            (dict_1, dict_2),(dict_1, dict_2),(dict_1, dict_2),...(dict_1, dict_2)
        ]
    然后使用了zip(*obs_list), 得到所有两个对象(dict_1,....,dict_1)
    '''
    # -----------
    instance_sample = obs_list[0]
    if isinstance(instance_sample, dict):
        sub_obs = dict()
        for key in instance_sample:
            # -------- 将这个子字典中的所有key获取到 -------
            sub_obs[key] = convert_list_to_dict([single_sample[key] for single_sample in obs_list])

    elif isinstance(instance_sample, (list, tuple)):
        # --------- 如果是一个由list，或者是tuple构成的列表，就合并 --------
        # [[a,b],[a,b],[a,b],...,[a,b]],或者是[(a,b),(a,b),...,(a,b)]形式，instance_sample[0]就是[a,b]，或者(a,b),因此递归拼接 
        # --------- 如果说instance_sample 中就是一个一个的element，则直接使用np.array就好，不需要维度拼接 -------
        if isinstance(instance_sample[0], dict):
            # --------- 如果传入的向量中每一个element都是一个字典，则考虑如何合并 ------
            flatten_list = []
            for instance_sample_element in obs_list:
                flatten_list += instance_sample_element
            return convert_list_to_dict(flatten_list)
            
        elif not isinstance(instance_sample[0], (list, tuple)):
            # -------- 这样子得到的就是batch_size * state_dim的矩阵
            sub_obs = np.array(obs_list)
        else: 
            sub_obs = [convert_list_to_dict(zip_object) for zip_object in zip(*obs_list)]
            if isinstance(sub_obs[0], dict):
                # ----------- 如果合并之后，sub_obs中的两个element都是字典，就update一下
                merge_dict = dict()
                for sub_dict in sub_obs:
                    merge_dict.update(sub_dict)
                return merge_dict
    else:
        if type(obs_list[0]) == np.ndarray:
            sub_obs = np.stack(obs_list, 0)
        else:
            sub_obs = np.array(obs_list)
        # --- 如果zip之后得到的是一个向量，就增加一个维度 ----------
        if len(sub_obs.shape) == 1:
            sub_obs = np.expand_dims(sub_obs, -1)
    return sub_obs
    

def squeeze(feature):
    '''convert any data format into a list. ignore its keys and takes all values
    Reshaper.squeeze({'a':1,'b':{'c':[2,3,4], 'd':{5}}}) -> [1,2,3,4,5]
    这个函数是一个递归的做法，将这个树上的所有叶子节点都拿出来构成一个列表
    '''
    if isinstance(feature, dict):
        return [
            value
            for key in sorted(feature)
            for value in squeeze(feature[key])
        ]
    elif isinstance(feature, (tuple, list, set)):
        return [
            value
            for item in feature
            for value in squeeze(item)
        ]
    elif isinstance(feature, (int, float, str)) or feature is None:
        return (feature,)
    else:
        # ValueError('Cannot process type: {}, {}'.format(type(feature), feature))
        return (feature,)

def reversed_action_dict(action_dict):
    # --------- 传入的action_dict的样子是{'split':{}, 'cycle':{}}
    reversed_dict = dict()
    for key in action_dict:
        for tls_id in action_dict[key]:
            if tls_id not in reversed_dict:
                reversed_dict[tls_id] = dict()
                reversed_dict[tls_id][key] = action_dict[key][tls_id]
            else:
                reversed_dict[tls_id][key] = action_dict[key][tls_id]
    return reversed_dict

def merge_dict(dict_one, dict_two):
    if isinstance(dict_one, dict):
        # ----------- 合并两个字典 --------
        merged_dict = dict()
        for key in dict_one.keys():
            assert key in dict_two, '---------- 合并的前提，这两个字典所有的key都是一样的 --------'
            merged_dict[key] = merge_dict(dict_one[key], dict_two[key])
        return merged_dict

    elif isinstance(dict_one, list):
        # --------- 如果dict_one是一个列表，dict_two不是，就将这个dict_two添加到dict_one里面 ------
        if not isinstance(dict_two, list):
            merged_list = dict_one + [dict_two]
        else:
            assert isinstance(dict_two, list), '----------- dict two要么是一个值，要么是一个列表 --------'
            merged_list = dict_one + dict_two
        return merged_list
    else:
        # ---------- 在这种情况下，必然传入的是两个数 ---------
        return [dict_one, dict_two]


def mean_dict(info_dict):
    if isinstance(info_dict, dict):
        mean_info_dict = dict()
        for key in info_dict.keys():
            mean_info_dict[key] = mean_dict(info_dict[key])
        return mean_info_dict
    else:
        assert isinstance(info_dict, list), '--------- 字典的value必然是一个list ---------'
        return np.mean(info_dict)


class OUNoise():
    def __init__(self, config):
        self.ou_teta = 0.15
        self.ou_mu = 0.0
        self.ou_sigma = 0.2
        self.ou_epsilon = 1.0

        self.action_dim = config['action_dim']
        self.low = config['action_low']
        self.high = config['action_high']

    def reset(self):
        self.state = np.zeros(self.action_dim)

    def step(self, current_actions):
        self.state += self.ou_teta * (self.ou_mu - self.state) # ou_teta为0.15
        self.state += self.ou_sigma * np.random.normal(size=self.action_dim)
        # a_state的修改为 ou_tehta * (mu - a_state) + ou_sigma * (长度为action_dim的正太噪声)
        ou_explore_action = current_actions + self.ou_epsilon * torch.FloatTensor(self.state).unsqueeze(0)
        actions = torch.clamp(ou_explore_action, self.low, self.high)
        return actions


def GAE_estimator_single_agent(data_list, gamma, tau, bootstrap_value):
    # ----------- 这个函数处理两种情况：要么multiagent_scenario,要么是single agent的数据 -------
    advantages = np.ones((len(data_list)), np.float32)
    deltas = np.ones((len(data_list)), np.float32)
    rewards = np.array([data["instant_reward"] for data in data_list], dtype=np.float32)
    dones = [data["done"] for data in data_list]
    try:
        values = [data["old_state_value"] for data in data_list]
    except:
        values = [data["old_obs_value"] for data in data_list]
    values = np.array(values, dtype=np.float32)  
    prev_value = bootstrap_value
    prev_advantage = 0
    # -------- 最开始的terminal state的时候，计算出来的a就是r - v -------
    for i in reversed(range(len(data_list))):
        deltas[i] = rewards[i] + gamma * prev_value * (1-dones[i]) - values[i]
        advantages[i] = deltas[i] + (gamma * tau) * prev_advantage  * (1-dones[i])
        prev_value = values[i]
        prev_advantage = advantages[i]
        data_list[i]['advantages'] = advantages[i]    
        data_list[i]['target_state_value'] = advantages[i] + values[i]

def GAE_estimator(data_dict, gamma, tau, bootstrap_value, multiagent_scenario):
    if multiagent_scenario:
        GAE_estimator_single_agent(data_dict, gamma, tau, bootstrap_value)
    else:
        for agent_name in data_dict.keys():
            GAE_estimator_single_agent(data_dict[agent_name], gamma, tau, bootstrap_value[agent_name])

