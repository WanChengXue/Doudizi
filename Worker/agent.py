import pathlib
import pickle
import lz4.frame as frame
import torch
import zmq
import random
import os
import sys
from copy import deepcopy
current_path = os.path.abspath(__file__)
root_path = '/'.join(current_path.split('/')[:-2])
sys.path.append(root_path)
import numpy as np

from Worker.policy_fetcher import fetcher
from Utils.model_utils import deserialize_model, create_model
from Utils.data_utils import OUNoise, convert_data_format_to_torch_interference


class Agent:
    def __init__(self, model_config):
        # ------------- 创建网络，只需要决策网络 --------------------
        self.model_config = model_config
        self.net_work = create_model(model_config)
    
    def synchronize_model(self, model_path):
        # ------------ 这个函数用来同步本地策略模型 ——---------
        deserialize_model(self.net_work, model_path)


    def compute(self, obs):
        with torch.no_grad():
            output = self.net_work(obs)
        return output

    def init_ou_process(self, ou_config):
        self.ou_theta = ou_config['ou_theta']
        self.ou_sigma = ou_config['ou_sigma']
        self.ou_epsilon = ou_config['ou_epsilon']
        self.ou_mu = ou_config['ou_mu']
        self.last_action = torch.zeros(1, self.model_config['action_dim'])

    def explore_by_ou_process(self, current_action):
        # --------- 这个地方是使用OU过程来进行动作的探索 -------
        self.last_action += self.ou_theta
        self.last_action += self.ou_theta * (self.ou_mu - self.last_action)
        self.last_action += self.ou_sigma * torch.randn(1, self.model_config['action_dim'])
        current_action += self.ou_epsilon * self.last_action
        actions = torch.clamp(current_action, -1, 1)
        return actions

    def explore_by_gaussian(self, current_action):
        random_noise = torch.randn(1, self.model_config['action_dim'])
        return torch.clamp(0.3 * random_noise + current_action, -1, 1)

class Agent_manager:
    def __init__(self, config_dict, context, statistic, process_uid, logger, port_num=None):
        self.config_dict = config_dict
        self.statistic = statistic
        self.context = context
        self.agent_nums = self.config_dict['env']['agent_nums']
        self.policy_config = self.config_dict['policy_config']
        self.agent_name_list = list(self.policy_config['agent']['policy'].keys())
        self.model_info = None
        self.logger = logger
        self.init_socket(port_num)
        self.policy_fetcher = fetcher(self.context, self.config_dict, self.statistic, process_uid, self.logger)
        # --------- 这两个参数是设置的作用是，在多智能体场景中，所有智能体都是homogeneous的条件下，才能设置parameter sharing为True -------
        self.parameter_sharing = self.policy_config['parameter_sharing']
        self.homogeneous_agent = self.policy_config['homogeneous_agent']
        # ----------- 定义训练类型 ------------------
        self.training_type = self.policy_config['training_type']
        self.construct_agent()
        if self.training_type == 'RL':
            # --------- 关于critic的配置上面，有policy和critic连在一起，有中心化的critic，还有一个policy就带一个critic那种 -----------------
            self.centralized_critic = self.policy_config['centralized_critic'] # 这个变量控制中心化训练，当为True就表示初始化一个critic
            self.seperate_critic = self.policy_config['seperate_critic'] # 这个变量表示每一个智能体有一个分离的policy和ciritci
        self.logger.info("--------------- 完成agentmanager的创建 ------------")

    def init_socket(self, port_num):
        # ----------- 确定下来这个worker要链接哪一个端口 ------------
        self.data_sender = self.context.socket(zmq.PUSH)
        device_number_per_machine = self.policy_config['device_number_per_machine']
        server_number_per_device = self.policy_config['server_number_per_device']
        machine_number = len(self.policy_config['machine_list'])
        total_port_per_machine = device_number_per_machine * server_number_per_device  
        # ------------ 先选择机器，然后选择端口 ---------------
        select_machine_index = random.randint(0, machine_number-1) 
        if port_num is not None:
            random_port = port_num % server_number_per_device
        else:
            random_port = random.randint(0, total_port_per_machine-1) 
        connect_port =  random_port + self.policy_config['start_data_server_port'] + total_port_per_machine * select_machine_index
        self.logger.info('-----------数据server开始port {}, random_port {} -----------'.format(self.policy_config['start_data_server_port'], random_port))
        connect_ip = self.policy_config['machine_list'][select_machine_index]
        # connect_port = 9527
        self.data_sender.connect("tcp://{}:{}".format(connect_ip, connect_port))
        self.logger.info("------------ 套接字初始化成功，数据发送到的ip为: tcp://{}:{}--------------".format(connect_ip, connect_port))


    def construct_agent(self):
        # ----------- 就算是多智能体场景，所有灯都共享参数 ——-----------
        self.agent = dict()
        for model_type in self.policy_config['agent'].keys():
            self.agent[model_type] = dict()
            for agent_name in self.policy_config['agent'][model_type].keys():
                policy_config = deepcopy(self.policy_config['agent'][model_type][agent_name])
                self.agent[model_type][agent_name] = Agent(policy_config)
        self.agent_name_list = list(self.agent['policy'].keys())
        if self.policy_config.get('ou_enabled', False):
            self._construct_ou_noise_explorator()
        
    def _construct_ou_noise_explorator(self):
        self.ou_explorator = dict()
        for agent_number in range(self.config_dict['env']['agent_nums']):
            self.ou_explorator['agent_{}'.format(agent_number)] = OUNoise(self.policy_config['ou_config'])
            self.ou_explorator['agent_{}'.format(agent_number)].reset()

    def send_data(self, packed_data):
        # ----------- 发送数据 ------------
        compressed_data = frame.compress(pickle.dumps(packed_data))
        self.data_sender.send(compressed_data)

    def compute(self, obs):
        # ---------- 这里都是使用的DDPG-based的算法，因此是不需要进行计算V或者Q进行advantage的计算 ------
        # -------- 这个函数是默认多智能体环境，因此需要遍历agent字典来分别计算 ------------
        torch_type_data = convert_data_format_to_torch_interference(obs)
        network_decision = dict()
        for index, name in enumerate(torch_type_data.keys()):
            if self.parameter_sharing:
                network_output = self.agent['policy'][self.agent_name_list[0]].compute(torch_type_data[name])    
            else:
                network_output = self.agent['policy'][self.agent_name_list[index]].compute(torch_type_data[name])
            network_output = network_output.squeeze().numpy()
            if self.policy_config['ou_enabled']:
                network_output = self.ou_explorator['agent_{}'.format(index)].step(network_output)
            else:
                # network_output = self.agent['policy'][self.agent_name_list[index]].explore_by_gaussian(network_output)
                pass
            network_decision[name] = network_output.tolist()
            assert isinstance(network_decision[name], list), '------------ 网络的输出结果需要是一个列表 -----------'
        # --------- 确保这个输出结果是一个一维list --------
        return network_decision

    def synchronize_model(self):
        # ---------- 在训练模式下，先将模型的最新信息获取下来 -----------
        model_info = self.policy_fetcher.reset()
        if model_info is not None:
            self.logger.info("----------- 模型重置，使用model_fetcher到的模型数据:{} -----------".format(model_info))
            # ---------- 当获取到了最新的模型后，进行模型之间的同步 ------------
            for model_type in self.policy_fetcher.model_path.keys():
                for model_name in self.policy_fetcher.model_path[model_type]:
                    try:
                        self.agent[model_type][model_name].synchronize_model(self.policy_fetcher.model_path[model_type][model_name])
                    except:
                        self.logger.info('-------- 模型类型为{},模型名称为{},载入的模型路径为:{} ------'.format(model_type, model_name, self.policy_fetcher.model_path[model_type][model_name]))
        else:
            self.logger.info("------------- agent调用reset函数之后没有获取到新模型,检测fetcher函数 ------------")

    def reset(self):
        # --------- 模型重置 ------------
        self.synchronize_model()
            

    def step(self):
        # -------------- 这个函数是在训练的过程中，下载最新的模型 ---------------
        '''
        model_info的样子
            {
                'policy_path': string, 如果网络参数不同享,否则是一个字典 {'agent_0': string, 'agent_1': string}
                'critic_path': 只有SL训练的时候才会有这个key，如果是中心化训练，value是一个字符串，分布式训练的话是一个字典，如果critic和policy在一个网络也是没有这个key的
            }
        '''
        self.synchronize_model()
    
    def get_model_info(self):
        return self.model_info


if __name__ == '__main__':
    import argparse
    from Worker.statistic import StatisticsUtils
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='/Env/default_config.yaml', help='yaml format config')
    args = parser.parse_args()
    # ------------- 构建绝对地址 --------------
    # Linux下面是用/分割路径，windows下面是用\\，因此需要修改
    # abs_path = '/'.join(os.path.abspath(__file__).split('/')[:-2])
    abs_path = '/'.join(os.path.abspath(__file__).split('/')[:-2])
    concatenate_path = abs_path + args.config_path
    context = zmq.Context()
    statistic = StatisticsUtils()
    from Utils.config import parse_config
    from Utils.config import setup_logger
    config_dict = parse_config(concatenate_path)
    logger_path = pathlib.Path(config_dict['log_dir']+ '/sampler/test_agent_log')
    logger = setup_logger('Test_agent', logger_path)
    test_agent_manager = Agent_manager(config_dict, context, statistic, logger)
    test_agent_manager.reset()
