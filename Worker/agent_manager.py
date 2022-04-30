import pathlib
import pickle
import lz4.frame as frame
import zmq
import random
import os
import sys
from copy import deepcopy
current_path = os.path.abspath(__file__)
root_path = '/'.join(current_path.split('/')[:-2])
sys.path.append(root_path)


from Worker import agent
from Worker.policy_fetcher import fetcher
from Utils.data_utils import convert_data_format_to_torch_interference



class Agent_manager:
    def __init__(self, config_dict, context, statistic, process_uid, logger, port_num=None):
        self.config_dict = config_dict
        self.agent_nums = self.config_dict['env']['agent_nums']
        self.policy_config = self.config_dict['policy_config']
        # ---------- 智能体的名称,统一用env字段中的 ---------
        self.agent_name_list = self.config_dict['env']['agent_name_list']
        self.model_info = None
        self.logger = logger
        # ---------- eavl mode ----------
        self.eval_mode = self.policy_config.get('eval_mode', False)
        if self.eval_mode:
            # ----------- 在测试模式下,需要先生成模型信息然后进行加载 ----------
            self.model_info = self.generate_model_info()
        else:
            # -------- 只要在训练模式下,才需要初始化套接字以及策略更新对象 ---------
            self.statistic = statistic
            self.context = context
            self.init_socket(port_num)
            self.policy_fetcher = fetcher(self.context, self.config_dict, self.statistic, process_uid, self.logger)
        # ----------- 定义训练类型 ------------------
        self.training_type = self.policy_config['training_type']
        self.parameter_sharing = self.policy_config.get('parameter_sharing', False)
        
        if self.training_type == 'RL':
            # --------- 关于critic的配置上面，有policy和critic连在一起，有中心化的critic，还有一个policy就带一个critic那种 -----------------
            self.centralized_critic = self.policy_config['centralized_critic'] # 这个变量控制中心化训练，当为True就表示初始化一个critic
            self.seperate_critic = self.policy_config['seperate_critic'] # 这个变量表示每一个智能体有一个分离的policy和ciritci
            self.multiagent_scenario = self.config_dict['env'].get('multiagent_scenario', False)
        self.construct_agent()
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
        self.data_sender.connect("tcp://{}:{}".format(connect_ip, connect_port))
        self.logger.info("------------ 套接字初始化成功，数据发送到的ip为: tcp://{}:{}--------------".format(connect_ip, connect_port))


    def construct_agent(self):
        # ----------- 智能体名称,依次为在环境中获得 ——-----------
        self.agent = dict()
        for agent_name in self.agent_name_list:
            self.agent[agent_name] =  agent.get_agent(self.policy_config['agent_name'])(self.policy_config['agent'][agent_name])  

        if self.multiagent_scenario:
            # -------- TODO 在多智能体场景下启用,还是必须要计算中心状态的Q值才需要 -----------
            pass

    def generate_model_info(self):
        # ------------- 这个函数是通过对policy config进行解析，然后得到模型的相关信息, 仅在测试模式下使用 --------------
        model_info = dict()
        model_type_list = ['policy', 'critic']
        for agent_name in self.agent_name_list:
            model_info[agent_name] = dict()
            for model_type in self.policy_config['agent'][agent_name].keys():
                if model_type in model_type_list:
                    model_info[agent_name] = self.policy_config['agent'][agent_name][model_type]['model_path']

        if self.multiagent_scenario:
            pass
        return model_info


    def send_data(self, packed_data):
        # ----------- 发送数据 ------------
        compressed_data = frame.compress(pickle.dumps(packed_data))
        self.data_sender.send(compressed_data)



    def compute(self, obs):
        torch_type_data = convert_data_format_to_torch_interference(obs)
        network_decision = dict()
        for agent_name in torch_type_data.keys():
            # --------- torch_type_data的key必须和agent name list是一致的 ----------
            assert agent_name in self.agent_name_list, '----- torch_type_data的key和agent name list必须要一致 --------'
            if self.eval_mode:
                network_output = self.agent[agent_name].compute_action_eval_mode(torch_type_data[agent_name])
            else:
                network_output = self.agent[agent_name].compute_action_training_mode(torch_type_data[agent_name])
            network_output = network_output.squeeze().numpy()
            network_decision[agent_name] = network_output.tolist()
            assert isinstance(network_decision[agent_name], list), '------------ 网络的输出结果需要是一个列表 -----------'
        # --------- 确保这个输出结果是一个一维list --------
        return network_decision


    def synchronize_model(self):
        if self.eval_mode:
            model_info = self.model_info
        else:
            model_info = self.policy_fetcher.reset()
            if model_info is not None:
                self.logger.info("----------- 模型重置，使用model_fetcher到的模型数据:{} -----------".format(model_info))
            else:
                self.logger.info("------------- agent调用reset函数之后没有获取到新模型,检测fetcher函数 ------------")


        if model_info is not None:
            # ---------- 当获取到了最新的模型后，进行模型之间的同步 ------------
            for agent_name in self.agent_name_list:
                self.agent[agent_name].synchronize_model(model_info[agent_name])

            if self.multiagent_scenario:
                pass


    def reset(self):
        # --------- 模型重置 ------------
        self.synchronize_model()
            

    def step(self):
        # -------------- 这个函数是在训练的过程中，下载最新的模型 ---------------
        '''
        model_info的样子
            {
                'agent_0': {'policy': string , 'critic' : string (optional)}
                ...
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
