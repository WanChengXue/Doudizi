# ====== 这个函数用来评估一个智能体的性能 =========
from copy import deepcopy
import pickle
import torch
import lz4.frame as frame
import os
import sys
current_path = os.path.abspath(__file__)
root_path = '/'.join(current_path.split('/')[:-2])


from Utils.model_utils import deserialize_model, create_model
from Utils.data_utils import convert_data_format_to_torch_interference


class Agent:
    # 定义一个采样智能体，它能够完成的事情有：加载模型，保存模型，发送数据，给定状态计算动作
    def __init__(self, policy_config):
        # ----------- 这个地方创建出来的net_work其实是一个策略网络 ------- deepcopy(self.policy_config['agent'][model_type][agent_name])
        self.net_work = create_model(policy_config)
        
    def synchronize_model(self, model_path):
        # ---------- 这个函数是用来同步本地模型的 ----------
        deserialize_model(self.net_work, model_path)

    def compute(self, agent_obs):
        with torch.no_grad():
            action = self.net_work(agent_obs)
            return action

    def compute_prob_action(self, agent_obs):
        with torch.no_grad():
            log_prob, action = self.net_work(agent_obs)
            return log_prob, action

    def compute_state_value(self, agent_obs):
        with torch.no_grad():
            state_value = self.net_work(agent_obs)
        return state_value


    def compute_action_and_state_value(self, agent_obs):
        with torch.no_grad():
            log_prob, action, state_value = self.net_work(agent_obs)
        return log_prob, action, state_value


class Agent_manager:
    def __init__(self, config_dict, context, statistic, process_uid, logger, port_num=None):
        self.config_dict = config_dict
        self.statistic = statistic
        self.context = context
        self.logger = logger
        self.agent_nums = self.config_dict['env']['agent_nums']
        self.policy_config = self.config_dict['policy_config']
        self.agent_name_list = list(self.policy_config['agent']['policy'].keys())
        assert self.policy_config['eval_mode'], '-----测试模式下面, eval_model的值一定是True --------'
        self.logger.info('----------- 测试模式下面,数据不需要通过socket发送到dataserver --------')
        # ------- 构建info信息， 能够让智能体载入模型数据 --------
        self.model_info = self.generate_model_info()
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


    def generate_model_info(self):
        # ------------- 这个函数是通过对policy config进行解析，然后得到模型的相关信息 --------------
        model_info = dict()
        for model_type in self.policy_config['agent'].keys():
            if model_type == 'policy':
                model_info[model_type] = dict()
                for agent_name in self.policy_config['agent'][model_type].keys():
                    model_info[model_type][agent_name] = self.policy_config['agent'][model_type][agent_name]['model_path']
        return model_info

    def construct_agent(self):
        # ----------- 就算是多智能体场景，所有灯都共享参数 ——-----------
        self.agent = dict()
        for model_type in self.policy_config['agent'].keys():
            self.agent[model_type] = dict()
            for agent_name in self.policy_config['agent'][model_type].keys():
                policy_config = deepcopy(self.policy_config['agent'][model_type][agent_name])
                self.agent[model_type][agent_name] = Agent(policy_config)
        self.agent_name_list = list(self.agent['policy'].keys())   


    def send_data(self, packed_data):
        # ----------- 发送数据 ------------
        compressed_data = frame.compress(pickle.dumps(packed_data))
        with open("sampler_data", 'wb') as f:
            f.write(compressed_data)

    def compute(self, obs):
        # -------- 这个函数是默认多智能体环境，因此需要遍历agent字典来分别计算 ------------
        torch_type_data = convert_data_format_to_torch_interference(obs)
        network_decision = dict()
        for index, name in enumerate(torch_type_data.keys()):
            if self.parameter_sharing:
                network_output = self.agent['policy'][self.agent_name_list[0]].compute(torch_type_data[name])    
            else:
                network_output = self.agent['policy'][self.agent_name_list[index]].compute(torch_type_data[name])
            network_decision[name] = network_output.squeeze().numpy().tolist()
            assert isinstance(network_decision[name], list), '------------ 网络的输出结果需要是一个列表 -----------'
        # --------- 确保这个输出结果是一个一维list --------
        return network_decision

    def synchronize_model(self):
        for model_type in self.model_info.keys():
            for model_name in self.model_info[model_type]:
                self.agent[model_type][model_name].synchronize_model(self.model_info[model_type][model_name])

    def reset(self):
        self.logger.info('-------- 测试模式下不需要和learner相关的server进行数据的交互,直接本地读取就好, 模型信息为: {} ---------'.format(self.model_info))
        self.synchronize_model()
            

    
            


