import argparse
import uuid
import zmq
from copy import deepcopy
import pickle
import numpy as np
import pathlib
import os
import sys
import time
import traceback
import queue
current_path = os.path.abspath(__file__)
root_path = '/'.join(current_path.split('/')[:-2])
sys.path.append(root_path)


from Utils.config import parse_config
from Utils.utils import setup_logger
from Utils.data_utils import GAE_estimator
from Worker.statistic import StatisticsUtils
from Worker.agent_manager import Agent_manager
from Env.env_utils import Environment
class sample_generator:
    def __init__(self, config_path, port_num=None):
        self.config_dict = parse_config(config_path, parser_obj='rollout')
        # ----------- 给这个进程唯一的标识码 ----------
        self.uuid = str(uuid.uuid4())
        self.policy_config = self.config_dict['policy_config']
        self.eval_mode = self.config_dict['policy_config'].get('eval_mode', False)
        self.agent_name_list = self.config_dict['env']['trained_agent_name_list']
        self.agent_nums = len(self.agent_name_list)
        self.context = zmq.Context()
        self.statistic = StatisticsUtils()
        logger_path = pathlib.Path(self.config_dict['log_dir'] + '/Worker_log/' + self.uuid[:6])
        self.logger = setup_logger('Worker_'+ self.uuid[:6], logger_path)
        if not self.eval_mode:
            # ---------- 这里之所以不直接继承Learner中的base server,是因为很可能采样端和learning端不在同一个机器上 --------
            self.log_sender = self.context.socket(zmq.PUSH)
            self.log_sender.connect("tcp://{}:{}".format(self.config_dict['log_server_address'], self.config_dict['log_server_port']))
            self.transmit_interval = self.policy_config['woker_transmit_interval']
            # -------   如果说训练类型是RL，就需要不断使用policy fetcher获取最新的模型 --------
            self.update_policy = self.policy_config['training_type'] != 'supervised_learning'
            self.buildin_ai = self.policy_config['buildin_ai']
            
        self.agent = Agent_manager(self.config_dict, self.context, self.statistic, self.uuid[:6], self.logger, port_num=port_num)
        self.agent.reset()
        self.multiagent_scenario = self.config_dict['env'].get('multiagent_scenario', False)
        self.policy_based_RL = self.config_dict['policy_config'].get('policy_based_RL', False)
        self.logger.info("---------------- 完成sampler的构建 ------------")

    def pack_data(self, data_dict, bootstrap_value):
        # ----------- 这个函数用来将数据进行打包，然后发送 ——----------
        if self.policy_based_RL:
            GAE_estimator(data_dict, self.policy_config['gamma'], self.policy_config['tau'], bootstrap_value, self.multiagent_scenario)
        # ----------- 使用SL算法，不需要使用GAE估计Advantage值 --------
        return data_dict

    def send_data(self, data_dict, bootstrap_value=None):
        # ----------- 这个函数用来发送数据到dataserver -------------
        packed_data = self.pack_data(data_dict, bootstrap_value)
        self.agent.send_data(packed_data)

    def _generate_obs(self, centralized_state):
        state_dict = dict()
        if self.agent_nums == 1:
            state_dict[self.agent_name_list[0]] = deepcopy(centralized_state)
        else:
            pass
        return state_dict

    def _revise_episodic_dict(self, episodic_dict, n_step_reward_list, done, next_agent_obs, next_centralized_state=None):
        discount_ratio = self.policy_config.get('gamma', 1)
        episodic_dict['done'] = done
        episodic_dict['next_agent_obs'] = deepcopy(next_agent_obs) 
        if next_centralized_state is not None:
            episodic_dict['next_centralized_state'] = deepcopy(next_centralized_state)
        # ----------- 反序遍历这个list，计算reward --------
        n_step_reward = 0
        for instant_reward in reversed(n_step_reward_list):
                n_step_reward *= discount_ratio
                n_step_reward += instant_reward
        episodic_dict['instant_reward'] = n_step_reward
        episodic_dict['next_state_action_length'] = episodic_dict['next_agent_obs']['x'].shape[0]
        n_step_reward_list.pop(0)

    def _app_episodic_dict(self, n_step_state_queue, data_dict, n_step_reward_list, done, next_agent_obs, next_centralized_state=None):
        if done:
            # ------- 如果trajectory结束了，就要把这个队列中的所有值都吐出来 ----
            while not n_step_state_queue.empty():
                episodic_dict = n_step_state_queue.get()
                if isinstance(data_dict, dict):
                    # -------- 传入的data_dict有列表形式和字典形式两种 ------
                    for agent_name in data_dict:
                        self._revise_episodic_dict(episodic_dict[agent_name], n_step_reward_list[agent_name], done, next_agent_obs[agent_name])
                        data_dict[agent_name].append(episodic_dict[agent_name])
                else:
                    self._revise_episodic_dict(episodic_dict, n_step_reward_list, done, next_agent_obs, next_centralized_state)
                    data_dict.append(episodic_dict)
        else:
            # ------ 不然就吐一个值 -------
            episodic_dict = n_step_state_queue.get()
            if isinstance(data_dict, dict):
                for agent_name in self.agent_name_list:
                    self._revise_episodic_dict(episodic_dict[agent_name], n_step_reward_list[agent_name], done, next_agent_obs[agent_name])
                    data_dict[agent_name].append(episodic_dict[agent_name])
            else:
                self._revise_episodic_dict(episodic_dict, n_step_reward_list, done, next_agent_obs, next_centralized_state)
                data_dict.append(episodic_dict)


    def rollout_one_episode_evaluate(self):
        self.logger.info("------------- evaluate 程序 {} 开始启动 ------------".format(self.uuid[:6]))
        self.rollout_one_episode_evaluate_by_agent()
        # --------- 这个地方使用传统算法跑一次 ----------

    def select_data(self, data_dict, index):
        selected_dict ={}
        for key in data_dict:
            selected_dict[key] = data_dict[key][index]
        return selected_dict
    
    def _revised_all_reward(self, data_dict, sum_reward):
        for sample_point in data_dict[self.agent_name_list[0]]:
            sample_point['instant_reward'] = sum_reward

    def rollout_one_episode_multi_agent_scenario(self):
        # ----------- 这个rollout函数专门用来给RL算法进行采样，这个只用来给MARL场景进行采样,基于CTDE范式 --------------------
        self.logger.info('------------- 采样sampler {} 开始启动, 样本用来传递给MARL算法 --------------'.format(self.uuid[:6]))
        start_env_time = time.time()
        self.env = Environment()
        self.env.set_buildin_ai(self.agent.agent[self.buildin_ai], self.agent_name_list[0])
        current_centralized_state = self.env.reset()
        # --------- 设置内置AI，和需要被训练的智能体 ------
        
        self.logger.info("-------------------- env reset ------------------")
        reward_list = []
        n_step = self.policy_config.get('n_step',1)
        # ------------ 开一个队列，用来存放一下中间的step ------
        n_step_state_queue = queue.Queue(maxsize=n_step)
        if self.multiagent_scenario:
            data_dict = []
            n_step_reward_list = []
        else:
            data_dict = dict()
            n_step_reward_list = dict()
            for agent_name in self.agent_name_list:
                data_dict[agent_name] = []
                n_step_reward_list[agent_name] = []
        step = 0
        while True:
            current_agent_obs = self._generate_obs(current_centralized_state)
            if self.policy_based_RL:
                if self.multiagent_scenario:
                    centralized_state_value = self.agent.compute_state_value(current_centralized_state)
                else:
                    obs_value_dict = self.agent.compute_state_value(current_agent_obs)
                action_dict, log_prob_dict = self.agent.compute(current_agent_obs)
            else:
                action_dict = self.agent.compute(current_agent_obs)
            action = action_dict[self.agent_name_list[0]]['action']
            next_centralized_state, instant_reward, done = self.env.step(action)
            if done:
                next_centralized_state = current_centralized_state
            next_agent_obs = self._generate_obs(next_centralized_state)
            step += 1
            if self.multiagent_scenario:
                episodic_dict = dict()
                episodic_dict['current_agent_obs'] = current_agent_obs
                episodic_dict['current_centralized_state'] = deepcopy(current_centralized_state)
                episodic_dict['actions'] = deepcopy(action)
                if self.policy_based_RL:
                    episodic_dict['old_state_value'] = centralized_state_value
                    episodic_dict['log_prob_dict'] = deepcopy(log_prob_dict)
                n_step_state_queue.put(episodic_dict)
                n_step_reward_list.append(instant_reward)
                if n_step_state_queue.full() or done:
                    # ------ 如果队列满了，或者说结束了episode，就从这个队列中拿数据 ------
                    self._app_episodic_dict(n_step_state_queue, data_dict, n_step_reward_list, done, next_agent_obs, next_centralized_state)
            else:
                episodic_dict = dict()
                for agent_name in self.agent_name_list:
                    episodic_dict[agent_name] = dict()
                    episodic_dict[agent_name]['current_agent_obs'] = deepcopy(self.select_data(current_agent_obs[agent_name], action))
                    n_step_reward_list[agent_name].append(instant_reward)
                    episodic_dict[agent_name]['next_agent_obs'] = deepcopy((next_agent_obs[agent_name]))
                    if self.policy_based_RL:
                        episodic_dict[agent_name]['old_obs_value'] = obs_value_dict[agent_name]
                        episodic_dict[agent_name]['old_log_prob'] = log_prob_dict[agent_name]
                n_step_state_queue.put(episodic_dict)
                if n_step_state_queue.full() or done:
                    self._app_episodic_dict(n_step_state_queue, data_dict, n_step_reward_list, done, next_agent_obs)
            reward_list.append(instant_reward)
            if done:
                # -------------- 需要计算一下bootstrap value -------------
                if self.policy_based_RL:
                    if self.multiagent_scenario:
                        bootstrap_value = self.agent.compute_state_value(next_centralized_state)
                    else:
                        bootstrap_value = self.agent.compute_state_value(next_agent_obs)
                else:
                    bootstrap_value = None
                self._revised_all_reward(data_dict, sum(reward_list))
                self.send_data(data_dict, bootstrap_value)
                self.agent.step()
                if self.policy_config.get('ou_enabled', False):
                    self.agent._construct_ou_noise_explorator()
                break
            # --------- 状态更新  ---------
            current_centralized_state = next_centralized_state
            # if step % self.transmit_interval == 0:
            #     self.logger.info("----------- worker端口发送数据 ---------")
            #     self.send_data(data_dict)
            #     # self.logger.info("-------- 数据长度为：{} ------".format(len(data_dict)))
            #     if self.multiagent_scenario:
            #         data_dict = []
            #     else:
            #         data_dict = dict()
            #         for agent_name in list(self.policy_config['agent']['policy'].keys()):
            #             data_dict[agent_name] = []
            #     self.agent.step()

        end_env_time = time.time()
        self.statistic.append('Worker/sample_time_per_episode/{}'.format(self.config_dict['policy_name']), (end_env_time-start_env_time)/3600)
        self.statistic.append('result/sum_instant_reward/{}'.format(self.config_dict['policy_name']), sum(reward_list))
        self.statistic.append('Worker/sample_step_per_episode/{}'.format(self.config_dict['policy_name']), step)
        # self.statistic.append('Worker/sample_cycle_time_per_epsode/{}'.format(self.config_dict['policy_name']), sum(cycle_list)/3600.0)
        self.statistic.append('result/sum_instant_reward/{}'.format(self.config_dict['policy_name']), sum(reward_list))
        result_info = {'worker_id': self.uuid}
        for key, value in self.statistic.iter_key_avg_value():
            result_info[key] = value
        self.log_sender.send(pickle.dumps([result_info]))
        self.statistic.clear()
            

    def run(self):
        if self.eval_mode:
            self.rollout_one_episode_evaluate()
        else:
            try:
                while True:
                    if self.update_policy:
                        self.rollout_one_episode_multi_agent_scenario()
                    else:
                        pass

            except Exception as e:
                error_str = traceback.format_exc()
                self.logger.error(e)
                self.logger.error(error_str)
                if not self.eval_mode:
                    # -------------- 如果不是评估模式，将报错信息也发送到logserver处 ---------
                    error_message = {"error_log": error_str}
                    p = pickle.dumps([error_message])
                    self.log_sender.send(p)
                exit()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default='Config/Training/DQN_config.yaml')
    # Independent_D4PG_heterogeneous_network_eval_config
    # heterogeneous_network_eval_config
    args = parser.parse_args()
    worker = sample_generator(args.config_path)
    worker.run()
