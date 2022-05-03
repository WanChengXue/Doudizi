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

import gym
import pybullet_envs
from Utils.config import parse_config
from Utils.utils import setup_logger
from Worker.statistic import StatisticsUtils
from Worker.agent_manager import Agent_manager

class sample_generator:
    def __init__(self, config_path, port_num=None):
        self.config_dict = parse_config(config_path, parser_obj='rollout')
        # ----------- 给这个进程唯一的标识码 ----------
        self.uuid = str(uuid.uuid4())
        self.policy_config = self.config_dict['policy_config']
        self.eval_mode = self.config_dict['policy_config'].get('eval_mode', False)
        self.agent_nums = self.config_dict['env']['agent_nums']
        self.agent_name_list = self.config_dict['env']['agent_name_list']
        self.context = zmq.Context()
        self.statistic = StatisticsUtils()
        logger_path = pathlib.Path(self.config_dict['log_dir'] + '/Worker_log/' + self.uuid[:6])
        self.logger = setup_logger('Worker_'+ self.uuid[:6], logger_path)
        if self.eval_mode:
            self.init_random_seed()
        else:
            # ---------- 这里之所以不直接继承Learner中的base server,是因为很可能采样端和learning端不在同一个机器上 --------
            self.log_sender = self.context.socket(zmq.PUSH)
            self.log_sender.connect("tcp://{}:{}".format(self.config_dict['log_server_address'], self.config_dict['log_server_port']))
            self.transmit_interval = self.policy_config['woker_transmit_interval']
        # -------   如果说训练类型是RL，就需要不断使用policy fetcher获取最新的模型 --------
        self.update_policy = self.policy_config['training_type'] != 'supervised_learning'
        self.agent = Agent_manager(self.config_dict, self.context, self.statistic, self.uuid[:6], self.logger, port_num=port_num)
        self.agent.reset()
        self.env = gym.make(self.config_dict['env']['env_name'])
        self.multiagent_scenario = self.config_dict['env'].get('multiagent_scenario', False)
        self.logger.info("---------------- 完成sampler的构建 ------------")

    def pack_data(self, data_dict):
        # ----------- 这个函数用来将数据进行打包，然后发送 ——----------
        # ----------- 使用SL算法，不需要使用GAE估计Advantage值 --------
        return data_dict

    def send_data(self, data_dict):
        # ----------- 这个函数用来发送数据到dataserver -------------
        packed_data = self.pack_data(data_dict)
        self.agent.send_data(packed_data)

    def _generate_obs(self, centralized_state):
        state_dict = dict()
        if self.agent_nums == 1:
            state_dict[self.agent_name_list[0]] = deepcopy(centralized_state)
        else:
            pass
        return state_dict

    def _generate_action(self, action_dict):
        if self.agent_nums == 1:
            return action_dict[self.agent_name_list[0]]
        else:
            pass
            return action_dict

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

    def rollout_one_episode_evaluate_by_agent(self):
        self.env.reset()
        self.env.seed(0)
        
        waiting_time = []
        instant_reward_list = []
        start_time = time.time()
        rosea_action_dict = dict()
        rosea_action_dict['step_time'] = []
        rosea_action_dict['cum_reward'] =  []
        step = 0
        while True:
            features = self.env.produce_feature()
            current_obs = self.generate_obs(features['observation'])
            action = self.agent.compute(current_obs)
            # -------- 将这个action分离开来，变成rosea需要的样子 ---------
            rosea_action = self.convert_net_action_to_rosea_action(action)
            rosea_action_dict['step_{}'.format(step)] = rosea_action
            _, instant_reward, done, info = self.env.step(rosea_action)
            rosea_action_dict['step_time'].append(info['current_time'])
            rosea_action_dict['cum_reward'].append(info['cum_reward'])
            instant_reward_list.append(instant_reward)
            waiting_time.append(features['reward']['wait_time_per_vehicle'])
            step += 1
            if done or (time.time()-start_time)/3600>2:
                features = self.env.produce_feature()
                waiting_time.append(features['reward']['wait_time_per_vehicle'])
                self.logger.info('-------------- 使用RL智能体进行evaluate耗时: {} --------------'.format((time.time() - start_time)/3600))
                break
        # ------- 结果存放到本地，路径为 Exp/Result/Evaluate/multi-point_heterogeneous_policy_instant_reward.npy
        reward_saved_path = os.path.join(self.policy_config['result_save_path'], self.config_dict['policy_name']+'_instant_reward.npy')
        waiting_time_saved_path = os.path.join(self.policy_config['result_save_path'], self.config_dict['policy_name']+'_waiting_time.npy')
        rosea_action_saved_path = os.path.join(self.policy_config['result_save_path'], self.config_dict['policy_name'] + '_rosea_action.pkl')
        np.save(reward_saved_path, np.array(instant_reward_list))
        np.save(waiting_time_saved_path, np.array(waiting_time))
        open_file = open(rosea_action_saved_path, 'wb')
        pickle.dump(rosea_action_dict, open_file)
        open_file.close()

    def rollout_one_episode_evaluate(self):
        self.logger.info("------------- evaluate 程序 {} 开始启动 ------------".format(self.uuid[:6]))
        self.rollout_one_episode_evaluate_by_agent()
        # --------- 这个地方使用传统算法跑一次 ----------

    def rollout_one_episode_multi_agent_scenario(self):
        # ----------- 这个rollout函数专门用来给RL算法进行采样，这个只用来给MARL场景进行采样,基于CTDE范式 --------------------
        self.logger.info('------------- 采样sampler {} 开始启动, 样本用来传递给MARL算法 --------------'.format(self.uuid[:6]))
        start_env_time = time.time()
        current_centralized_state = self.env.reset()
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
            action_dict = self.agent.compute(current_agent_obs)
            action = self._generate_action(action_dict)
            next_centralized_state, instant_reward, done, info = self.env.step(action)
            next_agent_obs = self._generate_obs(next_centralized_state)
            step += 1
            if self.multiagent_scenario:
                episodic_dict = dict()
                episodic_dict['current_agent_obs'] = current_agent_obs
                episodic_dict['current_centralized_state'] = deepcopy(current_centralized_state)
                episodic_dict['actions'] = deepcopy(action)
                n_step_state_queue.put(episodic_dict)
                n_step_reward_list.append(instant_reward)
                if n_step_state_queue.full() or done:
                    # ------ 如果队列满了，或者说结束了episode，就从这个队列中拿数据 ------
                    self._app_episodic_dict(n_step_state_queue, data_dict, n_step_reward_list, done, next_agent_obs, next_centralized_state)
            else:
                episodic_dict = dict()
                for agent_name in self.agent_name_list:
                    episodic_dict[agent_name] = dict()
                    episodic_dict[agent_name]['current_agent_obs'] = deepcopy(current_agent_obs[agent_name])
                    episodic_dict[agent_name]['actions'] = deepcopy(action_dict[agent_name])
                    n_step_reward_list[agent_name].append(instant_reward)
                n_step_state_queue.put(episodic_dict)
                if n_step_state_queue.full() or done:
                    self._app_episodic_dict(n_step_state_queue, data_dict, n_step_reward_list, done, next_agent_obs)
            # --------- 状态更新  ---------
            current_centralized_state = next_centralized_state
            reward_list.append(instant_reward)
            if done:
                self.send_data(data_dict)
                self.agent.step()
                if self.policy_config.get('ou_enabled', False):
                    self.agent._construct_ou_noise_explorator()
                break

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
    parser.add_argument("--config_path", type=str, default='Env/SAC_config.yaml')
    # Independent_D4PG_heterogeneous_network_eval_config
    # heterogeneous_network_eval_config
    args = parser.parse_args()
    worker = sample_generator(args.config_path)
    worker.run()
