import argparse
import time
import pickle
import importlib
import pyarrow.plasma as plasma
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import zmq
import pathlib
import os
import sys
import queue
from copy import deepcopy

current_path = os.path.abspath(__file__)
root_path = '/'.join(current_path.split('/')[:-2])
sys.path.append(root_path)


from Learner.base_server import base_server
from Utils.utils import create_folder, setup_logger
from Utils.model_utils import serialize_model, deserialize_model, create_model
from Utils.utils import generate_plasma_id, generate_plasma_id_for_PEB
from Utils.data_utils import convert_data_format_to_torch_trainig

def get_algorithm_cls(name):
    # 这个函数返回的是一个类，返回一个具体的算法类
    return importlib.import_module("Algorithm.{}".format(name)).get_cls()


class learner_server(base_server):
    # 这个函数是用来对网络进行参数更新
    def __init__(self, args):
        super(learner_server, self).__init__(args.config_path)
        self.policy_config = self.config_dict['policy_config']
        # ----------- 这个world size表示的是有多少张卡 ------------
        self.world_size = args.world_size
        # ------------ 这个变量表示这是第几个learner ----------
        self.policy_name = self.config_dict['policy_name']
        # ------------ 这个global_rank表示的是这个learner使用的是第几张卡，绝对索引 ----------
        self.global_rank = args.rank
        self.local_rank = self.global_rank % self.policy_config['device_number_per_machine']
        logger_path = pathlib.Path(self.config_dict['log_dir'] + '/learner_log/{}_{}'.format(self.policy_name,self.local_rank)) 
        self.logger = setup_logger('LearnerServer_log_{}'.format(self.local_rank), logger_path)
        self.logger.info("============== 开始构造learner server，这个server的全局id是:{}, 具体到某一个机器上的id是: {} ===========".format(self.global_rank, self.local_rank))
        self.logger.info('================ 开始初始化模型 ===========')
        # ------------ 默认是多机多卡，然后这个local rank表示的是某台机器上卡的相对索引 ----------
        self.machine_index = self.global_rank // self.policy_config['device_number_per_machine']
        self.parameter_sharing = self.policy_config['parameter_sharing']
        self.homogeneous_agent = self.policy_config['homogeneous_agent']
        self.training_type = self.policy_config['training_type']
        self.load_data_from_model_pool = self.config_dict.get('load_data_from_model_pool', False)
        self.priority_replay_buffer = self.policy_config.get('priority_replay_buffer', False)
        self.logger.info("============== global rank {}开始创建模型 ==========".format(self.global_rank))
        # --------------------- 开始创建网络,定义两个optimizer，一个优化actor，一个优化critic ------------------
        self.construct_model()
        self.total_training_steps = 0
        self.training_steps_per_mins = 0
        if self.global_rank == 0:
            # ------------ 多卡场景下的训练，只有第0张卡的learner才会存模型下来 ----------
            # ---------------- 发送新的模型的路径给configserver，然后configserver会将模型信息下发给所有的worker -----------
            self.model_sender = self.context.socket(zmq.PUSH)
            self.model_sender.connect("tcp://{}:{}".format(self.config_dict['config_server_address'], self.config_dict['config_server_model_from_learner']))
            # ---------------- 每次模型发送出去后，下一次发送的时间间隔 -------------------------
            self.model_update_interval = self.config_dict['model_update_intervel']
            self.next_model_transmit_time = time.time()
            self.model_pool_path = self.policy_config['model_pool_path']
            # ---------------- 定义最新模型发布到哪一个网站上 --------------------------
            self.model_url = self.policy_config['model_url']
            self.logger.info("--------------- 讲初始化模型发送给configserver --------------")
            self._send_model(self.total_training_steps)
        # ------------- 由于一个plasma对应多个data_server，因此需要循环的从这个plasma id列表中选择 -------------
        self.plasma_id_queue = queue.Queue(maxsize=self.policy_config['server_number_per_device'])
        for i in range(self.policy_config['server_number_per_device']):
            plasma_id = plasma.ObjectID(generate_plasma_id(self.machine_index, self.local_rank, i))
            self.plasma_id_queue.put(plasma_id)
        if self.priority_replay_buffer:
            self.create_plasma_server_for_prioritized_replay_buffer()
        # ------------- 连接plasma 服务，这个地方需要提前启动这个plasma服务，然后让client进行连接 -------------
        self.plasma_client = plasma.connect(self.policy_config['plasma_server_location'], 2)
        # ------------- 这个列表是等待数据的时间 -------------
        self.wait_data_times = []
        # ------------- 定义一个变量，观察在一分钟之内参数更新了的次数 -------------
        self.training_steps = 0
        # ------------- 定义变量，每一次训练需要的时间 -------------
        self.training_time_list = []
        # ------------- 定义一下模型热更新的时间 -------------
        self.warm_up_time = time.time() + self.config_dict['warmup_time']
        # ------------- 定义一个变量，这个是每过一分钟，就朝着log server发送数据的 -------------
        self.next_check_time = time.time()
        self.next_send_log_time = time.time()


    def create_plasma_server_for_prioritized_replay_buffer(self):
        # ------------- 如果使用了优先级replay buffer，则需要建立一个额外的plasma id用来放weight数据 ------
        self.plasma_id_for_weight_queue = queue.Queue(maxsize=self.policy_config['server_number_per_device'])
        for i in range(self.policy_config['server_number_per_device']):
            plasma_id = plasma.ObjectID(generate_plasma_id_for_PEB(self.machine_index, self.local_rank, i))
            self.plasma_id_for_weight_queue.put(plasma_id)
    
    def construct_target_model(self):
        # --------- 这个只有那种需要创建target网络的算法，比如说MADDPG，DQN等，才需要进入 ----------
        self.target_model = dict()
        for model_type in self.policy_config['agent'].keys():
            self.target_model[model_type] = dict()
            for agent_name in self.policy_config['agent'][model_type].keys():
                model_config  = deepcopy(self.policy_config['agent'][model_type][agent_name])
                self.target_model[model_type][agent_name] = create_model(model_config)
                # ------------- target model参数copy active model -------------
                self.target_model[model_type][agent_name].load_state_dict(self.model[model_type][agent_name].state_dict())
            
    def construct_model(self):
        self.optimizer = {}
        self.model = {}
        self.scheduler = {}
        # ------- 这个字典只用来保存模型路径，只有在测试的时候会用到 -------------------
        self.model_path = {}
        for model_type in self.policy_config['agent'].keys():
            self.optimizer[model_type] = dict()
            self.model[model_type] = dict()
            self.scheduler[model_type] = dict()
            self.model_path[model_type] = dict()
            for agent_name in self.policy_config['agent'][model_type].keys():
                model_config = deepcopy(self.policy_config['agent'][model_type][agent_name])
                self.model[model_type][agent_name] = create_model(model_config)
                # ------- 如果在原有的基础上进行RL的训练，就需要载入预训练模型了 ---------
                if self.load_data_from_model_pool and model_type == 'policy':
                    self.logger.info('----------- 载入预训练模型，模型的保存路径为:{} ----------'.format(self.policy_config['agent'][model_type][agent_name]['model_path']))
                    deserialize_model(self.model[model_type][agent_name], self.policy_config['agent'][model_type][agent_name]['model_path'])
                self.optimizer[model_type][agent_name] = torch.optim.Adam(self.model[model_type][agent_name].parameters(), lr=float(self.policy_config['agent'][model_type][agent_name]['learning_rate']))
                self.scheduler[model_type][agent_name] = CosineAnnealingWarmRestarts(self.optimizer[model_type][agent_name], self.policy_config['T_zero'])
        if self.policy_config.get('using_target_network', False):
            self.construct_target_model()

        # ----------- 训练模式, 使用DDP进行包装  --------------
        dist.init_process_group(init_method=self.policy_config["ddp_root_address"], backend="nccl",rank=self.global_rank, world_size=self.world_size)
        # ----- 把模型放入到设备上 ---------
        for model_type in self.model: 
            for sub_model in self.model[model_type]:
                self.model[model_type][sub_model].to(self.local_rank).train()
                if self.policy_config.get('using_target_network', False):
                    self.target_model[model_type][sub_model].to(self.local_rank)
                self.model[model_type][sub_model] = DDP(self.model[model_type][sub_model], device_ids=[self.local_rank])
        torch.manual_seed(194864146)
        self.logger.info('----------- 完成模型的创建 ---------------')
        # ----------- 调用更新算法 ---------------
        algo_cls = get_algorithm_cls(self.policy_config['algorithm'])
        if self.policy_config.get('using_target_network', False):
            self.algo = algo_cls(self.model, self.target_model ,self.optimizer, self.scheduler, self.policy_config['training_parameters'])
        else:
            self.algo = algo_cls(self.model, self.optimizer, self.scheduler, self.policy_config['training_parameters'])


    def _send_model(self, training_steps):
        if time.time() > self.next_model_transmit_time:
            '''
                url_path的样子:
                    url_path['policy_url'] : string
                如果说有critic网络在，则还有一个key:
                    url_path['critic_url']: string
            '''
            url_path = serialize_model(self.model_pool_path, self.policy_config['model_url'], self.model, self.config_dict['model_cache_size'], self.logger)
            model_infomation = {'policy_name': self.policy_name, 'url': url_path}
            self.next_model_transmit_time += self.model_update_interval
            self.logger.info("-------------------- 发送模型到configserver，发送的信息为: {}，当前的模型更新次数为: {}".format(model_infomation, training_steps+1))
            self.model_sender.send(pickle.dumps(model_infomation))
            self.logger.info('------- 完成模型的发送 -------')

    def _training(self, training_batch):
        if time.time()>self.warm_up_time:
            if self.priority_replay_buffer:
                torch_training_batch = convert_data_format_to_torch_trainig(training_batch[0],self.local_rank)
                torch_weights_batch = convert_data_format_to_torch_trainig(training_batch[1], self.local_rank)
                info, weight = self.algo.step(torch_training_batch, torch_weights_batch)
                # --------- 把这个weight塞到plasma buffer里面 --------
                weight_plasma_id = self.plasma_id_for_weight_queue.get()
                assert not self.plasma_client.contains(weight_plasma_id), '---- 确保learner朝plasma服务中写入权重字典时不存在对应的plasma ID ---'
                self.plasma_client.put(weight, weight_plasma_id, memcopy_threads=12)
                # --------- 更新buffer中的权重，再把这个plasma id塞回去 --------
                self.plasma_id_for_weight_queue.put(weight_plasma_id)
            else:
                torch_training_batch = convert_data_format_to_torch_trainig(training_batch,self.local_rank)
                info = self.algo.step(torch_training_batch)
            self.logger.info("----------- 完成一次参数更新，更新的信息为 {} -------------".format(info))
            self.recursive_send(info, None, self.policy_name)
        else:
            self.logger.info("----------- 模型处于预热阶段，不更新参数 ----------")
        if self.total_training_steps % 20 == 0:
            self._test_model()

    def _test_model(self):
        import gym
        import pybullet_envs
        import torch
        env = gym.make(self.config_dict['env']['env_name'])
        env.seed(0)
        reward_list = []
        step = 0
        current_state = env.reset()
        while True:
            with torch.no_grad():
                action = self.model['policy']['default'](torch.FloatTensor(current_state).unsqueeze(0).to(0)).cpu().squeeze().numpy()
            next_state, instant_reward, done, _  = env.step(action)
            reward_list.append(instant_reward)
            current_state = next_state
            step += 1
            if done:
                break
        self.send_log({"result/test_reward_{}".format(self.policy_name): sum(reward_list)})


    def training_and_publish_model(self):  
        start_time = time.time()
        selected_plasma_id = self.plasma_id_queue.get()
        batch_data = self.plasma_client.get(selected_plasma_id)
        if self.global_rank == 0:
            self.wait_data_times.append(time.time()-start_time)
        self._training(batch_data)
        self.training_steps_per_mins += 1
        self.total_training_steps += 1
        self._send_model(self.total_training_steps)
        # ------------ 将训练数据从plasma从移除 ------------
        self.plasma_client.delete([selected_plasma_id])
        self.plasma_id_queue.put(selected_plasma_id)

        if self.global_rank == 0:
            self.logger.info("----------------- 完成第{}次训练 --------------".format(self.total_training_steps))
            end_time = time.time()
            self.training_time_list.append(end_time-start_time)
            if end_time > self.next_send_log_time:
                # ---------- 将每分钟更新模型的次数，每次更新模型的时间发送回去 -------------
                self.send_log({"learner_server/model_update_times_per_min/{}".format(self.policy_name): self.training_steps_per_mins})
                self.send_log({"learner_server/average_model_update_time_consuming_per_mins/{}".format(self.policy_name): sum(self.training_time_list)/self.training_steps_per_mins})
                self.send_log({"learner_server/time_of_wating_data_per_mins/{}".format(self.policy_name): sum(self.wait_data_times)/self.training_steps_per_mins})
                self.next_send_log_time += 60
                self.training_steps_per_mins = 0
                self.training_time_list = []
                self.wait_data_times = []
            if self.total_training_steps % self.policy_config['model_save_interval'] == 0:
                self._save_model()

    def _save_model(self):
        timestamp = str(time.time())
        for model_type in self.policy_config['agent'].keys():
            for agent_name in self.policy_config['agent'][model_type].keys():
                model_save_path = self.policy_config['saved_model_path'] + '/' + model_type + '_' + agent_name + '_'+ timestamp
                torch.save(self.model[model_type][agent_name].state_dict(), model_save_path)

    def run(self):
        self.logger.info("------------------ learner: {} 开始运行 ----------------".format(self.global_rank))
        while True:
            self.training_and_publish_model()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--rank', default= 0, type=int, help="rank of current process")
    parser.add_argument('--world_size', default=1, type=int, help='total gpu card')
    parser.add_argument('--init_method', default='tcp://120.0.0.1:23456')
    parser.add_argument('--config_path', type=str, default='Env/D4PG_config.yaml', help='yaml format config')
    args = parser.parse_args()
    # abs_path = '/'.join(os.path.abspath(__file__).splits('/')[:-2])
    # concatenate_path = abs_path + '/' + args.config_path
    # args.config_path = concatenate_path
    learner_server_obj = learner_server(args)
    learner_server_obj.run()


