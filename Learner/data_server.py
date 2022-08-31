import pathlib
import zmq
import pyarrow.plasma as plasma
import lz4.frame as frame
import time
import pickle
import random
import sys
import os

current_path = os.path.abspath(__file__)
root_path = "/".join(current_path.split("/")[:-2])
sys.path.append(root_path)

from Utils.utils import generate_plasma_id, generate_plasma_id_for_PEB
from Utils import data_utils
from Utils.data_utils import convert_list_to_dict
from Utils.utils import setup_logger, create_folder
from Learner.base_server import base_server


# ---------- 开多个服务，用来接收来自worker的数据 ------------
class data_receiver_server(base_server):
    """
    这个类是用来接收来自于worker端的发送数据,然后送给网络进行训练,区别device和device版本
        - device版本：如果说一个device卡对应八个数据server，意思是说，它的采样数据只能够从这八个server中取出来
        - device版本：device不进行区分，也就是说，所有的dataserver放在一起
        - server_id这个参数在device场景下就对应于数据服务的id，在device场景下就需要进行划分，看看是对应的第几台机器的第几张卡
            - server_id的计算公式为： 每张卡开的进程数 * 每台机器卡的数量 * 机器的数量
    """

    def __init__(self, args):
        # ---------- 传入三个参数，分别是配置文件，日志对象，这个服务的id序号 --------
        super(data_receiver_server, self).__init__(args.config_path)
        self.global_rank = args.rank
        self.world_size = args.world_size
        self.policy_config = self.config_dict["policy_config"]
        self.device_number_per_machine = self.policy_config["device_number_per_machine"]
        # -------- global_rank表示所有的卡排序第几张卡，local_rank表示在这台机器上面的对应第几张卡 -------
        self.local_rank = self.global_rank % self.device_number_per_machine
        self.data_server_local_rank = args.data_server_local_rank
        self.policy_name = self.config_dict["policy_name"]
        # ---------- 这个变量一旦打开，则表示所有的数据将从本地进行读取 -------------
        self.read_data_from_local_machine = self.policy_config.get(
            "read_data_from_local_machine", False
        )
        # ---------- 这个地方链接上dataserver，接受来自worker的数据 -------------
        self.start_data_saved = self.config_dict.get("data_save_start", False)
        self.priority_replay_buffer = self.policy_config.get(
            "priority_replay_buffer", False
        )
        # ----------- 这个变量表示的是这是不是多智能体合作场景，默认为False -------------
        self.multiagent_scenario = self.config_dict["env"].get(
            "multiagent_scenario", False
        )
        self.agent_name_list = self.config_dict["env"]["trained_agent_name_list"]
        self.trained_agent = self.agent_name_list[0]
        self._parse_server_id_and_connect_server()
        self._define_performance_parameters()
        if not self.read_data_from_local_machine:
            self._create_replay_buffer()

    def _create_replay_buffer(self):
        self.replay_buffer = dict()
        if self.multiagent_scenario:
            self.replay_buffer = getattr(
                data_utils, self.policy_config["replay_buffer_config"]["buffer_name"]
            )(self.policy_config["replay_buffer_config"])
        else:
            for agent_name in self.agent_name_list:
                self.replay_buffer[agent_name] = getattr(
                    data_utils,
                    self.policy_config["replay_buffer_config"]["buffer_name"],
                )(self.policy_config["replay_buffer_config"])
        self.data_receive_count = 0

    def _parse_server_id_and_connect_server(self):
        self.machine_index = self.global_rank // self.device_number_per_machine
        self.server_ip = self.policy_config["machine_list"][self.machine_index]
        self.server_port = (
            self.policy_config["start_data_server_port"]
            + self.local_rank * self.policy_config["server_number_per_device"]
            + self.data_server_local_rank
        )
        logger_Path = pathlib.Path(
            self.config_dict["log_dir"]
            + "/dataserver_log/{}_{}/{}".format(
                self.local_rank, self.policy_name, self.data_server_local_rank
            )
        )
        self.logger = setup_logger(
            "DataServer_log_{}_{}".format(self.local_rank, self.data_server_local_rank),
            logger_Path,
        )
        self.logger.info(
            "-------------- 该数据进程使用的机器id为: {}, 对应的设备的索引为: {}, 数据进程的索引为: {}".format(
                self.machine_index, self.local_rank, self.data_server_local_rank
            )
        )
        if not self.read_data_from_local_machine:
            # ------------- 这个部分是用来打开数据的接收服务，worker端的数据通过这个socket进行发送 --------
            self.receiver = self.context.socket(zmq.PULL)
            #  ------------ 这个地方设置这个receiver socket的最大换存量 ---------
            self.receiver.set_hwm(1000000)
            self.receiver.bind("tcp://{}:{}".format(self.server_ip, self.server_port))
            self.poller.register(self.receiver, zmq.POLLIN)
        else:
            # ------------- 如果要从本地读取数据来进行训练，先将所有的pickle文件加载 ------------
            self._load_data_from_memory()
        self.batch_size = self.policy_config["batch_size"]
        # ------------- 这个地方链接上共享内存服务 -------------------------
        # 生成两个plasma_id
        self.plasma_data_id_dict = {}
        for agent in self.agent_name_list:
            plasma_id = generate_plasma_id(
                self.machine_index, self.local_rank, self.data_server_local_rank, agent
            )
            self.plasma_data_id_dict[agent] = plasma.ObjectID(plasma_id)
        if self.priority_replay_buffer:
            weight_plamsa_id = generate_plasma_id_for_PEB(
                self.machine_index, self.local_rank, self.data_server_local_rank
            )
            self.weight_plasma_id = plasma.ObjectID(weight_plamsa_id)
        # -------------- 将这个服务链接到plasma共享内存服务，需要提前开启，给两次链接机会 ------------
        self.plasma_client = plasma.connect(
            "{}".format(self.policy_config["plasma_server_location"]), 2
        )

        self.logger.info(
            "------------- 创建dataserver成功, worker端链接的地址为:{} -------------".format(
                "tcp://{}:{}".format(self.server_ip, self.server_port)
            )
        )
        if self.start_data_saved:
            # ------- 只有要将数据保存到本地的开关打开的时候，才需要创建一个dataserver的保存路劲 ---------
            self.data_saved_path = (
                self.config_dict["data_saved_folder"]
                + "/"
                + "{}/{}_{}/{}".format(
                    self.policy_name,
                    self.machine_index,
                    self.local_rank,
                    self.data_server_local_rank,
                )
            )
            create_folder(self.data_saved_path)
            self.logger.info(
                "------------- dataserver接收来自于worker端的数据, 保存的位置为:{} -----------------".format(
                    self.data_saved_path
                )
            )

    def _load_data_from_memory(self):
        # ------------------ 定义数据保存的路径 -----------------
        self.data_folder = (
            self.config_dict["data_saved_folder"]
            + "/"
            + "{}/{}_{}/{}".format(
                self.policy_name,
                self.machine_index,
                self.local_rank,
                self.data_server_local_rank,
            )
        )
        # ------------------ 定义权重保存的路径 ------------------
        self.weight_folder = (
            self.config_dict["weight_saved_folder"]
            + "/"
            + "{}/{}_{}/{}".format(
                self.policy_name,
                self.machine_index,
                self.local_rank,
                self.data_server_local_rank,
            )
        )
        data_file_list = os.listdir(self.data_folder)
        self.data_path_list = []
        for file_name in data_file_list:
            self.data_path_list.append(file_name)
        self.sample_file_number = (
            self.policy_config["batch_size"]
            // self.policy_config["woker_transmit_interval"]
        )

    def _save_data(self, frame_data):
        # ------ 这个函数是将接收到的数据保存到本地 ----------
        # -- 构建保存路径 ----
        saved_path = self.data_saved_path + "/{}_pickle_data.pickle".format(
            self.data_receive_count
        )
        open_file = open(saved_path, "wb")
        pickle.dump(frame_data, open_file)
        open_file.close()
        self.data_receive_count += 1

    def _define_performance_parameters(self):
        # ----------- 这个函数用来定义一些性能参数 ----------
        self.next_sample_time = time.time()  # 这个表示下一次将数据从buffer转移到plasma server的时间门限值
        self.sampling_count = 0  # 这个值表示采样了多少次
        self.next_send_log_time = time.time()  # 这个表示下一次将日志数据发送给logserver的时间
        if not self.read_data_from_local_machine:
            # -------- 这个是使用socket接收来自于worker端耗费的时间,将数据解析出来放入到buffer里面耗费的时间 -------------
            self.socket_recv_time_list = []
            self.parse_recv_data_time_list = []
            # ---------- 需要统计一下一分钟内流入了多少数据，接收数据花了多久，解析数据花了多久 ---------
            self.recv_data_number = 0
        self.sample_data_time_list = []

    def _recv_data(self, socket):
        # ------------ 这个函数用来接受来自worker端的数据 -------------
        raw_data_list = []
        if self.receiver in socket:
            start_recv_data_time = time.time()
            while True:
                try:
                    data = self.receiver.recv(zmq.NOBLOCK)
                    raw_data_list.append(data)
                except zmq.ZMQError as e:
                    if type(e) != zmq.error.Again:
                        self.logger.warn("---------- 异常错误发生 -------".format(e))
                    break
            if len(raw_data_list) > 0:
                self.socket_recv_time_list.append(time.time() - start_recv_data_time)
            # ---------- 上面接收到的是一个worker打包了多个数据点构成的对象，现在需要将其解析出来 -----------
            current_sample_recv_number = 0
            start_process_data_time = time.time()
            for compressed_data in raw_data_list:
                if self.start_data_saved:
                    self._save_data(compressed_data)
                pickled_data = pickle.loads(frame.decompress(compressed_data))
                # ---------- 这个返回来的pickled_data, 如果是多智能体场景，则不需要给这个数据区分 ---------
                if self.multiagent_scenario:
                    self.replay_buffer.append_data(pickled_data)
                    current_sample_recv_number += len(pickled_data)
                else:
                    for key in pickled_data:
                        self.replay_buffer[key].append_data(pickled_data[key])
                        self.trained_agent = key
                    current_sample_recv_number += len(pickled_data[key])

            # ---------- 解析完此次接收，看看到底流入了多少的数据量 -----------
            self.recv_data_number += current_sample_recv_number
            if self.multiagent_scenario:
                self.logger.info(
                    "------------- 本次接收数据数目为：{},此时buffer中的数据为:{} ------------".format(
                        current_sample_recv_number, self.replay_buffer.buffer_size
                    )
                )
            else:
                for key in self.replay_buffer:
                    self.logger.info(
                        "------------- 本次接收数据数目为：{},此时buffer {} 中的数据为:{} ------------".format(
                            current_sample_recv_number,
                            key,
                            self.replay_buffer[key].buffer_size,
                        )
                    )
            if len(raw_data_list) > 0:
                parse_data_time = time.time() - start_process_data_time
                self.parse_recv_data_time_list.append(parse_data_time)
            del raw_data_list

    def _sample_data_from_replay_buffer(self):
        # ----------- 此处随机采样出一个batch的训练数据 ----------------
        if self.multiagent_scenario:
            if self.priority_replay_buffer:
                self.sample_index, sample_data_dict = self.replay_buffer.sample_data()
            else:
                sample_data_dict = self.replay_buffer.sample_data()
            return sample_data_dict
        else:
            sample_data_dict = dict()
            if self.priority_replay_buffer:
                self.sample_index = dict()
                weight_data_dict = dict()
                for key in self.replay_buffer:
                    (
                        agent_sample_data_dict,
                        agent_sample_index,
                        agent_weight_data,
                    ) = self.replay_buffer[key].sample_data()
                    self.sample_index[key] = agent_sample_index
                    sample_data_dict[key] = agent_sample_data_dict
                    weight_data_dict[key] = agent_weight_data
                return (sample_data_dict, weight_data_dict)
            else:
                sample_data_dict[self.trained_agent] = self.replay_buffer[self.trained_agent].sample_data()
            return sample_data_dict

    def sampling_data(self):
        if self.global_rank == 0:
            self.logger.info("============== 开始采样 ===============")
        start_time = time.time()
        # ----------- 数据转移到plasma client里面去 -------------------
        sample_data_dict = self._sample_data_from_replay_buffer()
        if self.global_rank == 0:
            if self.multiagent_scenario:
                self.logger.info(
                    "================= 采样时间为 {}, batch size为 {}, 目前buffer的数据为 {} =============".format(
                        time.time() - start_time,
                        self.batch_size,
                        self.replay_buffer.buffer_size,
                    )
                )
            else:
                for key in self.replay_buffer:
                    self.logger.info(
                        "================= 操作buffer {}, 采样时间为 {}, batch size为 {}, 目前buffer的数据为 {} =============".format(
                            key,
                            time.time() - start_time,
                            self.batch_size,
                            self.replay_buffer[key].buffer_size,
                        )
                    )
        start_convert_to_plasma_time = time.time()
        self.plasma_client.put(
            sample_data_dict, self.plasma_data_id_dict[self.trained_agent], memcopy_threads=12
        )
        self.logger.info(
            "=============== 将采样出来的数据转移到plasma中耗费的时间为: {} =================".format(
                time.time() - start_convert_to_plasma_time
            )
        )
        del sample_data_dict
        # ------------------------------------------------------------
        if self.priority_replay_buffer:
            # --------------- 如果说使用了优先级replay buffer，则需要获取更新之后的权重信息，获取learner塞进到plasma server的数据 ----
            weight_data = self.plasma_client.get(self.weight_plasma_id)
            # --------------- 如果是独立场景，则这个weight_data是一个字典，否则，这个weight_data就是一个向量 -----------
            if self.multiagent_scenario:
                self.replay_buffer.update_priorities(self.sample_index, weight_data)
            else:
                for key in weight_data:
                    self.replay_buffer[key].update_priorities(
                        self.sample_index[key], weight_data[key]
                    )
            # --------- 将plasma client中权重id的数据删除 --------
            self.plasma_client.delete([self.weight_plasma_id])
            self.logger.info(
                "================== 更新完成priority replay buffer中的数据样本优先级 ========="
            )

    def full_buffer(self):
        # --------- 如果buffer满了二十分之一就可以训练了 ---------
        if self.multiagent_scenario:
            return self.replay_buffer.full_buffer
        else:
            if self.replay_buffer[self.trained_agent].full_buffer:
                return True
            else:
                return False

    def run(self):
        # ----------- 函数一旦运行起来，就不断的监听套接字 ---------------
        self.logger.info(
            "-------------- 数据服务开始启动, 对应的plasma_id是:{} -------------".format(
                self.plasma_data_id_dict
            )
        )
        while True:
            sockets = dict(self.poller.poll(timeout=100))
            self._recv_data(sockets)
            # ------ 如果说这个这个plasma id不在plasma客户端里面，并且这个replaybuffer要是满的，还要当前时间超过了采样间隔才放进去，默认采样间隔为0 ---------
            if (
                self.full_buffer()
                and time.time() > self.next_sample_time
                and not self.plasma_client.contains(self.plasma_data_id_dict[self.trained_agent])
            ):
                current_time = time.time()
                self.next_sample_time = (
                    current_time + self.config_dict["data_server_sampling_interval"]
                )
                self.sampling_data()
                self.sample_data_time_list.append(time.time() - current_time)

            # ------------------ 朝着logserver发送日志，日志设置为每一分钟发送一次 --------------------
            if time.time() > self.next_send_log_time:
                self.next_send_log_time += 60
                # ---------- 这个地方将这个worker在一分钟内收到的数据量发送给logserver ----------------
                self.send_log(
                    {
                        "data_server/data_server_recv_data_numbers_per_min/{}".format(
                            self.policy_name
                        ): self.recv_data_number
                    }
                )
                # ---------- 这个地方将这一分钟内，耗费在socket上面的时间发送个logserver ----------------
                self.send_log(
                    {
                        "data_server/socket_recv_time_per_min/{}".format(
                            self.policy_name
                        ): sum(self.socket_recv_time_list)
                    }
                )
                # ----------- 这个地方将这一分钟内，解析数据时间，以及将数据转移到replaybuffer里面的时间发送给logserver -----------
                self.send_log(
                    {
                        "data_server/parse_recv_data_time_per_min/{}".format(
                            self.policy_name
                        ): sum(self.parse_recv_data_time_list)
                    }
                )
                self.send_log(
                    {
                        "sampler/transfer_data_from_buffer_to_plasma_server_per_min/{}".format(
                            self.policy_name
                        ): sum(
                            self.sample_data_time_list
                        )
                    }
                )
                # ---------- 重置这四个变量 -------------
                self.recv_data_number = 0
                self.socket_recv_time_list = []
                self.parse_recv_data_time_list = []
                self.sample_data_time_list = []


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", default=0, type=int, help="rank of current process")
    parser.add_argument(
        "--config_path",
        type=str,
        default="Config/Training/DQN_config.yaml",
        help="yaml format config",
    )
    parser.add_argument(
        "--data_server_local_rank", default=0, type=int, help="data_server_local_rank"
    )
    parser.add_argument("--world_size", default=1, type=int, help="world_size")
    args = parser.parse_args()
    # abs_path = '/'.join(os.path.abspath(__file__).splits('/')[:-2])
    # concatenate_path = abs_path + '/' + args.config_path
    # args.config_path = concatenate_path
    test_server = data_receiver_server(args)
    test_server.run()
