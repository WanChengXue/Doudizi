import pickle
import os
from numpy import isin
import zmq
import time
import pathlib
import requests

# 由于这个类只有在training的时候才会实现,因此不需要考虑


class fetcher:
    def __init__(self, context, config_dict, statistic, process_uid, logger):
        self.context = context
        self.config_dict = config_dict
        self.policy_config = self.config_dict["policy_config"]
        self.policy_name = self.config_dict["policy_name"]
        self.agent_name_list = self.config_dict["env"]["agent_name_list"]
        self.config_server_address = self.config_dict["config_server_address"]
        self.statistic = statistic
        self.policy_type = self.policy_config["policy_type"]
        self.process_uid = process_uid
        self.logger = logger
        self.trained_agent = None
        self.model_type = ["policy", "critic", "double_critic"]
        self.use_centralized_critic = self.config_dict["policy_config"][
            "use_centralized_critic"
        ]
        # ------------ 构建模型请求套接字 ---------------------
        self.latest_model_requester = self.context.socket(zmq.REQ)
        self.latest_model_requester.connect(
            "tcp://{}:{}".format(
                self.config_dict["config_server_address"],
                self.config_dict["config_server_model_to_worker"],
            )
        )
        # ------------------- 定义变量,模型下一次用来更新的时间 -------------------
        self.next_model_update_time = 0
        # -------------- 当前模型的timestamp，url ------------------
        self.current_model_time_stamp = time.time()
        self.current_model_url = None
        # -------------- 定义获取模型类型，可以获取过去时刻的模型, value就两种，latest，history -------
        self.policy_type = self.policy_config["policy_type"]
        # ----------- 这个变量表示获取最新模型的时间间隔 ----------------
        self.sampler_model_interval = self.config_dict["sampler_model_update_interval"]
        # ---------- 保存到worker文件夹下面 ---------------
        self.worker_folder_path = pathlib.Path(
            os.path.dirname(pathlib.Path(__file__).resolve())
        )
        self.construct_latest_model_path()
        # ------------------- 定义模型每次发送获取最新模型的请求间隔 -------------------
        self.logger.info(
            "======================= 构建fetcher成功, 创建最新模型请求套接字 ==============="
        )

    def construct_latest_model_path(self):
        self.model_path = dict()
        for agent_name in self.agent_name_list:
            self.model_path[agent_name] = dict()
            for model_type in self.policy_config["agent"][agent_name].keys():
                if model_type in self.model_type:
                    self.model_path[agent_name][model_type] = (
                        self.worker_folder_path
                        / "Download_model"
                        / (
                            "{}.model".format(
                                (
                                    self.process_uid + "_{}_{}_" + self.policy_name
                                ).format(agent_name, model_type)
                            )
                        )
                    )
        # ---------- 如果在多智能体的场景中，需要额外获取中心化的critic网路 ------
        if self.use_centralized_critic:
            self.model_path["centralized_critic"] = (
                self.worker_folder_path
                / "Download_model"
                / (
                    "{}.model".format(
                        self.process_uid + "_centralized_critic_" + self.policy_name
                    )
                )
            )
        self.logger.info(
            "---------------- policy fetcher将最新的模型存放的地方为: {} ----------------".format(
                self.model_path
            )
        )

    def _remove_exist_model(self):
        for agent_name in self.agent_name_list:
            for model_type in self.model_path[agent_name].keys():
                if os.path.exists(self.model_path[agent_name][model_type]):
                    os.remove(self.model_path[agent_name][model_type])
        if self.use_centralized_critic:
            os.remove(self.model_path["centralized_critic"])

    def _get_model(self):
        # ----------- 这个函数是用来获取最新的模型 ---------------
        self.latest_model_requester.send(
            pickle.dumps({"policy_name": self.policy_name, "type": self.policy_type})
        )
        start_time = time.time()
        self.logger.info("------------ 等待configserver发送回来的信息 -----------")
        raw_model_info = self.latest_model_requester.recv()
        self.statistic.append(
            "sampler/model_requester_time/{}".format(self.policy_name),
            time.time() - start_time,
        )
        model_info = pickle.loads(raw_model_info)
        self.logger.info(
            "-------------- 收到configserver发送回来的信息 {}------------".format(model_info)
        )
        if model_info["time_stamp"] == self.current_model_time_stamp:
            # ---------- 这个表示接收回来模型的时间戳没有发生变化，判断为同一个模型
            return None
        else:
            self.logger.info("----------- 开始从configserver下载模型 ------------")
            self.trained_agent = model_info["trained_agent"]
            self._download_model(model_info)
            self.logger.info("------------- 完成模型下载 ----------------")
            return model_info

    def _download_model(self, model_info):
        # --------- 这个函数表示根据model_info这个dict，从指定的路径下载模型 -----------
        self._remove_exist_model()
        for agent_name in model_info["url"].keys():
            if isinstance(model_info["url"][agent_name], dict):
                for model_type in model_info["url"][agent_name]:
                    model_url = model_info["url"][agent_name][model_type]
                    saved_path = self.model_path[agent_name][model_type]
                    # ------------ 为了避免请求http server过于频繁 ----------
                    self._download_single_mdel(model_url, saved_path)
            else:
                model_url = model_info["url"][agent_name]
                saved_path = self.model_path[agent_name]
                self._download_single_mdel(model_url, saved_path)

    def _download_single_mdel(self, model_url, saved_path):
        while True:
            try:
                res = requests.get(model_url)
                with open(saved_path, "wb") as f:
                    f.write(res.content)
                break
            except:
                self.logger.info(
                    "---------- Connection refused by the server.. --------"
                )
                time.sleep(2)
                self.logger.info(
                    "------- Was a nice sleep, now let me continue... -----"
                )

    def reset(self):
        model_info = self.step()
        return model_info

    def step(self):
        if time.time() < self.next_model_update_time:
            self.logger.info("-------------- 当前时间还没达到下一次模型的更新时间，暂时不更新模型，跳过 ----------")
            return None
        else:
            model_info = self._get_model()
            if model_info is not None:
                # ------------- 如果模型信息不是None，则表示从configserver那里拿到了新的信息 -------
                self.next_model_update_time += self.sampler_model_interval
                # ------------- 统计模型的更新时间 ----------------
                self.statistic.append(
                    "sampler/model_update_interval/{}".format(self.policy_name),
                    model_info["time_stamp"] - self.current_model_time_stamp,
                )
                self.current_model_time_stamp = model_info["time_stamp"]
                return self.model_path
            else:
                return None

    @property
    def get_trained_agent(self):
        return self.trained_agent
