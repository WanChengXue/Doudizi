import argparse
from multiprocessing import Process

import pickle
import zmq
import time
import pathlib
import os
import sys

current_path = os.path.abspath(__file__)
root_path = "/".join(current_path.split("/")[:-2])
sys.path.append(root_path)

from Learner.base_server import base_server
from Utils.utils import setup_logger
from Utils.zmq_utils import zmq_nonblocking_multipart_recv, zmq_nonblocking_recv

# Learner ----> ConfigServer <---- Worker
class config_server(base_server):
    def __init__(self, config_path):
        super(config_server, self).__init__(config_path)
        config_server_log_path = pathlib.Path(
            self.config_dict["log_dir"] + "/config_server_log"
        )
        self.logger = setup_logger("Config_server_log", config_server_log_path)
        # ------------ 接收来自于learner的模型数据 ----------------------------
        self.model_receiver = self.context.socket(zmq.PULL)
        self.model_receiver.set_hwm(1000000)
        self.model_receiver.bind(
            "tcp://{}:{}".format(
                self.config_dict["config_server_address"],
                self.config_dict["config_server_model_from_learner"],
            )
        )
        self.poller.register(self.model_receiver, zmq.POLLIN)
        # ------------ 这个model receiver是接收来自worker端的请求, 然后下发 --------------
        self.model_request = self.context.socket(zmq.REP)
        self.model_request.set_hwm(1000000)
        self.model_request.bind(
            "tcp://{}:{}".format(
                self.config_dict["config_server_address"],
                self.config_dict["config_server_model_to_worker"],
            )
        )
        self.poller.register(self.model_request, zmq.POLLIN)
        # ------ 最新的模型信息 -----------
        self.latest_model_information = dict()
        self.policy_name = self.config_dict["policy_name"]
        self.logger.info("---------------------- 构建ConfigServer成功 -----------")

        # ----------- 此处需要开一个子线程，运行http服务，让worker端能够通过requests从这个ip上面下载文件 --------------
        self.http_server_process()

    def http_server_process(self):
        # 传入文件路径，ip，端口
        def _help(folder_path, server_ip, server_port):
            import http.server
            import socketserver
            import os

            os.chdir(folder_path)
            # ----------- 切换路径到folder_path下面 ---------
            Handler = http.server.SimpleHTTPRequestHandler
            httpd = socketserver.TCPServer((server_ip, server_port), Handler)
            httpd.serve_forever()

        model_folder = self.config_dict["policy_config"]["model_pool_path"]
        server_ip = self.config_dict["config_server_address"]
        server_port = self.config_dict["config_server_http_port"]
        self.logger.info(
            "-------- 打开httpserver, server的ip和端口分别是: {}:{} ---------".format(
                server_ip, server_port
            )
        )
        p = Process(
            target=_help,
            args=(
                model_folder,
                server_ip,
                server_port,
            ),
        )
        p.start()

    def process_model_request(self, raw_data_list):
        # --------- 这个函数用来处理来自于worker的请求 ——-------
        for raw_data in raw_data_list:
            model_information = None
            request_information = pickle.loads(raw_data[-1])
            policy_name = request_information["policy_name"]
            if request_information["type"] == "latest":
                if self.latest_model_information:
                    assert policy_name == self.latest_model_information["policy_name"]
                    model_information = self.latest_model_information
                else:
                    self.logger.warn(
                        "------------ 接收到了来自于worker端的信息, 但是configserver没有接收到learner的模型 --------------"
                    )
            else:
                self.logger.warn("------------- 目前只支持使用最新的模型 ----------")
            # ---------------- 将数据返回给worker --------------
            if model_information is not None:
                raw_data[-1] = pickle.dumps(model_information)
                self.model_request.send_multipart(raw_data)

    def process_new_model(self, raw_data_list):
        # ---------- 这个函数是用来处理来自于learner的最新模型 ——----------
        for raw_data in raw_data_list:
            model_information = pickle.loads(raw_data)
            self.logger.info(
                "------------ 接收到了新模型，模型路径在：{} --------------".format(
                    model_information["url"]
                )
            )
            # ------- 更新latest model information ------------
            policy_name = model_information["policy_name"]
            assert policy_name == self.policy_name, "--- learner发送过来的策略信息有错误 ----"
            self.latest_model_information = {
                "url": model_information["url"],
                "policy_name": policy_name,
                "time_stamp": time.time(),
            }

    def run(self):
        while True:
            sockets = dict(self.poller.poll(timeout=100))
            for key, value in sockets.items():
                if key == self.model_receiver:
                    # ------------ 如果当前是接收来自于learner的新模型 ——-------
                    raw_data_list = zmq_nonblocking_recv(key)
                    self.process_new_model(raw_data_list)
                elif key == self.model_request:
                    # ------------ 如果当前是接收来自于worker端的请求，需要新模型 -------
                    raw_data_list = zmq_nonblocking_multipart_recv(key)
                    self.process_model_request(raw_data_list)

                else:
                    self.logger.warn(
                        "---------- 接收到了一个未知的套接字{}:{} -------------".format(key)
                    )
            # else:
            #     self.logger.info("----------- 没有接收到socket ----------")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="")
    args = parser.parse_args()
    server = config_server(args.config_path)
    server.run()
