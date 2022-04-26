import pickle
import zmq
from Utils.config import parse_config


class base_server:
    def __init__(self, config_path):
        config_dict = parse_config(config_path)
        self.config_dict = config_dict
        self.context = zmq.Context()
        self.context.setsockopt(zmq.MAX_SOCKETS, 10000)
        self.poller = zmq.Poller()
        self.log_sender = self.context.socket(zmq.PUSH)
        self.log_sender.connect("tcp://{}:{}".format(self.config_dict['log_server_address'], self.config_dict['log_server_port']))
        self.cached_log_list = []

    def send_log(self, log_dict, send_threshold=10):
        # ---------- 这个函数是朝着logserver发送日志，默认每十条日志发送一次 ----------
        self.cached_log_list.append(log_dict)
        if len(self.cached_log_list) >= send_threshold:
            p = pickle.dumps(self.cached_log_list)
            self.log_sender.send(p)
            self.cached_log_list = []

    def recursive_send(self, log_info, prefix_string, suffix_string=None):
        # ------------ 这个传入进来的数据是一个字典，需要以递归的形式全部展开加头加尾进行发送 --------------
        if isinstance(log_info, dict):
            for key, value in log_info.items():
                if prefix_string is not None:
                    new_prefix_string = "{}/{}".format(prefix_string, key)
                else:
                    new_prefix_string = key
                self.recursive_send(value, new_prefix_string, suffix_string)
        elif isinstance(log_info, (tuple,list)):
            for index, value in enumerate(log_info):
                if prefix_string is not None:
                    new_prefix_string = "{}_{}".format(prefix_string, index)
                else:
                    new_prefix_string = key
                self.recursive_send(value, new_prefix_string, suffix_string)
        else:
            key = "{}/{}".format(prefix_string, suffix_string) if suffix_string is not None else prefix_string
            self.send_log({key: log_info})


