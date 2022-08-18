import os 
import sys
import pickle
import time
import zmq
import argparse

from torch.utils.tensorboard import SummaryWriter
import numpy as np
import pathlib

current_path = os.path.abspath(__file__)
root_path = '/'.join(current_path.split('/')[:-2])
sys.path.append(root_path)
from Learner.base_server import base_server
from Utils.utils import setup_logger, create_folder

# ----- ->log_server，PULL日志下来 -------------

class summary_logger:
    def __init__(self, tensorboard_folder):
        self.summarywriter = SummaryWriter(tensorboard_folder)
        # ------- 这个是标签值对应的字典 ------
        self.tag_values_dict = {}
        # ------- 这个表示这个标签写了多少次进去
        self.tag_step_dict = {}
        # -------- 这个列表表示要超过多少个值才会写入到tensorboard上面 -----
        self.tag_output_threshold_dict = {}
        # --------- 定义tag的计算类型，是sum，min，max，average等等 ------
        self.tag_func_dict = {}
        # ------ 下面三个字典是设计到于时间有关的tag -----------
        self.tag_time_threshold = {}
        self.tag_time_data_timestamp = {}
        self.tag_time_last_print_time  = {}
        # ======== 分别表示多久统计一次，最新的打印时间等 ==========
        self.total_tag_type = ["time_mean", "time_sum", "mean", "sum", "max", "min"]

    def add_tag(self, tag, output_threshold, calculate_type, time_threshold=0):
        self.tag_values_dict[tag] = []
        self.tag_step_dict[tag] = 0
        self.tag_output_threshold_dict[tag] = output_threshold
        self.tag_func_dict[tag] = calculate_type
        if 'time' in calculate_type:
            # --------- 如果这个tag是与时间有关 --------
            self.tag_time_threshold[tag] = time_threshold
            self.tag_time_data_timestamp[tag] = []
            self.tag_time_last_print_time[tag] = 0

    def has_tag(self, tag):
        if tag in self.tag_values_dict:
            return True
        else:
            return False
        
    @property
    def get_tag_count(self, tag):
        return self.tag_step_dict[tag]

    def generate_time_data_output(self, tag):
        threshold = self.tag_time_threshold[tag]
        current_time = time.time()
        if current_time - self.tag_time_last_print_time[tag] > threshold:
            # -------------- 如果说当前的时间和上一次输出的时间大于门限值，就进行输出 ----------
            valid_value = []
            for index, timestamp in enumerate(self.tag_time_data_timestamp[tag]):
                if current_time - timestamp < threshold:
                    valid_value.append(self.tag_values_dict[index])
            if len(valid_value) >=1:
                if self.tag_func_dict[tag] == 'time_mean':
                    output_value = sum(valid_value) / len(valid_value)
                elif self.tag_func_dict[tag] == 'time_sum':
                    output_value = sum(valid_value)
                else:
                    raise NotImplementedError
            else:
                output_value = 0
        self.summarywriter.add_scalar(tag, output_value, self.tag_step_dict[tag])
        self.tag_step_dict[tag] += 1
        self.tag_values_dict[tag] = []
        self.tag_time_data_timestamp[tag] = []
        self.tag_time_last_print_time[tag] = current_time

    
    def add_summary(self, tag, value, timestamp=time.time()):
        self.tag_values_dict[tag].append(value)
        if 'time' in self.tag_func_dict[tag]:
            self.tag_time_data_timestamp[tag].append(timestamp)
            self.generate_time_data_output(tag)
        else:
            # -------- 非时间类型的tag -----------
            if len(self.tag_values_dict[tag]) >= self.tag_output_threshold_dict[tag]:
                if self.tag_func_dict[tag] == 'sum':
                    output_value = sum(self.tag_values_dict[tag])
                elif self.tag_func_dict[tag] == 'mean':
                    output_value = sum(self.tag_values_dict[tag]) / len(self.tag_values_dict[tag])
                elif self.tag_func_dict[tag] == 'max':
                    output_value = max(self.tag_values_dict[tag])
                elif self.tag_func_dict[tag] == 'min':
                    output_value = min(self.tag_values_dict[tag])
                else:
                    raise NotImplementedError
                self.summarywriter.add_scalar(tag, output_value, self.tag_step_dict[tag])
                self.tag_step_dict[tag] += 1
                self.tag_values_dict[tag] = []

class log_server(base_server):
    def __init__(self, config_path):
        super(log_server,self).__init__(config_path)
        self.policy_name = self.config_dict['policy_name']
        self.policy_config = self.config_dict['policy_config']
        self.total_data_server_per_machine = self.policy_config['device_number_per_machine'] * self.policy_config['server_number_per_device']
        self.log_receiver = self.context.socket(zmq.PULL)
        self.log_receiver.bind("tcp://{}:{}".format(self.config_dict['log_server_address'], self.config_dict['log_server_port']))
        self.poller.register(self.log_receiver, zmq.POLLIN)
        log_path = pathlib.Path(self.config_dict['log_dir']+'/log_server_log')
        self.logger = setup_logger('LogServer_log', log_path)
        self.active_worker = dict()
        self.next_calculate_worker_time = time.time()
        self.summary_logger = summary_logger(self.policy_config['tensorboard_folder'])
        self.logger.info('----------- 完成log server的构建，数据保存到tensorboard的路径为：{} ---------------'.format(self.policy_config['tensorboard_folder']))
        # ----- 打开tensorboard服务，将输出结果和错误信息丢掉，本地打开训练机器的tensorboard，直接网页上打开,tensorboard默认6006端口，ip:6006 ---------
        self.open_tensorboard_server()
        create_folder(self.config_dict['policy_config']['tensorboard_folder'], delete_origin=True)
        self.logger.info("================== 完成log server的构建，配置好了tensorboard的路径为 {}".format(self.config_dict['policy_config']['tensorboard_folder']))


    def open_tensorboard_server(self):
        tensorboard_folder = self.config_dict['policy_config']['tensorboard_folder']
        # ------- 构建绝对tensorboard路径 --------------
        function_path = os.path.abspath(__file__)
        # ------ 这个就是到了Pretrained_model这一层路径下面 ----- ~/Desktop/ICC/code_part
        root_path = '/'.join(function_path.split('/')[:-2])
        # ------------ tensorboard_folder的路径是./logs/Tensorboard，需要把./去掉
        abs_path = root_path + '/' + tensorboard_folder
        server_ip = self.config_dict['log_server_address']
        tensorboard_command = "nohup python -m tensorboard.main --logdir={} --host={} > /dev/null 2>&1 &".format(abs_path, server_ip)
        os.system(tensorboard_command)
        self.logger.info("----------------- 创建好了tensorboard日志文件，命令是{} -----------------".format(tensorboard_command))


    def summary_definition(self):
        # --------------- 预先添加一些tag进去 --------------
        # =============== 效果类型的指标，表示采样完毕后，整条轨迹的累计奖励值 ============
        self.summary_logger.add_tag('result/accumulated_reward/{}'.format(self.policy_name), 1, 'mean')
        # =============== 采样端的指标，请求模型的时间，更新模型的时间，从configserver下载模型的时间，完整的采样一条轨迹需要的时间 ===============
        self.summary_logger.add_tag('sampler/episode_time/{}'.format(self.policy_name), 10, 'mean')
        self.summary_logger.add_tag('sampler/model_request_time/{}'.format(self.policy_name), 10, 'mean')
        self.summary_logger.add_tag('sampler/model_update_interval/{}'.format(self.policy_name), 10, 'mean')
        self.summary_logger.add_tag('sampler/http_model_download_time/{}'.format(self.policy_name), 10, 'mean')
        # =============== 定义dataserver的相关参数， 每分钟接收到的数据量，每分钟解析数据的时间，每分钟解析套接字的时间，从trainingset转移到plasma client的时间。以及采样worker的数目 =============
        self.summary_logger.add_tag('data_server/dataserver_recv_instance_per_min/{}'.format(self.policy_name), self.total_data_server_per_machine, 'sum')
        self.summary_logger.add_tag('data_server/dataserver_parse_time_per_min/{}'.format(self.policy_name), 1, 'sum')
        self.summary_logger.add_tag('data_server/dataserver_socket_time_per_min/{}'.format(self.policy_name), 1, 'mean')
        self.summary_logger.add_tag('data_server/dataserver_sampling_time_per_min/{}'.format(self.policy_name), 1, 'mean')
        self.summary_logger.add_tag('Worker/active_worker', 1, 'sum')
        self.summary_logger.add_tag('Worker/sample_step_per_episode/{}'.format(self.policy_name), 10, 'mean')
        # =============== 定义策略相关的指标，包括loss，等等 =========================
        self.summary_logger.add_tag('model/loss/{}'.format(self.policy_name), 10, 'mean')
        self.summary_logger.add_tag('result/sum_instant_reward/{}'.format(self.policy_name), 10, 'mean')


    def log_parse(self, data):
        # ------------- 这个函数用来解析log_server接收到的日志 -----------
        for field_key, value in data.items():
            if field_key == 'worker_id':
                self.active_worker[value] = 1
            else:
                if not self.summary_logger.has_tag(field_key):
                    self.summary_logger.add_tag(field_key, 1, 'mean')
                try:
                    self.summary_logger.add_summary(field_key, value)
                except:
                    self.logger.info("------------- 添加summary报错，键值对为{}:{}---------".format(field_key, value))
                    
    def init_string(self):
        output_string  = r"""
                                _ooOoo_
                                o8888888o
                                88" . "88
                                (| -_- |)
                                O\  =  /O
                            ____/`---'\____
                            .'  \\|     |//  `.
                        /  \\|||  :  |||//  \
                        /  _||||| -:- |||||-  \
                        |   | \\\  -  /// |   |
                        | \_|  ''\---/''  |   |
                        \  .-\__  `-`  ___/-. /
                        ___`. .'  /--.--\  `. . __
                    ."" '<  `.___\_<|>_/___.'  >'"".
                    | | :  `- \`.;`\ _ /`;.`/ - ` : | |
                    \  \ `-.   \_ __\ /__ _/   .-` /  /
            ======`-.____`-.___\_____/___.-`____.-'======
                                `=---='
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                        佛祖保佑        永无BUG
                佛曰:
                        写字楼里写字间，写字间里程序员；
                        程序人员写程序，又拿程序换酒钱。
                        酒醒只在网上坐，酒醉还来网下眠；
                        酒醉酒醒日复日，网上网下年复年。
                        但愿老死电脑间，不愿鞠躬老板前；
                        奔驰宝马贵者趣，公交自行程序员。
                        别人笑我忒疯癫，我笑自己命太贱；
                        不见满街漂亮妹，哪个归得程序员？
                """
        self.logger.info(output_string)

    def run(self):
        # --------- 主函数 --------------
        self.init_string()
        self.summary_definition()
        while True:
            if time.time() > self.next_calculate_worker_time:
                self.next_calculate_worker_time = time.time() + 60 * 3
                self.summary_logger.add_summary("Worker/active_worker", len(self.active_worker))
                self.active_worker = {}

            socks = dict(self.poller.poll(timeout=100))
            if self.log_receiver in socks and socks[self.log_receiver] == zmq.POLLIN:
                raw_data_list = []
                while True:
                    try:
                        data = self.log_receiver.recv(zmq.NOBLOCK)
                        raw_data_list.append(data)
                    except zmq.ZMQError as e:
                        if type(e) != zmq.error.Again:
                            self.logger.warn("recv zmq {}".format(e))
                        break
                for raw_data in raw_data_list:
                    data = pickle.loads(raw_data)
                    # self.logger.info("------------ 接收到的数据为: {} ------------------".format(data))
                    for log in data:
                        if "error_log" in log:
                            self.logger.error("client_error, %s"%(log["error_log"]))
                        elif "log_info" in log:
                            self.logger.info(data)
                        else:
                            self.log_parse(log)
                        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='')
    args = parser.parse_args()
    server = log_server(args.config_path)
    server.run()