import pickle
import os
import zmq
import time
import pathlib
import requests
# 由于这个类只有在training的时候才会实现,因此不需要考虑

class fetcher:
    def __init__(self, context, config_dict, statistic, process_uid, logger):
        self.context = context
        self.config_dict = config_dict
        self.policy_config = self.config_dict['policy_config']
        self.policy_name = self.config_dict['policy_name']
        self.config_server_address = self.config_dict['config_server_address']
        self.statistic = statistic
        self.policy_type = self.policy_config['policy_type']
        self.process_uid = process_uid
        self.logger = logger
        
        # ------------ 构建模型请求套接字 ---------------------
        self.latest_model_requester = self.context.socket(zmq.REQ)
        self.latest_model_requester.connect("tcp://{}:{}".format(self.config_dict['config_server_address'], self.config_dict['config_server_model_to_worker']))
        #------------------- 定义变量,模型下一次用来更新的时间 -------------------
        self.next_model_update_time = 0
        # -------------- 当前模型的timestamp，url ------------------
        self.current_model_time_stamp = time.time()
        self.current_model_url = None
        # -------------- 定义获取模型类型，可以获取过去时刻的模型, value就两种，latest，history -------
        self.policy_type = self.policy_config['policy_type']
        # ----------- 这个变量表示获取最新模型的时间间隔 ----------------
        self.sampler_model_interval = self.config_dict['sampler_model_update_interval']
        # ---------- 保存到worker文件夹下面 ---------------
        self.worker_folder_path = pathlib.Path(os.path.dirname(pathlib.Path(__file__).resolve())) 
        self.construct_latest_model_path()
        #------------------- 定义模型每次发送获取最新模型的请求间隔 -------------------
        self.logger.info("======================= 构建fetcher成功, 创建最新模型请求套接字 ===============")


    def construct_latest_model_path(self):
        self.model_path = dict()
        for model_type in self.policy_config['agent'].keys():
            self.model_path[model_type] = dict()
            for model_name in self.policy_config['agent'][model_type].keys():
                self.model_path[model_type][model_name] = self.worker_folder_path/'Download_model'/("{}.model".format((self.process_uid +'_{}_{}_'+self.policy_name).format(model_type, model_name)))
        self.logger.info("---------------- policy fetcher将最新的模型存放的地方为: {} ----------------".format(self.model_path))

    def _remove_exist_model(self):
        for model_type in self.policy_config['agent'].keys():
            for model_name in  self.policy_config['agent'][model_type].keys():
                if os.path.exists(self.model_path[model_type][model_name]):
                    os.remove(self.model_path[model_type][model_name])

    def _get_model(self):
        # ----------- 这个函数是用来获取最新的模型 ---------------
        self.latest_model_requester.send(pickle.dumps({'policy_name':self.policy_name, 'type': self.policy_type}))
        start_time = time.time()
        self.logger.info('------------ 等待configserver发送回来的信息 -----------')
        raw_model_info = self.latest_model_requester.recv()
        self.statistic.append("sampler/model_requester_time/{}".format(self.policy_name), time.time()-start_time)
        model_info = pickle.loads(raw_model_info)
        self.logger.info('-------------- 收到configserver发送回来的信息 {}------------'.format(model_info))
        if model_info['time_stamp'] == self.current_model_time_stamp:
            # ---------- 这个表示接收回来模型的时间戳没有发生变化，判断为同一个模型
            return None
        else:
            self.logger.info('----------- 开始从configserver下载模型 ------------')
            self._download_model(model_info)
            self.logger.info('------------- 完成模型下载 ----------------')
            return model_info

    def _download_model(self, model_info):
        # --------- 这个函数表示根据model_info这个dict，从指定的路径下载模型 -----------
        self._remove_exist_model()
        for model_type in model_info['url'].keys():
            for model_name in model_info['url'][model_type].keys():
                model_url = model_info['url'][model_type][model_name]
                saved_path = self.model_path[model_type][model_name]
                # ------------ 为了避免请求http server过于频繁 ----------
                while True:
                    try:
                        res = requests.get(model_url)
                        with open(saved_path, 'wb') as f:
                            f.write(res.content)
                        break
                    except:
                        self.logger.info("---------- Connection refused by the server.. --------")
                        time.sleep(2)
                        self.logger.info("------- Was a nice sleep, now let me continue... -----")

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
                self.statistic.append("sampler/model_update_interval/{}".format(self.policy_name), model_info['time_stamp']-self.current_model_time_stamp)
                self.current_model_time_stamp = model_info['time_stamp']
            return model_info


    