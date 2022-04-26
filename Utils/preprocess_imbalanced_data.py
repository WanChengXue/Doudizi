# ------------- 这个函数用来处理采样端传来的数据 -----------
from multiprocessing import Pool
import pickle
import os
import sys
from tqdm import tqdm
import numpy as np
import lz4.frame as frame

current_path = os.path.abspath(__file__)
root_path = '/'.join(current_path.split('/')[:-2])
sys.path.append(root_path)

from Utils.utils import create_folder
from Utils.config import load_yaml
# -------------- 读取两个文件夹的数据 -------------



def calculate_single_file_weight(freq_cycle, freq_splits, source_data_file, saved_data_file, tls_list):
    # ------------ 传入的四个变量，分别表示两个频率字典，源文件路径，保存文件的路径 ---------
    load_f = open(source_data_file, 'rb')
    pickle_compressed_data = pickle.load(load_f)
    data = pickle.loads(frame.decompress(pickle_compressed_data))
    weight_list = []
    for data_dict in data:
        action = data_dict['action']
        weight_dict = dict()
        weight_dict['splits'] = dict()
        weight_dict['cycle'] = dict()
        for tls_id in tls_list:
            tls_action_weight = []
            for action_index, action_value in enumerate(action['splits'][tls_id]):
                # ------------ 保留两位小数，然后取值 ---------
                mapping_key = '%.2f' % (action_value)
                mapping_key = '0.99' if mapping_key == '1.00' else mapping_key
                action_weight = freq_splits[tls_id]['green_phase_{}'.format(action_index)][mapping_key]
                tls_action_weight.append(action_weight)
            weight_dict['splits'][tls_id] = tls_action_weight
            cycle_mapping_key =  '%.2f' % (action['cycle'][tls_id])
            cycle_mapping_key =  '0.99' if cycle_mapping_key == '1.00' else cycle_mapping_key
            weight_dict['cycle'][tls_id] = freq_cycle[tls_id][cycle_mapping_key]
        weight_list.append(weight_dict)
    # --------------- 把这个weight_list保存到本地 --------------
    open_file = open(saved_data_file, 'wb')
    pickle.dump(weight_list, open_file)
    open_file.close()
    
class saved_data_weight:
    def __init__(self, config_path):
        self.config_dict = load_yaml(config_path)
        self.source_data_folder = self.config_dict['data_saved_folder']
        self.policy_name = self.config_dict['policy_name']
        # ---------- 定义好这有多少台机器，每一个机器有多少张卡，每一张卡对应多少个数据服务 ----------
        self.machine_list = self.config_dict['policy_config']['machine_list']
        self.server_number_per_device = self.config_dict['policy_config']['server_number_per_device']
        self.device_number_per_machine = self.config_dict['policy_config']['device_number_per_machine']
        self.tls_list = self.config_dict['env']['tls_list']
        self._read_freq_dict()
        self._read_data_file_list_and_create_folder_for_weight_data()

    def _read_freq_dict(self):
        # ---------- 这个位置将所有动作的每一个点的频率读取出来 ----------
        freq_saved_folder = 'Exp/Result/Hist_figure/{}'.format(self.policy_name)
        freq_cycle_file = open(freq_saved_folder+'/freq_cycle_dict.pickle', 'rb')
        freq_splits_file = open(freq_saved_folder+'/freq_splits_dict.pickle', 'rb')
        self.freq_cycle_data = pickle.load(freq_cycle_file)
        self.freq_splits_data = pickle.load(freq_splits_file)

    def _read_data_file_list_and_create_folder_for_weight_data(self):
        # ---------- 这个函数用来提取文件保存的文件字典和创建一个文件夹用来存放权重数据 ----------
        self.saved_data_path_dict = dict()
        self.saved_weight_data_path_dict = dict()
        data_saved_path = self.config_dict['data_saved_folder']
        # ------ 创建文件夹用来保存数据样本的权重 --------
        data_weighted_saved_root_path = self.config_dict['weight_saved_folder']
        for machine_index in range(len(self.machine_list)):
            for device_index in range(self.device_number_per_machine):
                self.saved_data_path_dict['machine_{}_device_{}'.format(machine_index, device_index)] = dict()
                self.saved_weight_data_path_dict['machine_{}_device_{}'.format(machine_index, device_index)] = dict()
                for server_index in range(self.server_number_per_device):
                    source_data_folder = '{}/{}/{}_{}/{}'.format(data_saved_path, self.policy_name, machine_index, device_index, server_index)
                    weight_saved_data_folder = '{}/{}/{}_{}/{}'.format(data_weighted_saved_root_path, self.policy_name, machine_index, device_index, server_index)
                    self.saved_data_path_dict['machine_{}_device_{}'.format(machine_index, device_index)]['data_server_{}'.format(server_index)] = []
                    self.saved_weight_data_path_dict['machine_{}_device_{}'.format(machine_index, device_index)]['data_server_{}'.format(server_index)] = []
                    source_data_file_list = os.listdir(source_data_folder)
                    # self.saved_data_path_dict['machine_{}_device_{}'.format(machine_index, device_index)]['data_server_{}_file_number'.format(server_index)] = len(source_data_file_list)
                    create_folder(weight_saved_data_folder)
                    for file_name in source_data_file_list:
                        self.saved_data_path_dict['machine_{}_device_{}'.format(machine_index, device_index)]['data_server_{}'.format(server_index)].append(os.path.join(source_data_folder, file_name))
                        self.saved_weight_data_path_dict['machine_{}_device_{}'.format(machine_index, device_index)]['data_server_{}'.format(server_index)].append(os.path.join(weight_saved_data_folder, file_name))
                
    def _preprocess_and_save_data(self):
        pool = Pool(processes=10)
        for machine_device_key in self.saved_data_path_dict:
            for data_server_key in self.saved_data_path_dict[machine_device_key]:
                for file_index, source_data_name in tqdm(enumerate(self.saved_data_path_dict[machine_device_key][data_server_key])):
                    saved_weight_file_name = self.saved_weight_data_path_dict[machine_device_key][data_server_key][file_index]
                    pool.apply_async(calculate_single_file_weight, (self.freq_cycle_data, self.freq_splits_data, source_data_name, saved_weight_file_name,self.tls_list, ))
                    # calculate_single_file_weight(self.freq_cycle_data, self.freq_splits_data, source_data_name, saved_weight_file_name, self.tls_list)
        pool.close()
        pool.join()


    def run(self):
        self._preprocess_and_save_data()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default='Env/heterogeneous_network_config.yaml')
    args = parser.parse_args()
    worker = saved_data_weight(args.config_path)
    worker.run()