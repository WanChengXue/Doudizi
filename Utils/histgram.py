# ------------ 这个函数用来拟合saved_data中的动作分布 -----------
import os
import sys
import pickle
import lz4.frame as frame
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
current_path = os.path.abspath(__file__)
root_path = '/'.join(current_path.split('/')[:-2])
sys.path.append(root_path)

from Utils.utils import create_folder

data_saved_path = 'Data_saved'
policy_name = 'multi_point_heterogeneous_policy'
machine_number = 1
device_number_per_machine = 1
data_server_number_per_device = 1
data_path_list = []
tls_list = ['1', '2', '3', '4','5','6','7']
# ------ 创建文件夹用来保存可视化的直方图 --------
hist_saved_folder = 'Exp/Result/Hist_figure/{}/'.format(policy_name)
create_folder(hist_saved_folder)
for machine_index in range(machine_number):
    for device_index in range(device_number_per_machine):
        for server_index in range(data_server_number_per_device):
            data_folder = '{}/{}/{}_{}/{}'.format(data_saved_path, policy_name, machine_index, device_index, server_index)
            data_file_list = os.listdir(data_folder)
            for file_name in data_file_list:
                data_path_list.append(os.path.join(data_folder, file_name))
# --------- 定义两个字典用来保存各个动作区间区间出现的频率 --------
# --- splits_freq_dict['1'] = {'action_1':{'0.0': w, '0.01': w, ....}, 'action_2':{} , ..., }
splits_freq_dict = dict()
cycle_freq_dict = dict()
# ---------------------------------------------------------
splits_dict = dict()
cycle_dict = dict()
for tls_id in tls_list:
    splits_dict[tls_id] = []
    cycle_dict[tls_id] = []
for file_path in tqdm(data_path_list):
    load_f = open(file_path, 'rb')
    pickle_compressed_data = pickle.load(load_f)
    data = pickle.loads(frame.decompress(pickle_compressed_data))
    if len(data) == 0:
        # -------- 删除这个文件 ---------
        os.remove(file_path)
    else:
        for data_dict in data:
            action = data_dict['action']
            for tls_id in tls_list:
                splits_dict[tls_id].append(action['splits'][tls_id])
                cycle_dict[tls_id].append(action['cycle'][tls_id])

# ------------ 把列表变成numpy类型的数据 ---------------
for tls_id in tls_list:
    splits_dict[tls_id] = np.array(splits_dict[tls_id])
    # --------- batch_size * 4 ----------
    cycle_dict[tls_id] = np.array(cycle_dict[tls_id])
    # ---------- batch_size * 1
# --------------- 画图 ---------------
for tls_id in tls_list:
    # --------- 遍历动作 --------
    splits_freq_dict[tls_id] = dict()
    for action_index in range(4):
        tls_splits_action = splits_dict[tls_id][:,action_index]
        # ------------ 绘制直方图 --------------
        plt.figure()
        n, bins, patches = plt.hist(x=tls_splits_action, bins=100, range=(0,1), density=True, color='#0504aa', alpha=0.7, rwidth=0.85)
        plt.grid(axis='y', alpha=0.75)
        plt.xlabel('Value')
        plt.ylabel('PDF')
        plt.title("The histgram of green phase {} of tls id {}".format(str(action_index), tls_id))
        saved_fig = os.path.join(hist_saved_folder, tls_id+'_splits_action_' + str(action_index) + '.png')
        plt.savefig(saved_fig)
        plt.close()
        # ---------- 这个位置开始记录n,bins, n表示的是每一个beam的权重，bins是一个长度为101的array，因为bins设置的是100 ------
        splits_freq_dict[tls_id]['green_phase_{}'.format(action_index)] = dict()
        for bins_index in range(100):
            assert - np.log(n[bins_index]/100+1e-6) > 0
            splits_freq_dict[tls_id]['green_phase_{}'.format(action_index)]['%.2f'%(bins_index/100)] = - np.log(n[bins_index]/100+1e-6)
    # -------- 给cycle绘制直方图 ---------
    tls_cycle = cycle_dict[tls_id]
    plt.figure()
    n, bins, patches = plt.hist(x=tls_cycle, bins=100, density=True, color='#0504aa', alpha=0.7, rwidth=0.85)
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('Value')
    plt.ylabel('PDF')
    plt.title("The histgram of cycle of tls {}".format(tls_id))
    saved_fig = os.path.join(hist_saved_folder, tls_id+'_cycle.png')
    plt.savefig(saved_fig)
    plt.close()

    cycle_freq_dict[tls_id] = dict()
    for bins_index in range(100):
        cycle_freq_dict[tls_id]['%.2f'%(bins_index/100)] = -np.log(n[bins_index]/100 + 1e-6)
# -------------- 保存，freq_splits_dict, freq_cycle_dict ----------
freq_splits_dict_path = os.path.join(hist_saved_folder, 'freq_splits_dict.pickle')
freq_cycle_dict_path = os.path.join(hist_saved_folder, 'freq_cycle_dict.pickle')
# ------------ 保存splits dict ----------------
open_file = open(freq_splits_dict_path, 'wb')
pickle.dump(splits_freq_dict, open_file)
open_file.close()
# ------------ 保存cycle dict -----------------
open_file = open(freq_cycle_dict_path, 'wb')
pickle.dump(cycle_freq_dict, open_file)
open_file.close()
