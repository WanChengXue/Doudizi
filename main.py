import argparse
import os
import shutil
from Utils.config import parse_config


def start_learner_data_server(config_path, world_size, machine_index, device_number_per_machine, server_number_per_device, log_dir):
    learner_log_path = log_dir + '/learner_log_out'
    data_log_path = log_dir + '/data_server_log_out'
    for local_rank in range(device_number_per_machine):
        global_rank = machine_index * device_number_per_machine + local_rank
        for data_server_index in range(server_number_per_device):
            data_server_command = "nohup python -m Learner.data_server --rank {} --config_path {} --data_server_local_rank {} >> {} 2>&1 &".format(global_rank, config_path, data_server_index, data_log_path)
            os.system(data_server_command)
        # -------- 启动lerner server -----------
        learner_server_command = "nohup python -m Learner.learner_server --rank {} --world_size {} --config_path {} >> {}  2>&1 &".format(global_rank, world_size, config_path, learner_log_path)
        os.system(learner_server_command)

def main(args):
    config_dict = parse_config(args.config_path)
    log_dir = config_dict["log_dir"]
    # ------ 如果这个log文件夹存在，就删掉 --------
    # shutil.rmtree(log_dir)
    # ------- 打开plasma server，log server，config server ------
    if args.machine_index == 0:
        # ----- 打开plasma server -----------
        plasma_command = "nohup plasma_store -s {} -m 2000000000 &".format(config_dict['policy_config']['plasma_server_location'])
        os.system(plasma_command)
        # ----- 打开log server ----------
        log_command = "nohup python -m Learner.log_server --config {} > {} 2>&1 &".format(args.config_path, log_dir+'/log_output')
        os.system(log_command)
        # ------ 打开config server -----------
        config_command = "nohup python -m Learner.config_server --config {} > {} 2>&1 &".format(args.config_path, log_dir + '/config_log_output')
        os.system(config_command)
    # ------ 开启learner和data server ----------
    # -------- 计算有多少张卡 ---------
    world_size = len(config_dict['policy_config']['machine_list']) * config_dict['policy_config']['device_number_per_machine']
    start_learner_data_server(args.config_path, world_size, args.machine_index, config_dict['policy_config']['device_number_per_machine'], config_dict['policy_config']['server_number_per_device'], log_dir)
    print("---------- main函数顺利启动 -----------")

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default='Env/SAC_config.yaml')
    parser.add_argument("--machine_index", type=int, default=0)
    args = parser.parse_args()
    main(args)