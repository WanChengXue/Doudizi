import multiprocessing
from multiprocessing import Process
import argparse
import os
import sys
current_path = os.path.abspath(__file__)
root_path = '/'.join(current_path.split('/')[:-2])
sys.path.append(root_path)

# from Worker.data_generate import sample_generator
from Worker.rollout import sample_generator
def single_process_generate_sample(config_path, port_num):
    worker = sample_generator(config_path, port_num=port_num)
    while True:
        worker.run()


if __name__=='__main__':
    # ---------- 导入配置文件 ------------
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='/Config/Training/MAPPO_config.yaml')
    parser.add_argument('--parallel_env', type=int, default=10)
    args = parser.parse_args()
    abs_path = '/'.join(os.path.abspath(__file__).split('/')[:-2])
    concatenate_path = abs_path + args.config_path
    args.config_path = concatenate_path
    # parallel_env_number = 10
    multiprocessing.set_start_method('spawn')
    print('---------- 并行化的worker数目为 {} -----------'.format(args.parallel_env))
    for i in range(args.parallel_env):
        # logger_path = pathlib.Path("./config_folder") / ("process_"+ str(i))
        # logger_name = "Process_"+ str(i)
        # logger = setup_logger(logger_name, logger_path)
        # p = Process(target=single_process_generate_sample,args=(logger,))
        # p.start()
        p = Process(target=single_process_generate_sample, args=(args.config_path, i))
        p.start()

#  worker ssh -p 22112 serena@10.19.92.79

