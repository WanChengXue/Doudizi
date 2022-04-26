import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
import sys
current_path = os.path.abspath(__file__)
root_path = '/'.join(current_path.split('/')[:-2])
sys.path.append(root_path)
# ------------ 绘制单点控制的可视化对比曲线 ------------
# ' + policy_name  + '
# MADDPG_' + policy_name  + '
# target_method = 'MADDPG'
# policy_name = 'multi_point_heterogeneous_policy'
# target_method = 'multi_point_heterogeneous_policy'
# target_method = 'Weight_SL'
target_method = 'Independent_D4PG_config'
policy_name = 'Independent_D4PG_config'
def single_point():
    single_point_RL = np.load('Exp/Result/Evaluate/default_policy/default_policy_instant_reward.npy')
    single_point_webster = np.load('Exp/Result/Evaluate/default_policy/Webster_instant_reward.npy')
    plt.figure()
    plt.plot(single_point_RL)
    plt.plot(single_point_webster)
    plt.legend([target_method,'Webster'])
    plt.savefig('Exp/Result/Evaluate/default_policy/reward_comprison.png')
    plt.close()

    waiting_time_RL = np.load('Exp/Result/Evaluate/default_policy/default_policy_waiting_time.npy')
    waiting_time_webster = np.load('Exp/Result/Evaluate/default_policy/Webster_waiting_time.npy')
    plt.figure()
    plt.plot(waiting_time_RL)
    plt.plot(waiting_time_webster)
    plt.legend([target_method, 'Webster'])
    plt.savefig('Exp/Result/Evaluate/default_policy/waiting_time_comprison.png')
    plt.close()

def multi_point():
    # ----------- 处理pickle 文件 ----------
    policy_pickle_file = open('Exp/Result/Evaluate/' + policy_name  + '/' + policy_name  + '_rosea_action.pkl', 'rb')
    webster_pickle_file = open('Exp/Result/Evaluate/' + policy_name  + '/Webster_rosea_action.pkl', 'rb')
    policy_action_data = pickle.load(policy_pickle_file)
    webster_action_data = pickle.load(webster_pickle_file)
    # ------------ 绘制step_time为横轴, 然后cum_reward为纵轴 -------------
    policy_step_time = policy_action_data['step_time']
    webster_step_time = webster_action_data['step_time']
    plt.figure()
    plt.plot(policy_step_time, policy_action_data['cum_reward'])
    plt.plot(webster_step_time, webster_action_data['cum_reward'])
    plt.legend([target_method,'Webster'])
    plt.savefig('Exp/Result/Evaluate/' + policy_name  + '/cum_reward.png')
    plt.close()
    # ----------- 将step_time和cum_reward移除出去 ----------
    policy_action_data.pop('cum_reward')
    policy_action_data.pop('step_time')
    webster_action_data.pop('cum_reward')
    webster_action_data.pop('step_time')
    # ----------- 绘制所有点的cycle,我现在需要对数据进行区分------------
    policy_cycle_dict = dict()
    for key in policy_action_data.keys():
        cycle_dict = policy_action_data[key]['cycle']
        for tls_id in cycle_dict.keys():
            if tls_id not in policy_cycle_dict:
                policy_cycle_dict[tls_id] = [cycle_dict[tls_id]]
            else:
                policy_cycle_dict[tls_id].append(cycle_dict[tls_id])
    # ---------- 读取webster的动作 ----------
    webster_cycle_dict = dict()
    for key in webster_action_data.keys():
        cycle_dict = webster_action_data[key]['cycle']
        for tls_id in cycle_dict.keys():
            if tls_id not in webster_cycle_dict:
                webster_cycle_dict[tls_id] = [cycle_dict[tls_id]]
            else:
                webster_cycle_dict[tls_id].append(cycle_dict[tls_id])
    for key in policy_cycle_dict.keys():
        plt.figure()
        plt.plot(policy_step_time, policy_cycle_dict[key])
        plt.plot(webster_step_time, webster_cycle_dict[key])
        plt.legend([target_method, 'Webster'])
        plt.savefig('Exp/Result/Evaluate/' + policy_name  + '/{}_cycle_comprison.png'.format(key))
    # --------- 再把第一个点的splits进行输出看看 ----------
    policy_splits = dict()
    webster_splits = dict()
    for key in policy_action_data.keys():
        splits_dict = policy_action_data[key]['splits']
        for tls_id in splits_dict.keys():
            if tls_id not in policy_splits.keys():
                policy_splits[tls_id] = [splits_dict[tls_id]]
            else:
                policy_splits[tls_id].append(splits_dict[tls_id])
    # --------- 读取webster的splits -----------------
    for key in webster_action_data.keys():
        splits_dict = webster_action_data[key]['splits']
        for tls_id in splits_dict.keys():
            if tls_id not in webster_splits.keys():
                webster_splits[tls_id] = [splits_dict[tls_id]]
            else:
                webster_splits[tls_id].append(splits_dict[tls_id])
    # ---------- 将两个字典变成numpy类型的数据 ---------
    for key in policy_splits.keys():
        policy_splits[key] = np.array(policy_splits[key])
        webster_splits[key] = np.array(webster_splits[key])
    # ---------- 绘制splits的对比曲线 ------------------
    for i in range(4):
        plt.figure()
        plt.plot(policy_step_time, policy_splits['1'][:,i])
        plt.plot(webster_step_time, webster_splits['1'][:, i])
        plt.legend([target_method,'Webster'])
        plt.savefig('Exp/Result/Evaluate/' + policy_name  + '/1_action_{}_splits_comprison.png'.format(str(i)))
        plt.close()
    # -----------------------------------------------
    multi_point_RL = np.load('Exp/Result/Evaluate/' + policy_name  + '/' + policy_name  + '_instant_reward.npy')
    multi_point_webster = np.load('Exp/Result/Evaluate/' + policy_name  + '/Webster_instant_reward.npy')
    plt.figure()
    plt.plot(policy_step_time, multi_point_RL)
    plt.plot(webster_step_time,multi_point_webster)
    plt.legend([target_method,'Webster'])
    plt.savefig('Exp/Result/Evaluate/' + policy_name  + '/reward_comprison.png')
    plt.close()

    multi_point_waiting_time_SL = np.load('Exp/Result/Evaluate/' + policy_name  + '/' + policy_name  + '_waiting_time.npy')
    multi_point_waiting_time_webster = np.load('Exp/Result/Evaluate/' + policy_name  + '/Webster_waiting_time.npy')
    plt.figure()
    plt.plot(policy_step_time, multi_point_waiting_time_SL[:-1])
    plt.plot(webster_step_time, multi_point_waiting_time_webster[:-1])
    plt.legend([target_method,'Webster'])
    plt.savefig('Exp/Result/Evaluate/' + policy_name  + '/waiting_time_comprison.png')
    plt.close()
multi_point()
