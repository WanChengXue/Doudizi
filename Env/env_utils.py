'''
Author: error: git config user.name && git config user.email & please set dead value or install git
Date: 2022-08-16 20:05:39
LastEditors: error: git config user.name && git config user.email & please set dead value or install git
LastEditTime: 2022-08-19 18:52:28
FilePath: /Doudizi/Env/env_utils.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import numpy as np
from Utils.data_utils import convert_data_format_to_torch_interference
from Env.env import Env
import copy
def _format_observation(obs):
    position = obs['position']
    # if not device == "cpu":
    #     device = 'cuda:' + str(device)
    # # device = torch.device(device)
    # x_batch = torch.from_numpy(obs['x_batch']).to(device)
    # z_batch = torch.from_numpy(obs['z_batch']).to(device)
    # x_no_action = torch.from_numpy(obs['x_no_action'])
    # z = torch.from_numpy(obs['z'])
    x_batch = np.array(obs['x_batch'])
    z_batch = np.array(obs['z_batch'])
    x_no_action = np.array(obs['x_no_action'])
    z = np.array(obs['z'])
    obs = {'x_batch': x_batch,
        'z_batch': z_batch,
        'legal_actions': obs['legal_actions'],
        }
    return  obs, x_no_action, z, position

class Environment:
    def __init__(self):
        """ Initialzie this environment wrapper
        """
        self.env = Env()
        self.episode_return = None
        

    def reset(self, visualize_process = False):
        initial_obs, x_no_action, z, init_position = _format_observation(self.env.reset())
        self.visualize_process = visualize_process
        assert init_position == 'landlord'
        self.landlord_legal_actions = initial_obs['legal_actions']
        self.record = dict()
        self.record['landlord'] = dict()
        self.record['farmer'] = dict()
        self.record['landlord']['hand'] = []
        self.record['landlord']['action'] = []
        self.record['farmer']['hand'] = []
        self.record['farmer']['action'] = []
        self.record['landlord']['hand'].append(copy.deepcopy(self.env._env.info_sets['landlord'].player_hand_cards))

        return {
            'x': initial_obs['x_batch'],
            'z': initial_obs['z_batch']
        }
        
        # return initial_position, initial_obs, dict(
        #     done=initial_done,
        #     episode_return=self.episode_return,
        #     obs_x_no_action=x_no_action,
        #     obs_z=z,
        #     )
    @property
    def get_legal_action_length(self):
        # ------ 返回合法动作的数量 ------
        return len(self.landlord_legal_actions)
        
    def set_buildin_ai(self, agent, trained_ai):
        self.buildin_ai = agent # 这个表示的是传入的內置AI
        self.trained_ai = trained_ai # 这个表示需要进行训练的AI
        
    def step(self, action):
        # ---- 可以训练的AI传入的动作，然后环境step之后获得内置AI需要的状态 -----
        # -------- 传入的动作是一个数字token，实际的执行动作需要从legal_acitons中获取 ------
        if self.visualize_process:
            print('-------- 地主手牌为 {} -----'.format(self.env._env.info_sets['landlord'].player_hand_cards))
            print("-------- 地主出牌 {}---------".format(self.landlord_legal_actions[action]))
        self.record['landlord']['action'].append(self.landlord_legal_actions[action])
        _op_obs, _reward, _done, _ = self.env.step(self.landlord_legal_actions[action])
        self.record['landlord']['hand'].append(copy.deepcopy(self.env._env.info_sets['landlord'].player_hand_cards))
        
        #  step之后，返回得到的_op_obs应该是内置AI(farmer)的状态了 
        # self.episode_return += reward
        # episode_return = self.episode_return 

        # if done:
        #     obs = self.env.reset()
        #     self.episode_return = torch.zeros(1, 1)
        
        # ---- 如果landlord执行完成了动作后没有结束游戏，则轮到farmer开始动作 --------
        if not _done:
            _op_obs, x_no_action, z, op_opsition = _format_observation(_op_obs)
            assert op_opsition != self.trained_ai
            self.farmer_legal_actions = _op_obs['legal_actions']
            op_obs = {
                'x': _op_obs['x_batch'],
                'z': _op_obs['z_batch']
            }
            # reward = torch.tensor(reward).view(1, 1)
            # done = torch.tensor(done).view(1, 1)
            self.record['farmer']['hand'].append(copy.deepcopy(self.env._env.info_sets['farmer'].player_hand_cards))
            buildin_ai_action = self.buildin_ai.compute_action_eval_mode(convert_data_format_to_torch_interference(op_obs))
            self.record['farmer']['action'].append(self.farmer_legal_actions[buildin_ai_action])
            if self.visualize_process:
                print('======== 农民手牌 {} =========='.format(self.env._env.info_sets['farmer'].player_hand_cards))
                print('======== 农民出牌 {} =========='.format(self.farmer_legal_actions[buildin_ai_action]))
            
                
            next_obs, after_buildin_reward, after_buildin_done, _ = self.env.step(self.farmer_legal_actions[buildin_ai_action])
            # ------- 如果对手执行完成了动作后，游戏没有结束，那么轮到landlord了 -----
            if not after_buildin_done:
                _next_obs, x_no_action, z, human_position = _format_observation(next_obs)
                assert human_position == self.trained_ai
                self.landlord_legal_actions = _next_obs['legal_actions']
                next_obs = {
                    'x': _next_obs['x_batch'],
                    'z': _next_obs['z_batch']
                }
                
            # -- 如果说对手执行完成了动作后，游戏结束，则landlord的next obs变成None
            else:
                next_obs = None
            reward = after_buildin_reward
            done = after_buildin_done
            # else:
            #     # ------- 如果不是内置ai动作 ------
            #     next_obs = op_obs
            #     done = _done
            #     reward = _reward 
        else:
            # ---- 当前玩家执行完成了动作之后，就直接结束了 ---
            next_obs = None
            done = _done
            reward = _reward 
    
        return next_obs, reward[self.trained_ai], done


    def close(self):
        self.env.close()
