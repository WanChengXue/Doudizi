'''
Author: error: git config user.name && git config user.email & please set dead value or install git
Date: 2022-08-16 20:05:39
LastEditors: error: git config user.name && git config user.email & please set dead value or install git
LastEditTime: 2022-08-16 22:09:28
FilePath: /Doudizi/Env/env_utils.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import numpy as np
from Utils.data_utils import convert_data_format_to_torch_interference
from Env.env import Env

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
        

    def reset(self):
        initial_obs, x_no_action, z, init_position = _format_observation(self.env.reset())
        # self.episode_return = torch.zeros(1, 1)
        # initial_done = torch.ones(1, 1, dtype=torch.bool)
        assert init_position != self.buildin_ai
        self.legal_actions = initial_obs['legal_actions']
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
        
    def set_buildin_ai(self, agent, trained_ai):
        self.buildin_ai = agent # 这个表示的是传入的內置AI
        self.trained_ai = trained_ai # 这个表示需要进行训练的AI
        
    def step(self, action, logger):
        # ---- 可以训练的AI传入的动作，然后环境step之后获得内置AI需要的状态 -----
        # -------- 传入的动作是一个数字token，实际的执行动作需要从legal_acitons中获取 ------
        _op_obs, _reward, _done, _ = self.env.step(self.legal_actions[action])

        # self.episode_return += reward
        # episode_return = self.episode_return 

        # if done:
        #     obs = self.env.reset()
        #     self.episode_return = torch.zeros(1, 1)

        
        if not _done:
            _op_obs, x_no_action, z, op_opsition = _format_observation(_op_obs)
            self.legal_actions = _op_obs['legal_actions']
            op_obs = {
                'x': _op_obs['x_batch'],
                'z': _op_obs['z_batch']
            }
            # reward = torch.tensor(reward).view(1, 1)
            # done = torch.tensor(done).view(1, 1)
            # ---- 如果游戏没有结束，则调用buildin ai ---
            if op_opsition == self.buildin_ai:
                try:
                    
                    buildin_ai_action = self.buildin_ai.compute_action_eval_mode(convert_data_format_to_torch_interference(op_obs))
                except :
                    logger.info("---- 内置AI报错，输入的数据为 {} -------".foramt(op_obs))
                next_obs, reward, done, _ = self.env.step(self.legal_actions[buildin_ai_action])
                if not done:
                    next_obs, x_no_action, z, op_position = _format_observation(next_obs)
                    assert op_position != self.buildin_ai
                    self.legal_actions = next_obs['legal_actions']
                    next_obs = {
                        'x': next_obs['x_batch'],
                        'z': next_obs['z_batch']
                    }
                else:
                    next_obs = None
            else:
                # ------- 如果不是内置ai动作 ------
                next_obs = op_obs
                done = _done
                reward = _reward 
        else:
            # ---- 当前玩家执行完成了动作之后，就直接结束了 ---
            next_obs = None
            done = _done
            reward = _reward 
    
        return next_obs, reward[self.trained_ai], done
        # return position, obs, dict(
        #     done=done,
        #     episode_return=episode_return,
        #     obs_x_no_action=x_no_action,
        #     obs_z=z,
        #     )

    def close(self):
        self.env.close()
