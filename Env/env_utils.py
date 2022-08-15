import numpy as np
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
    return position, obs, x_no_action, z

class Environment:
    def __init__(self):
        """ Initialzie this environment wrapper
        """
        self.env = Env()
        self.episode_return = None
        

    def reset(self):
        initial_position, initial_obs, x_no_action, z = _format_observation(self.env.reset())
        # self.episode_return = torch.zeros(1, 1)
        # initial_done = torch.ones(1, 1, dtype=torch.bool)
        return initial_obs
        # return initial_position, initial_obs, dict(
        #     done=initial_done,
        #     episode_return=self.episode_return,
        #     obs_x_no_action=x_no_action,
        #     obs_z=z,
        #     )
        
    def set_buildin_ai(self, agent, trained_ai):
        self.buildin_ai = agent # 这个表示的是传入的內置AI
        self.trained_ai = trained_ai # 这个表示需要进行训练的AI
        
    def step(self, action):
        # ---- 可以训练的AI传入的动作，然后环境step之后获得内置AI需要的状态 -----
        op_obs, _reward, _done, _ = self.env.step(action)

        # self.episode_return += reward
        # episode_return = self.episode_return 

        # if done:
        #     obs = self.env.reset()
        #     self.episode_return = torch.zeros(1, 1)

        position, obs, x_no_action, z = _format_observation(obs)
        # reward = torch.tensor(reward).view(1, 1)
        # done = torch.tensor(done).view(1, 1)
        if not _done:
            # ---- 如果游戏没有结束，则调用buildin ai ---
            buildin_ai_action = self.buildin_ai.compute_action_eval_mode(op_obs)
            next_obs, reward, done, _ = self.env.step(buildin_ai_action)
        else:
            # ---- 当前玩家执行完成了动作之后，就直接结束了 ---
            next_obs = op_obs
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
