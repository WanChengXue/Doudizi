from copy import deepcopy

from rosea.envs.base_env import BaseEnv
from rosea.envs.env_default_config import longhua as default_config
from rosea.envs.sumo_simulator import SUMO_Simulator
from rosea.utils.checker import check_no_offset

from .elements.controller import MultiPointIndependentController
from .elements.observer import Observer
from .elements.reward import Reward


import copy
import logging
from math import ceil

from rosea.envs.base_element import BaseElement
from rosea.envs.base_simulator import BaseSimulator
from rosea.feature import misc
from rosea.feature.state_manager import StateManager
from rosea.signal_controller.controller import BaseController
from rosea.utils.checker import check_cfg
from rosea.utils.log_writer import timethis
from rosea.utils.sumo.traci import Kernel


class BaseEnv(object):

    def __init__(
        self,
        cfg,
        obs_helper: BaseElement,
        reward_helper: BaseElement,
        action_helper: BaseController,
        simulator: BaseSimulator,
    ) -> None:
        r"""
        Overview:
            initialize sumo WJ 3 intersection Env
        Arguments:
            - cfg (:obj:`dict`): config, you can refer to `envs/env_default_config.py`
        """
        self._cfg = copy.deepcopy(cfg)
        check_cfg(self._cfg)
        self._logger = logging.getLogger(__name__)

        self._logger.info('ENV_CLASS: {}'.format(self.__class__.__name__))
        self._logger.info('ENV_CONFIG: {}'.format(self._cfg))

        self.mode = self._cfg['mode']

        self._traci_kernel = Kernel()
        self._obs_helper = obs_helper
        self._reward_helper = reward_helper
        self._action_helper = action_helper
        self._simulator = simulator

        self._sample_interval = 2
        self._state = StateManager()

        self._feature_calc = {
            'reward': misc.RewardFeature(self._cfg, self._state, action_helper)
        }

        if not self.mode['reset'] == 'random':
            self.set_seed(0)

    def set_seed(self, seed):
        self._simulator.set_seed(seed)

    def reset(self, route=None):
        r"""
        Overview:
            reset the current env
        Returns:
            - obs (:obj:`torch.Tensor` or :obj:`dict`): the observation to env after reset
        """

        if route is None and self.mode['run'] == 'train':
            route = 'random'
        if route is None and len(self) == 1 and self.mode['run'] == 'evaluate':
            route = 0

        self._logger.debug('ENV: reset')
        self._simulator.close()
        self._simulator.launch(route)
        self._traci_kernel.reset()

        default_states = self._simulator.get_default_states()

        self._action_helper.reset(self.mode['reset'], default_states=default_states)
        self._reward_helper.reset()
        self._obs_helper.reset()

        self._cum_reward = 0
        obs, *_ = self.step(None)
        return obs

    def _check_status(self):
        assert self._simulator.get_status('launched')
        done = self._simulator.get_status('done')
        if done:
            raise RuntimeError('RUN_STEP_AFTER_DONE')

    def _flush(self):
        self._obs_helper.flush()
        self._reward_helper.flush()

    def _accumulate(self, simulation_interval):
        self._action_helper.accumulate()
        self._obs_helper.accumulate(simulation_interval)
        self._reward_helper.accumulate(simulation_interval)

    @timethis
    def step(self, action: dict, simulate_timesteps=None):
        """
        Overview:
            step the sumo env with action
        Arguments:
            - action = {
                'cycle': int,
                'tls': {
                    tls_id: int(for two phase) or list[int](for multi phase),
                },
                'offset': {tls_id: int},  # optional, default is {tls_id: 0}
            }

        Returns:
            - timpstep(:obj:`SumoArterialEnv.timestep`): the timestep, contain obs(:obj:`torch.Tensor` or :obj:`dict`)\
            reward(:obj:`float` or :obj:`dict`), done(:obj:`bool`) and info(:obj:`dict`)
        """
        status = dict()
        self._check_status()
        self._action_helper(action)
        max_cycle = max((unit.param.get_cycle('real') for tls_id, unit in self._action_helper))
        if simulate_timesteps is None:
            simulate_timesteps = ceil(max_cycle)  # cycle
        self._logger.info('SIMULATE_TIMESTEPS: {}'.format(simulate_timesteps))

        self._flush()
        for _ in self._simulator.simulate(simulate_timesteps, self._sample_interval):
            self._accumulate(self._sample_interval)
        status['done'] = self._simulator.get_status('done')

        obs = self._obs_helper()
        reward_info = {'action': action, 'sample_window': simulate_timesteps}
        self._logger.debug('REWARD_INFO: {}'.format(reward_info))
        reward = self._reward_helper(reward_info)
        status['jam'] = self._feature_calc['reward'](
            'jam',
            wait_time=self._reward_helper.produce_feature()['wait_time_per_vehicle'],
            threshold=5
        )
        self._cum_reward += simulate_timesteps * reward
        current_time = self._traci_kernel.access('simulation', 'TIME')
        mean_reward = self._cum_reward / current_time
        info = {
            'cum_reward': self._cum_reward,
            'sample_window': simulate_timesteps,
            'eval_reward': simulate_timesteps * reward,
            'status': status,
            'current_time': current_time,
            'mean_reward': mean_reward,
            'final_eval_reward': mean_reward,
        }
        done = self._get_done(status)
        return obs, reward, done, info

    def _get_done(self, status):
        done = status['done']
        if 'on_jam' in self.mode['done']:
            done = done or status['jam']
        return done

    def produce_feature(self, reward_info=None, obs_info=None):
        obs = self._obs_helper.produce_feature(obs_info)
        reward = self._reward_helper.produce_feature(reward_info)
        return {'observation': obs, 'reward': reward}

    def __len__(self):
        '''Returs number of routes
        '''
        return len(self._simulator)

    def close(self):
        self._simulator.close()


class SumoIndependentEnv(BaseEnv):
    def __init__(self, cfg=None):
        if cfg is None:
            cfg = default_config
        cfg = deepcopy(cfg)
        check_no_offset(cfg)
        obs_helper = Observer(cfg)
        reward_helper = Reward(cfg)
        simulator = SUMO_Simulator(cfg)
        action_helper = MultiPointIndependentController(cfg)
        super().__init__(cfg, obs_helper, reward_helper, action_helper, simulator)
