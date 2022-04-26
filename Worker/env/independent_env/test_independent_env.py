import logging
from itertools import cycle

import pytest

from rosea import phase_map_dir_color
from rosea.envs import SumoIndependentEnv
from rosea.envs.env_default_config import default_config
from rosea.utils.log_writer import logger_indicator

logger = logging.getLogger(__name__)

act_generator = cycle((0, 1, 0, -1))


def get_green_num(cfg, tls_id):
    return len([
        phase_type
        for phase_type in cfg['action'][tls_id]['phase_type']
        if phase_map_dir_color[phase_type] == 'g'
    ])


def get_area_action(cfg):
    return {
        'cycle': next(act_generator),
        'tls': {
            tls_id: [next(act_generator) for _ in range(get_green_num(cfg, tls_id))]
            for tls_id in cfg['tls_list']
        }
    }


@ pytest.mark.single
@ pytest.mark.envs
@logger_indicator
def test_sumo_independent_env(max_episode=5):
    env = SumoIndependentEnv(default_config)
    obs = env.reset()

    features = env.produce_feature()
    logger.debug('PRODUCE_FEATURES: {}'.format(features))

    obs, reward, done, info = None, None, None, None
    for i in range(max_episode):
        obs, reward, done, info = env.step(get_area_action(default_config))
        logger.debug('STEP: {}, REWARD: {}'.format(i, reward))
        if done:
            logger.debug('EPISODE: finished')
            env.close()
            break
    env.close()
    logger.debug('INIT_OBSERVATION: {}'.format(obs))
    logger.debug('TIMESTEP: {}'.format({'obs': obs, 'reward': reward, 'done': done, 'info': info}))


if __name__ == "__main__":
    test_sumo_independent_env()
