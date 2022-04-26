import logging
import time

import pytest

from rosea.envs import _SumoTestSession
from rosea.utils.log_writer import logger_indicator
from rosea.utils.sumo.traci import Kernel, traci

from .reward import Reward

logger = logging.getLogger(__name__)


@ pytest.mark.single
@pytest.mark.reward
@pytest.mark.envs
@logger_indicator
def test_reward():
    start = time.time()

    with _SumoTestSession() as sess:
        default_config = sess.env._cfg
        traci_kernel = Kernel()
        traci_kernel.reset()
        reward_helper = Reward(default_config)
        reward_helper.reset()
        for i in range(100):
            traci.simulationStep()
            reward_helper.accumulate(1)
        info = {
            'action': {
                'cycle': 1,
                'tls': {
                    'htxdj_wjj': 1,
                    'haxl_wjj': 0,
                    'haxl_htxdj': -1,
                }
            },
            'success': [True, False, True],
            'sample_window': 120
        }
        reward = reward_helper(info)
        reward_helper.flush()

    end = time.time()
    logger.debug('REWARD_HELPER_INPUT_INFO: {}'.format(info))
    logger.debug('REWARD_AFTER_SIMULATE: {}'.format(reward))
    logger.debug('ELAPSED_TIME: {}'.format(end - start))


if __name__ == "__main__":
    test_reward()
