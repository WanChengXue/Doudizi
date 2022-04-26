import logging
import time

import pytest

from rosea.envs import SumoIndependentEnv, _SumoTestSession
from rosea.utils.log_writer import logger_indicator
from rosea.utils.sumo.traci import Kernel, traci

from .observer import Observer

logger = logging.getLogger(__name__)


@ pytest.mark.single
@pytest.mark.obs
@pytest.mark.envs
@logger_indicator
def test_observer():
    start = time.time()

    with _SumoTestSession(SumoIndependentEnv) as sess:
        default_config = sess.env._cfg
        traci_kernel = Kernel()
        traci_kernel.reset()
        obs_helper = Observer(default_config)
        obs_helper.reset()
        for i in range(100):
            traci.simulationStep()
            obs_helper.accumulate(1)
        observation = obs_helper()
        obs_helper.flush()

    end = time.time()
    logger.debug('OBSERVATION_AFTER_SIMULATE: {}'.format(observation))
    logger.debug('ELAPSED_TIME: {}'.format(end - start))


if __name__ == "__main__":
    test_observer()
