import torch
from Env.env_utils import Environment
from Env.env import Env

test_env = Environment(Env())
init_obs = test_env.reset()
init_action = [7]
test_env.step(init_action)
