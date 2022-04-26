import os
import time
import gym
import pybullet_envs
import argparse
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F

import sys
import os
current_path = os.path.abspath(__file__)
root_path = '/'.join(current_path.split('/')[:-2])
sys.path.append(root_path)

from Model.FC_model import Toy_model, Toy_critic_categorical
from Utils.data_utils import data_buffer

from tensorboardX import SummaryWriter
from copy import deepcopy
ENV_ID = "HalfCheetahBulletEnv-v0"
GAMMA = 0.99
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
REPLAY_SIZE = 100000
REPLAY_INITIAL = 1000
REWARD_STEPS = 1

TEST_ITERS = 1000

Vmax = 10
Vmin = -10
N_ATOMS = 51
DELTA_Z = (Vmax - Vmin) / (N_ATOMS - 1)


def soft_update(current_network, tart_network, tau):
    """
    Perform DDPG soft update (move target params toward source based on weight
    factor tau)
    Inputs:
        target (torch.nn.Module): Net to copy parameters to
        source (torch.nn.Module): Net whose parameters to copy
        tau (float, 0 < x < 1): Weight factor for update
    """
    for target_param, param in zip(tart_network.parameters(), current_network.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau)


def test_net(net, env, count=10, device="cpu"):
    rewards = 0.0
    steps = 0
    for _ in range(count):
        obs = env.reset()
        while True:
            obs_v = torch.FloatTensor(obs).unsqueeze(0).to(device)
            mu_v = net(obs_v)
            action = mu_v.squeeze(dim=0).data.cpu().numpy()
            action = np.clip(action, -1, 1)
            obs, reward, done, _ = env.step(action)
            rewards += reward
            steps += 1
            if done:
                break
    return rewards / count, steps / count


def distr_projection(next_distr_v, rewards_v, dones_mask_t,
                     gamma, device="cpu"):
    next_distr = next_distr_v.data.cpu().numpy()
    rewards = rewards_v.data.squeeze().cpu().numpy()
    dones_mask = dones_mask_t.squeeze().cpu().numpy().astype(np.bool)
    batch_size = len(rewards)
    proj_distr = np.zeros((batch_size, N_ATOMS), dtype=np.float32)

    for atom in range(N_ATOMS):
        tz_j = np.minimum(Vmax, np.maximum(
            Vmin, rewards + (Vmin + atom * DELTA_Z) * gamma))
        b_j = (tz_j - Vmin) / DELTA_Z
        l = np.floor(b_j).astype(np.int64)
        u = np.ceil(b_j).astype(np.int64)
        eq_mask = u == l
        proj_distr[eq_mask, l[eq_mask]] += \
            next_distr[eq_mask, atom]
        ne_mask = u != l
        proj_distr[ne_mask, l[ne_mask]] += \
            next_distr[ne_mask, atom] * (u - b_j)[ne_mask]
        proj_distr[ne_mask, u[ne_mask]] += \
            next_distr[ne_mask, atom] * (b_j - l)[ne_mask]

    if dones_mask.any():
        proj_distr[dones_mask] = 0.0
        tz_j = np.minimum(Vmax, np.maximum(
            Vmin, rewards[dones_mask]))
        b_j = (tz_j - Vmin) / DELTA_Z
        l = np.floor(b_j).astype(np.int64)
        u = np.ceil(b_j).astype(np.int64)
        eq_mask = u == l
        eq_dones = dones_mask.copy()
        eq_dones[dones_mask] = eq_mask
        if eq_dones.any():
            proj_distr[eq_dones, l[eq_mask]] = 1.0
        ne_mask = u != l
        ne_dones = dones_mask.copy()
        ne_dones[dones_mask] = ne_mask
        if ne_dones.any():
            proj_distr[ne_dones, l[ne_mask]] = (u - b_j)[ne_mask]
            proj_distr[ne_dones, u[ne_mask]] = (b_j - l)[ne_mask]
    return torch.FloatTensor(proj_distr).to(device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False, action='store_true', help='Enable CUDA')
    # parser.add_argument("-n", "--name", required=True, help="Name of the run")
    parser.add_argument("-n", "--name", default='D4PG')
    args = parser.parse_args()
    device = torch.device("cuda" if args.cuda else "cpu")

    save_path = os.path.join("saves", "d4pg-" + args.name)
    os.makedirs(save_path, exist_ok=True)

    env = gym.make(ENV_ID)
    test_env = gym.make(ENV_ID)

    act_config = dict()
    act_config['action_dim'] = env.action_space.shape[0]
    act_config['state_dim'] = env.observation_space.shape[0]
    critic_config = dict()
    critic_config['action_dim'] = env.action_space.shape[0]
    critic_config['state_dim'] = env.observation_space.shape[0]
    critic_config['n_atoms'] = N_ATOMS
    critic_config['value_min'] = Vmin
    critic_config['value_max'] = Vmax
    critic_config['hidden_dim'] = 400
    act_net = Toy_model(act_config).to(device)
    crt_net = Toy_critic_categorical(critic_config).to(device)
    tgt_act_net = Toy_model(act_config).to(device)
    tgt_crt_net = Toy_critic_categorical(critic_config).to(device)
    tgt_act_net.load_state_dict(act_net.state_dict())
    tgt_crt_net.load_state_dict(crt_net.state_dict())
    print(act_net)
    print(crt_net)
    writer = SummaryWriter(comment="-d4pg_" + args.name)
    buffer_config = dict()
    buffer_config['capacity'] = 10000
    buffer_config['batch_size'] = 256
    buffer = data_buffer(buffer_config)
    buffer.init_buffer()
    act_opt = optim.Adam(act_net.parameters(), lr=LEARNING_RATE)
    crt_opt = optim.Adam(crt_net.parameters(), lr=LEARNING_RATE)
    reward_record = []
    frame_idx = 0
    best_reward = None
    for i in range(10000):
        frame_idx += 1
        current_state = env.reset()
        reward_list = []
        while True:
            with torch.no_grad():
                action = act_net(torch.FloatTensor(current_state).unsqueeze(0).to(device)).squeeze().cpu().numpy()
                next_state, instant_reward, done, _ = env.step(action)
                current_state = next_state
                reward_list.append(instant_reward)
                data_dict = dict()
                data_dict['current_state'] = deepcopy(current_state)
                data_dict['instant_reward'] = instant_reward
                data_dict['next_state'] = deepcopy(next_state)
                data_dict['done'] = done
                data_dict['action'] = action
                buffer.append_data([data_dict])
            if buffer.buffer_size > 300:
                batch = buffer.sample_data()
                states_v = torch.FloatTensor(batch['current_state']).to(device)
                actions_v = torch.FloatTensor(batch['action']).to(device)
                rewards_v = torch.FloatTensor(batch['instant_reward']).to(device)
                last_states_v = torch.FloatTensor(batch['next_state']).to(device)
                dones_mask = torch.FloatTensor(batch['done']).to(device)
                # train critic
                crt_opt.zero_grad()
                prob_dist_v = crt_net(states_v, actions_v)
                last_act_v = tgt_act_net(last_states_v)
                last_distr_v = tgt_crt_net(last_states_v, last_act_v)
                proj_distr_v = distr_projection(last_distr_v, rewards_v, dones_mask,gamma=GAMMA**REWARD_STEPS, device=device)
                prob_dist_v = -prob_dist_v * proj_distr_v
                critic_loss_v = prob_dist_v.sum(dim=1).mean()
                critic_loss_v.backward()
                crt_opt.step()
                # train actor
                act_opt.zero_grad()
                cur_actions_v = act_net(states_v)
                crt_distr_v = crt_net(states_v, cur_actions_v)
                actor_loss_v = -crt_net.distribution_to_value(crt_distr_v)
                actor_loss_v = actor_loss_v.mean()
                actor_loss_v.backward()
                act_opt.step()
                soft_update(act_net, tgt_act_net, 1e-3)
                soft_update(crt_net, tgt_crt_net, 1e-3)
                
                if frame_idx % TEST_ITERS == 0:
                    ts = time.time()
                    rewards, steps = test_net(act_net, test_env, device=device)
                    print("Test done in %.2f sec, reward %.3f, steps %d" % (
                        time.time() - ts, rewards, steps))
                    writer.add_scalar("test_reward", rewards, frame_idx)
                    writer.add_scalar("test_steps", steps, frame_idx)
                    if best_reward is None or best_reward < rewards:
                        if best_reward is not None:
                            print("Best reward updated: %.3f -> %.3f" % (best_reward, rewards))
                            name = "best_%+.3f_%d.dat" % (rewards, frame_idx)
                            fname = os.path.join(save_path, name)
                            torch.save(act_net.state_dict(), fname)
                        best_reward = rewards
            if done:
                reward_record.append(sum(reward_list))
                break
    np.save(reward_record, 'test.npy')