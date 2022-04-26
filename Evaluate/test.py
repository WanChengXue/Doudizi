#!/usr/bin/env python3
from email.policy import default
import os
import ptan
import time
import gym
import pybullet_envs
import argparse
from tensorboardX import SummaryWriter
import numpy as np
import sys
current_path = os.path.abspath(__file__)
root_path = '/'.join(current_path.split('/')[:-2])
sys.path.append(root_path)

import buffer
import  model, common
import pickle

import torch
import torch.optim as optim
import torch.nn.functional as F


ENV_ID = "MinitaurBulletEnv-v0"
GAMMA = 0.99
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
REPLAY_SIZE = 100000
REPLAY_INITIAL = 1000
REWARD_STEPS = 5

TEST_ITERS = 1000

Vmax = 10
Vmin = -10
N_ATOMS = 51
DELTA_Z = (Vmax - Vmin) / (N_ATOMS - 1)


def test_net(net, env, count=10, device="cpu"):
    rewards = 0.0
    steps = 0
    for _ in range(count):
        obs = env.reset()
        while True:
            obs_v = ptan.agent.float32_preprocessor([obs]).to(device)
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
    rewards = rewards_v.data.cpu().numpy()
    dones_mask = dones_mask_t.cpu().numpy().astype(np.bool)
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

    act_net = model.DDPGActor(env.observation_space.shape[0], env.action_space.shape[0]).to(device)
    # act_net.load_state_dict(torch.load('policy.model'))
    crt_net = model.D4PGCritic(env.observation_space.shape[0], env.action_space.shape[0], N_ATOMS, Vmin, Vmax).to(device)
    print(act_net)
    print(crt_net)
    tgt_act_net = ptan.agent.TargetNet(act_net)
    tgt_crt_net = ptan.agent.TargetNet(crt_net)
    torch.save(act_net.state_dict(), 'policy.model')
    torch.save(crt_net.state_dict(), 'critic.model')
    writer = SummaryWriter(comment="-d4pg_" + args.name)
    agent = model.AgentDDPG(act_net, device=device)
    exp_source = buffer.ExperienceSourceFirstLast(env, agent, gamma=GAMMA, steps_count=REWARD_STEPS)
    replay_buffer = buffer.ExperienceReplayBuffer(exp_source, buffer_size=REPLAY_SIZE)
    act_opt = optim.Adam(act_net.parameters(), lr=LEARNING_RATE)
    crt_opt = optim.Adam(crt_net.parameters(), lr=LEARNING_RATE)

    frame_idx = 0
    best_reward = None
    temp_dict = dict()
    temp_dict['current_state'] = []
    temp_dict['instant_reward'] = []
    temp_dict['action'] = []
    temp_dict['next_state'] = []
    temp_dict['done'] = []
    count = 0
    with ptan.common.utils.RewardTracker(writer) as tracker:
        with ptan.common.utils.TBMeanTracker(writer, batch_size=10) as tb_tracker:
            while True:
                frame_idx += 1
                res = replay_buffer.populate(1)
                # if len(res) == 1:
                #     temp_dict['current_state'].append(replay_buffer.buffer[-1][0])
                #     temp_dict['action'].append(replay_buffer.buffer[-1][1])
                #     temp_dict['instant_reward'].append(replay_buffer.buffer[-1][2])
                #     temp_dict['next_state'].append(replay_buffer.buffer[-1][3])
                #     temp_dict['done'].append(False)
                # else:
                #     temp_dict['current_state'].append(replay_buffer.buffer[-1][0])
                #     temp_dict['action'].append(replay_buffer.buffer[-1][1])
                #     temp_dict['instant_reward'].append(replay_buffer.buffer[-1][2])
                #     temp_dict['next_state'].append(replay_buffer.buffer[-1][3])
                #     temp_dict['done'].append(True)
                #     count += 1
                #     if count == 5:
                #         import pickle
                #         open_file = open('ptan.pickle', 'wb')
                #         pickle.dump(temp_dict, open_file)
                #         open_file.close()
                #         exit()
                rewards_steps = exp_source.pop_rewards_steps()
                if rewards_steps:
                    rewards, steps = zip(*rewards_steps)
                    tb_tracker.track("episode_steps", steps[0], frame_idx)
                    tracker.reward(rewards[0], frame_idx)

                if len(replay_buffer) < REPLAY_INITIAL:
                    continue

                batch = replay_buffer.sample(BATCH_SIZE)
                states_v, actions_v, rewards_v, \
                dones_mask, last_states_v = \
                    common.unpack_batch_ddqn(batch, device)
                training_batch = dict()
                training_batch['current_state'] = states_v
                training_batch['actions'] = actions_v
                training_batch['rewards'] = rewards_v
                training_batch['done'] = dones_mask
                training_batch['next_state'] = last_states_v
                # train critic
                crt_opt.zero_grad()
                # ---- 计算当前状态v的分布 ----
                crt_distr_v = crt_net(states_v, actions_v)
                training_batch['current_q_dist'] = crt_distr_v.data
                # ----- 计算下一个状态的动作分布 ----
                last_act_v = tgt_act_net.target_model(last_states_v)
                training_batch['next_action'] = last_act_v.data
                # -------- 计算下一个状态的v值分布 -------
                last_distr_v = tgt_crt_net.target_model(last_states_v, last_act_v)
                training_batch['next_q_dist'] = last_distr_v.data
                proj_distr_v = distr_projection(
                    last_distr_v, rewards_v, dones_mask,
                    gamma=GAMMA**REWARD_STEPS, device=device)
                training_batch['proj_dist'] = proj_distr_v.data
                prob_dist_v = -torch.log(crt_distr_v) * proj_distr_v
                training_batch['prob_dist_v'] = prob_dist_v.data
                critic_loss_v = prob_dist_v.sum(dim=1).mean()
                training_batch['critic_vector'] = prob_dist_v.sum(dim=1).data
                training_batch['critic_vector'] = critic_loss_v.data
                critic_loss_v.backward()
                crt_opt.step()
                tb_tracker.track("loss_critic", critic_loss_v, frame_idx)
                
                # train actor
                act_opt.zero_grad()
                cur_actions_v = act_net(states_v)
                training_batch['actor_net_action'] = cur_actions_v.data
                crt_distr_v = crt_net(states_v, cur_actions_v)
                training_batch['actor_q_dist'] = crt_distr_v
                actor_loss_v = -crt_net.distribution_to_value(crt_distr_v)
                training_batch['actor_vector'] = actor_loss_v.data
                actor_loss_v = actor_loss_v.mean()
                training_batch['actor_loss'] = actor_loss_v.item()
                actor_loss_v.backward()
                act_opt.step()
                tb_tracker.track("loss_actor", actor_loss_v,
                                 frame_idx)

                tgt_act_net.alpha_sync(alpha=1 - 1e-3)
                tgt_crt_net.alpha_sync(alpha=1 - 1e-3)
                # ----- 保存训练了一次之后的，四个网络的参数，进行对比 -----
                open_file = open('ptan_training.pickle', 'wb')
                pickle.dump(training_batch, open_file)
                open_file.close()
                torch.save(act_net.state_dict(), 'ptan_next_policy.model')
                torch.save(crt_net.state_dict(), 'ptan_next_critic.model')
                torch.save(tgt_act_net.target_model.state_dict(), 'ptan_target_next_policy.model')
                torch.save(tgt_crt_net.target_model.state_dict(), 'ptan_target_next_critic.model')
                exit()
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

    pass
