"""
Author: error: git config user.name && git config user.email & please set dead value or install git
Date: 2022-08-16 20:05:39
LastEditors: error: git config user.name && git config user.email & please set dead value or install git
LastEditTime: 2022-08-19 18:52:28
FilePath: /Doudizi/Env/env_utils.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
"""
from distutils.command.build import build
import numpy as np
from Utils.data_utils import convert_data_format_to_torch_interference
from Env.env import Env
import copy
from Env.game import EnvCard2RealCard, RealCard2EnvCard


def _format_observation(obs):
    position = obs["position"]
    # if not device == "cpu":
    #     device = 'cuda:' + str(device)
    # # device = torch.device(device)
    # x_batch = torch.from_numpy(obs['x_batch']).to(device)
    # z_batch = torch.from_numpy(obs['z_batch']).to(device)
    # x_no_action = torch.from_numpy(obs['x_no_action'])
    # z = torch.from_numpy(obs['z'])
    x_batch = np.array(obs["x_batch"])
    z_batch = np.array(obs["z_batch"])
    x_no_action = np.array(obs["x_no_action"])
    z = np.array(obs["z"])
    obs = {
        "x_batch": x_batch,
        "z_batch": z_batch,
        "legal_actions": obs["legal_actions"],
    }
    return obs, x_no_action, z, position


class Environment:
    def __init__(self):
        """Initialzie this environment wrapper"""
        self.env = Env()
        self.episode_return = None
        self.human_action = False

    def reset(self, visualize_process=False):
        initial_obs, x_no_action, z, init_position = _format_observation(
            self.env.reset()
        )
        self.visualize_process = visualize_process
        if self.human_action or self.visualize_process:
            print("三张底牌为：{}".format(self.convert_number_to_str(self.env.diapi)))
            print("九张废牌为:{}".format(self.convert_number_to_str(self.env.feipai)))
            print("让牌数为:{}".format(self.env.rangpaishus))
            print("底牌倍数为:{}".format(self.env.dipai_beishu))
            print("叫地主倍数为:{}".format(self.env.jiaodizhu_beishu))

        assert init_position == "landlord"
        self.landlord_legal_actions = initial_obs["legal_actions"]
        self.record = dict()
        self.record["landlord"] = dict()
        self.record["farmer"] = dict()
        self.record["landlord"]["hand"] = []
        self.record["landlord"]["action"] = []
        self.record["farmer"]["hand"] = []
        self.record["farmer"]["action"] = []
        self.record["landlord"]["hand"].append(
            copy.deepcopy(self.env._env.info_sets["landlord"].player_hand_cards)
        )
        if self.trained_ai == "farmer":
            # --- 如果是训练farmer，最开始就需要调用内置ai，或者人传入一个动作 ---
            if self.human_action:
                while True:
                    try:
                        print(
                            "======== 地主手牌 {} ==========".format(
                                self.convert_number_to_str(
                                    self.env._env.info_sets[
                                        "landlord"
                                    ].player_hand_cards
                                )
                            )
                        )
                        input_card = input(f"请输入你要出的牌，如果不出输入pass，此外用逗号分开多张牌：")
                        # 将这个字符串变成一个列表
                        if input_card == "pass":
                            input_action = []
                        else:
                            input_action = [
                                RealCard2EnvCard[_card]
                                for _card in input_card.split(",")
                            ]
                            _op_obs, _reward, _done, _ = self.env.step(input_action)
                        break
                    except:
                        print("无效输入!")
            else:
                init_obs = {"x": initial_obs["x_batch"], "z": initial_obs["z_batch"]}
                init_landlord_obs = convert_data_format_to_torch_interference(init_obs)
                buildin_ai_action = self.buildin_ai.compute_action_eval_mode(
                    init_landlord_obs
                )
                self.record["landlord"]["action"].append(
                    copy.deepcopy(self.landlord_legal_actions[buildin_ai_action])
                )
                if self.visualize_process:
                    print(
                        "-------- 地主手牌为 {} -----".format(
                            self.convert_number_to_str(
                                self.env._env.info_sets["landlord"].player_hand_cards
                            )
                        )
                    )
                _op_obs, _reward, _done, _ = self.env.step(
                    self.landlord_legal_actions[buildin_ai_action]
                )
            self.farmer_legal_actions = _op_obs["legal_actions"]
            initial_obs, x_no_action, z, init_position = _format_observation(_op_obs)
            if self.visualize_process:
                print(
                    "-------- 地主出牌 {}---------".format(
                        self.convert_number_to_str(
                            self.landlord_legal_actions[buildin_ai_action]
                        )
                    )
                )
        return {"x": initial_obs["x_batch"], "z": initial_obs["z_batch"]}

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
        self.buildin_ai = agent  # 这个表示的是传入的內置AI
        self.trained_ai = trained_ai  # 这个表示需要进行训练的AI

    def convert_number_to_str(self, card_list):
        # 传入卡片列表，然后返回实际的牌
        if len(card_list) != 0:
            real_card_list = [EnvCard2RealCard[token] for token in card_list]
            return "".join(real_card_list)
        else:
            return "pass"

    def _step_landlord(self, action):
        if self.visualize_process or self.human_action:
            print(
                "-------- 地主手牌为 {} -----".format(
                    self.convert_number_to_str(
                        self.env._env.info_sets["landlord"].player_hand_cards
                    )
                )
            )
            print(
                "-------- 地主出牌 {}---------".format(
                    self.convert_number_to_str(self.landlord_legal_actions[action])
                )
            )
        self.record["landlord"]["action"].append(self.landlord_legal_actions[action])
        _op_obs, _reward, _done, _ = self.env.step(self.landlord_legal_actions[action])
        self.record["landlord"]["hand"].append(
            copy.deepcopy(self.env._env.info_sets["landlord"].player_hand_cards)
        )

        # ---- 如果landlord执行完成了动作后没有结束游戏，则轮到farmer开始动作 --------
        if not _done:
            _op_obs, x_no_action, z, op_opsition = _format_observation(_op_obs)
            # assert op_opsition != self.trained_ai
            self.farmer_legal_actions = _op_obs["legal_actions"]
            op_obs = {"x": _op_obs["x_batch"], "z": _op_obs["z_batch"]}
            # reward = torch.tensor(reward).view(1, 1)
            # done = torch.tensor(done).view(1, 1)
            self.record["farmer"]["hand"].append(
                copy.deepcopy(self.env._env.info_sets["farmer"].player_hand_cards)
            )
            if self.human_action:
                while True:
                    try:
                        # ------ termimal传入一个列表
                        print(
                            "======== 农民手牌 {} ==========".format(
                                self.convert_number_to_str(
                                    self.env._env.info_sets["farmer"].player_hand_cards
                                )
                            )
                        )
                        input_card = input(f"请输入你要出的牌，如果不出输入pass，此外用逗号分开多张牌：")
                        # 将这个字符串变成一个列表
                        if input_card == "pass":
                            input_action = []
                        else:
                            input_action = [
                                RealCard2EnvCard[_card]
                                for _card in input_card.split(",")
                            ]
                        (
                            next_obs,
                            after_buildin_reward,
                            after_buildin_done,
                            _,
                        ) = self.env.step(input_action)
                        break
                    except:
                        print("无效输入!")
            else:
                buildin_ai_action = self.buildin_ai.compute_action_eval_mode(
                    convert_data_format_to_torch_interference(op_obs)
                )
                self.record["farmer"]["action"].append(
                    self.farmer_legal_actions[buildin_ai_action]
                )
                if self.visualize_process:
                    print(
                        "======== 农民手牌 {} ==========".format(
                            self.env._env.info_sets["farmer"].player_hand_cards
                        )
                    )
                    print(
                        "======== 农民出牌 {} ==========".format(
                            self.farmer_legal_actions[buildin_ai_action]
                        )
                    )

                next_obs, after_buildin_reward, after_buildin_done, _ = self.env.step(
                    self.farmer_legal_actions[buildin_ai_action]
                )
            # ------- 如果对手执行完成了动作后，游戏没有结束，那么轮到landlord了 -----
            if not after_buildin_done:
                _next_obs, x_no_action, z, _next_position = _format_observation(
                    next_obs
                )
                assert _next_position == self.trained_ai
                self.landlord_legal_actions = _next_obs["legal_actions"]
                next_obs = {"x": _next_obs["x_batch"], "z": _next_obs["z_batch"]}

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

    def _step_farmer(self, action):
        if self.visualize_process or self.human_action:
            print(
                "-------- 农民手牌为 {} -----".format(
                    self.convert_number_to_str(
                        self.env._env.info_sets["farmer"].player_hand_cards
                    )
                )
            )
            print(
                "-------- 农民出牌 {}---------".format(
                    self.convert_number_to_str(self.farmer_legal_actions[action])
                )
            )
        self.record["farmer"]["action"].append(self.farmer_legal_actions[action])
        _op_obs, _reward, _done, _ = self.env.step(self.farmer_legal_actions[action])
        self.record["farmer"]["hand"].append(
            copy.deepcopy(self.env._env.info_sets["farmer"].player_hand_cards)
        )
        # ---- 如果farmer执行完成了动作后没有结束游戏，则轮到landlord开始动作 --------
        if not _done:
            _op_obs, x_no_action, z, op_opsition = _format_observation(_op_obs)
            self.landlord_legal_actions = _op_obs["legal_actions"]
            op_obs = {"x": _op_obs["x_batch"], "z": _op_obs["z_batch"]}
            self.record["landlord"]["hand"].append(
                copy.deepcopy(self.env._env.info_sets["landlord"].player_hand_cards)
            )
            if self.human_action:
                while True:
                    try:
                        print(
                            "======== 地主手牌 {} ==========".format(
                                self.convert_number_to_str(
                                    self.env._env.info_sets[
                                        "landlord"
                                    ].player_hand_cards
                                )
                            )
                        )
                        input_card = input(f"请输入你要出的牌，如果不出输入pass，此外用逗号分开多张牌：")
                        # 将这个字符串变成一个列表
                        if input_card == "pass":
                            input_action = []
                        else:
                            input_action = [
                                RealCard2EnvCard[_card]
                                for _card in input_card.split(",")
                            ]
                        (
                            next_obs,
                            after_buildin_reward,
                            after_buildin_done,
                            _,
                        ) = self.env.step(input_action)
                        break
                    except:
                        print("无效输入!")
            else:
                buildin_ai_action = self.buildin_ai.compute_action_eval_mode(
                    convert_data_format_to_torch_interference(op_obs)
                )
                self.record["landlord"]["action"].append(
                    self.landlord_legal_actions[buildin_ai_action]
                )
                if self.visualize_process:
                    print(
                        "======== 地主手牌 {} ==========".format(
                            self.convert_number_to_str(
                                self.env._env.info_sets["landlord"].player_hand_cards
                            )
                        )
                    )
                    print(
                        "======== 地主出牌 {} ==========".format(
                            self.convert_number_to_str(
                                self.landlord_legal_actions[buildin_ai_action]
                            )
                        )
                    )

                next_obs, after_buildin_reward, after_buildin_done, _ = self.env.step(
                    self.landlord_legal_actions[buildin_ai_action]
                )
            # ------- 如果对手执行完成了动作后，游戏没有结束，那么轮到farmer了 -----
            if not after_buildin_done:
                _next_obs, x_no_action, z, _next_position = _format_observation(
                    next_obs
                )
                assert _next_position == self.trained_ai
                self.farmer_legal_actions = _next_obs["legal_actions"]
                next_obs = {"x": _next_obs["x_batch"], "z": _next_obs["z_batch"]}
            # -- 如果说对手执行完成了动作后，游戏结束，则farmer的next obs变成None
            else:
                next_obs = None
            reward = after_buildin_reward
            done = after_buildin_done

        else:
            # ---- 当前玩家执行完成了动作之后，就直接结束了 ---
            next_obs = None
            done = _done
            reward = _reward
        return next_obs, reward[self.trained_ai], done

    def step(self, action):
        # ---- 可以训练的AI传入的动作，然后环境step之后获得内置AI需要的状态 -----
        # -------- 传入的动作是一个数字token，实际的执行动作需要从legal_acitons中获取 ------
        if self.trained_ai == "landlord":
            return self._step_landlord(action)
        else:
            return self._step_farmer(action)

    def close(self):
        self.env.close()
