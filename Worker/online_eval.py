# 这个函数用来进行在线决策的
import os
import sys
import copy

current_path = os.path.abspath(__file__)
root_path = "/".join(current_path.split("/")[:-2])
sys.path.append(root_path)
from Utils.config import parse_config
from Utils.data_utils import convert_data_format_to_torch_interference

from Worker import agent
from Env.game import EnvCard2RealCard, RealCard2EnvCard
from Env.game import InfoSet
from Env import move_selector, move_detector
from Env.move_generator import MovesGener


class OnlineAgent:
    def __init__(self, config):
        # 载入两个智能体
        self.config = config
        self.TotalCard = [
            5,
            5,
            5,
            5,
            6,
            6,
            6,
            6,
            7,
            7,
            7,
            7,
            8,
            8,
            8,
            8,
            9,
            9,
            9,
            9,
            10,
            10,
            10,
            10,
            11,
            11,
            11,
            11,
            12,
            12,
            12,
            12,
            13,
            13,
            13,
            13,
            14,
            14,
            14,
            14,
            17,
            17,
            17,
            17,
            20,
            30,
        ]

    def _construct_agent(self):
        # 构建landlord和farmer两个智能体
        self.landlord = agent.get_agent(self.config["policy_config"]["landlord"])
        self.farmer = agent.get_agent(self.config["policy_config"]["farmer"])
        # 载入模型的参数
        self.landlord.synchronize_model(
            {"policy": self.config["policy_config"]["landlord"]["model_path"]}
        )
        self.farmer.synchronize_model(
            {"policy": self.config["policy_config"]["farmer"]["model_path"]}
        )

    def _get_legal_card_play_actions(self, player_hand_cards, rival_move):
        mg = MovesGener(player_hand_cards)
        # --- 根据这个rival_move确定当前智能体能够出牌的类型 ---
        rival_type = move_detector.get_move_type(rival_move)
        rival_move_type = rival_type["type"]
        rival_move_len = rival_type.get("len", 1)
        moves = list()
        #
        if rival_move_type == move_detector.TYPE_0_PASS:
            moves = mg.gen_moves()
            # --- 这个gen_moves表示的是所有可能的动作构成的列表 ----

        elif rival_move_type == move_detector.TYPE_1_SINGLE:
            all_moves = mg.gen_type_1_single()
            moves = move_selector.filter_type_1_single(all_moves, rival_move)

        elif rival_move_type == move_detector.TYPE_2_PAIR:
            all_moves = mg.gen_type_2_pair()
            moves = move_selector.filter_type_2_pair(all_moves, rival_move)

        elif rival_move_type == move_detector.TYPE_3_TRIPLE:
            all_moves = mg.gen_type_3_triple()
            moves = move_selector.filter_type_3_triple(all_moves, rival_move)

        elif rival_move_type == move_detector.TYPE_4_BOMB:
            all_moves = mg.gen_type_4_bomb() + mg.gen_type_5_king_bomb()
            moves = move_selector.filter_type_4_bomb(all_moves, rival_move)

        elif rival_move_type == move_detector.TYPE_5_KING_BOMB:
            moves = []

        elif rival_move_type == move_detector.TYPE_6_3_1:
            all_moves = mg.gen_type_6_3_1()
            moves = move_selector.filter_type_6_3_1(all_moves, rival_move)

        elif rival_move_type == move_detector.TYPE_7_3_2:
            all_moves = mg.gen_type_7_3_2()
            moves = move_selector.filter_type_7_3_2(all_moves, rival_move)

        elif rival_move_type == move_detector.TYPE_8_SERIAL_SINGLE:
            all_moves = mg.gen_type_8_serial_single(repeat_num=rival_move_len)
            moves = move_selector.filter_type_8_serial_single(all_moves, rival_move)

        elif rival_move_type == move_detector.TYPE_9_SERIAL_PAIR:
            all_moves = mg.gen_type_9_serial_pair(repeat_num=rival_move_len)
            moves = move_selector.filter_type_9_serial_pair(all_moves, rival_move)

        elif rival_move_type == move_detector.TYPE_10_SERIAL_TRIPLE:
            all_moves = mg.gen_type_10_serial_triple(repeat_num=rival_move_len)
            moves = move_selector.filter_type_10_serial_triple(all_moves, rival_move)

        elif rival_move_type == move_detector.TYPE_11_SERIAL_3_1:
            all_moves = mg.gen_type_11_serial_3_1(repeat_num=rival_move_len)
            moves = move_selector.filter_type_11_serial_3_1(all_moves, rival_move)

        elif rival_move_type == move_detector.TYPE_12_SERIAL_3_2:
            all_moves = mg.gen_type_12_serial_3_2(repeat_num=rival_move_len)
            moves = move_selector.filter_type_12_serial_3_2(all_moves, rival_move)

        elif rival_move_type == move_detector.TYPE_13_4_2:
            all_moves = mg.gen_type_13_4_2()
            moves = move_selector.filter_type_13_4_2(all_moves, rival_move)

        elif rival_move_type == move_detector.TYPE_14_4_22:
            all_moves = mg.gen_type_14_4_22()
            moves = move_selector.filter_type_14_4_22(all_moves, rival_move)

        if rival_move_type not in [
            move_detector.TYPE_0_PASS,
            move_detector.TYPE_4_BOMB,
            move_detector.TYPE_5_KING_BOMB,
        ]:
            moves = moves + mg.gen_type_4_bomb() + mg.gen_type_5_king_bomb()
        if len(rival_move) != 0:  # rival_move is not 'pass'
            moves = moves + [[]]

        for m in moves:
            m.sort()

        return moves

    def _construct_infoset(self, agent_infoset: InfoSet, obs, current_position):
        agent_infoset.player_hand_cards = self._convert_str_to_number(
            obs["self._cards"]
        )
        agent_infoset.last_pid = "landlord"
        # 构建legal_action，首先获取上一个时刻的出牌是什么
        oppo_last_move_card_list = self._convert_str_to_number(obs["oppo_last_move"])
        agent_infoset.legal_actions = self._get_legal_card_play_actions(
            agent_infoset.player_hand_cards, oppo_last_move_card_list
        )
        # 设置炸弹的数量
        agent_infoset.bomb_num = int(obs["bomb_num"])
        # 设置最后一次出牌的动作
        agent_infoset.last_move = oppo_last_move_card_list
        # 将最后两次出牌的动作存放下来
        agent_infoset.last_two_moves = [
            self._convert_str_to_number(obs["history"]["self"][-1]),
            oppo_last_move_card_list,
        ]
        # 记录last move dict &  将卡片剩余数量构成字典
        if current_position == "landlord":
            agent_infoset.last_move_dict = {
                "landlord": self._convert_str_to_number(obs["history"]["self"][-1]),
                "farmer": oppo_last_move_card_list,
            }
            agent_infoset.num_cards_left_dict = {
                "landlord": len(obs["self_cards"]),
                "farmer": int(obs["oppo_left_cards"]),
            }
        else:
            agent_infoset.last_move_dict = {
                "landlord": oppo_last_move_card_list,
                "farmer": self._convert_str_to_number(obs["history"]["self"][-1]),
            }
            agent_infoset.num_cards_left_dict = {
                "landlord": int(obs["oppo_left_cards"]),
                "farmer": len(obs["self_cards"]),
            }
        # ---- 这个是在自己的视角来看，所有没有出过的牌构成的list，做法就是对history进行整合，然后和整幅牌取交 ----
        agent_infoset.other_hand_cards = []
        total_card_bak = copy.deepcopy(self.TotalCard)
        # 移除自己的手牌
        for element in agent_infoset.player_hand_cards:
            total_card_bak.remove(element)
        # 移除对局历史
        for history_list in obs["history"].values():
            for single_list in history_list:
                number_token = self._convert_str_to_number(single_list)
                for element in number_token:
                    total_card_bak.remove(element)
        agent_infoset.other_hand_cards = total_card_bak
        # 还需要card_play_action_seq

    def _generate_landlord_obs(self, obs):
        # 根据obs构建一个Infostate
        landlord_infoset = InfoSet("landlord")
        landlord_infoset.player_hand_cards

    def _generate_farmer_obs(self, obs):
        farmer_infoset = InfoSet("farmer")

    def _get_agent_obs(self, obs):
        # 根据传入的数据构建智能体的obs
        """
        {
            "self_cards": "5,10,5,6,6,6",
            "self_out": "7,8,9,10,J,Q,K,J,X,2,2,2,A,A",
            "oppo_last_move": "",
            "oppo_out": "8,9,10,J,Q,K,A,5,6,7,8,9,7,Q",
            "self_win_card_num": 0,
            "oppo_win_card_num": 1,
            "oppo_left_cards":3,
            "history":["5,7,7,7","8,8,8,6","9,J,J,J","","6,10,Q,Q,Q,K,K,K",""]
        }
        """
        # 首先必须要知道自己的是地主还是农民,提取self_win_card_num，如果大于0就是地主
        position_type = "landlord" if obs["self_win_card_num"] > 0 else "farmer"
        # 构建不同类型玩家的obs
        if position_type == "landlord":
            agent_obs = self._generate_landlord_obs(obs)
            action = self._decision(self.landlord, agent_obs)
        else:
            pass

    def _convert_number_to_str(self, card_list):
        # 传入卡片列表，然后返回实际的牌
        if len(card_list) != 0:
            real_card_list = [EnvCard2RealCard[token] for token in card_list]
            return ",".join(real_card_list)
        else:
            return ""

    def _convert_str_to_number(self, card_str):
        # 接收一个str，然后变成数字列表
        if len(card_str) != 0:
            card_token_list = [RealCard2EnvCard[token] for token in card_str]
            return card_token_list
        else:
            return []

    def _decision(self, agent, agent_obs, legal_action):
        # 将这个agent_obs变成tensor
        tensor_obs = convert_data_format_to_torch_interference(agent_obs)
        action_index = agent.compute_action_eval_mode(tensor_obs)
        # 根据index获取真实动作
        action = legal_action[action_index]
        # 然后将这个action变成实际的动作返回
