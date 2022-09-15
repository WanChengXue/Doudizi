# 这个函数用来进行在线决策的
import os
import sys
import copy
import argparse
import numpy as np

from flask import Flask, request, jsonify
from flask import Flask
import json

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
from Env.env import get_obs

 
app = Flask(__name__)
app.debug = True
 
parser = argparse.ArgumentParser()
parser.add_argument(
    "--config_path", type=str, default="Config/Testing/DQN_eval_config.yaml"
)
# Independent_D4PG_heterogeneous_network_eval_config
# heterogeneous_network_eval_config
args = parser.parse_args()

 

 
    # 这里指定了地址和端口号。也可以不指定地址填0.0.0.0那么就会使用本机地址ip
class OnlineAgent:
    def __init__(self, config_path):
        # 载入两个智能体
        self.config = parse_config(config_path)
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
        self._construct_agent()

    def _construct_agent(self):
        # 构建landlord和farmer两个智能体
        self.landlord = agent.get_agent(self.config["policy_config"]["agent_name"])(self.config["policy_config"]["agent"]["landlord"])
        self.farmer = agent.get_agent(self.config["policy_config"]["agent_name"])(self.config["policy_config"]["agent"]["farmer"])
        # 载入模型的参数
        self.landlord.synchronize_model(
            {"policy": self.config["policy_config"]["agent"]["landlord"]["policy"]["model_path"]}
        )
        self.farmer.synchronize_model(
            {"policy": self.config["policy_config"]["agent"]["farmer"]["policy"]["model_path"]}
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

    def _convert_list_string_to_number_token(self, list_string):
        token_list = []
        for single_record in list_string:
            token_list.extend(self._convert_str_to_number(single_record))
        return token_list

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
            card_token_list = [RealCard2EnvCard[token] for token in card_str.split(',')]
            return card_token_list
        else:
            return []

    def _filter_rest_card_token_list(self, self_history_token, op_history_token, self_hand):
        card_bak = copy.deepcopy(self.TotalCard)
        history_card = self_history_token + op_history_token + self_hand
        for element in history_card:
            card_bak.remove(element)
        return card_bak


    def _construct_infoset(self, agent_infoset: InfoSet, obs, current_position):
        agent_infoset.player_hand_cards = self._convert_str_to_number(
            obs["self_cards"]
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
        # 记录卡片剩余数量构成字典 & 两个玩家出过的牌构成的历史
        self_history_token = self._convert_str_to_number(obs['self_out'])
        op_history_token = self._convert_str_to_number(obs['oppo_out'])
        if current_position == "landlord":
            agent_infoset.num_cards_left_dict = {
                "landlord": len(obs["self_cards"].split(',')),
                "farmer": int(obs["oppo_left_cards"]),
            }
            agent_infoset.played_cards = {"landlord": self_history_token, "farmer": op_history_token}
            landlord_history = obs['history']['self']
            farmer_history = obs['history']['oppo']
        else:
            agent_infoset.num_cards_left_dict = {
                "landlord": int(obs["oppo_left_cards"]),
                "farmer": len(obs["self_cards"].split(',')),
            }
            agent_infoset.played_cards = {"landlord": op_history_token, "farmer": self_history_token}
            agent_infoset.last_move_dict = {"landlord": oppo_last_move_card_list}
            farmer_history = obs['history']['self']
            landlord_history = obs['history']['oppo']
        # ---- 这个是在自己的视角来看，所有没有出过的牌构成的list，做法就是对history进行整合，然后和整幅牌取交 ----
        agent_infoset.other_hand_cards = self._filter_rest_card_token_list(self_history_token, op_history_token, agent_infoset.player_hand_cards)
        # 还需要card_play_action_seq,这是一个交错的历史
        agent_infoset.card_play_action_seq = self._generate_alter_history(landlord_history, farmer_history, current_position)


    def _generate_alter_history(self, landlord_history, farmer_history, current_position):
        alter_history = []
        if current_position == "landlord":
            assert len(landlord_history) == len(farmer_history)
            for _landlord_history, _farmer_history in zip(landlord_history, farmer_history):
                alter_history.append(self._convert_str_to_number(_landlord_history))
                alter_history.append(self._convert_str_to_number(_farmer_history))
        else:
            assert len(landlord_history) == len(farmer_history) + 1
            for _landlord_history, _farmer_history in zip(landlord_history[:-1], farmer_history):
                alter_history.append(self._convert_str_to_number(_landlord_history))
                alter_history.append(self._convert_str_to_number(_farmer_history))
            alter_history.append(self._convert_str_to_number(landlord_history[-1]))
        return alter_history

    def _modify_obs(self, obs):
        x_batch = np.array(obs["x_batch"])
        z_batch = np.array(obs["z_batch"])
        x_no_action = np.array(obs["x_no_action"])
        z = np.array(obs["z"])
        new_obs = {
            "x_batch": x_batch,
            "z_batch": z_batch,
            "legal_actions": obs["legal_actions"],
        }
        return new_obs

    def _generate_landlord_obs(self, obs):
        # 根据obs构建一个Infostate
        landlord_infoset = InfoSet("landlord")
        self._construct_infoset(landlord_infoset, obs, "landlord")
        return self._modify_obs(get_obs(landlord_infoset))

    def _generate_farmer_obs(self, obs):
        farmer_infoset = InfoSet("farmer")
        self._construct_infoset(farmer_infoset, obs, "farmer")
        return self._modify_obs(get_obs(farmer_infoset))

    def decision(self, obs):
        # 首先必须要知道自己的是地主还是农民,提取self_win_card_num，如果大于0就是地主
        position_type = "landlord" if obs["self_win_card_num"] == 0 else "farmer"
        # 构建不同类型玩家的obs
        if position_type == "landlord":
            agent_obs = self._generate_landlord_obs(obs)
            action = self._decision(self.landlord, agent_obs)
        else:
            agent_obs = self._generate_farmer_obs(obs)
            action = self._decision(self.farmer, agent_obs)
        return action


    def _decision(self, agent, agent_obs):
        # 将这个agent_obs变成tensor
        input_dict = {"x": agent_obs["x_batch"], "z": agent_obs["z_batch"]}
        tensor_obs = convert_data_format_to_torch_interference(input_dict)
        action_index = agent.compute_action_eval_mode(tensor_obs)
        # 根据index获取真实动作
        legal_action = agent_obs['legal_actions']
        action = legal_action[action_index]
        # 然后将这个action变成实际的动作返回
        return self._convert_number_to_str(action)

    def run(self):
        while True:
            # 这个地方开一个http服务，然后每次接受参数，调用网络，返回一个出牌
            pass
        
worker = OnlineAgent(args.config_path)

@app.route('/', methods=['GET', 'POST'])
def post_http():
    if not request.data:  # 检测是否有数据
        return ('fail')
    state_dict = request.data.decode('utf-8')
    # 获取到POST过来的数据，因为我这里传过来的数据需要转换一下编码。根据晶具体情况而定
    prams = json.loads(state_dict)
    print(prams)
    res = worker.decision(prams)
    # 把区获取到的数据转为JSON格式。
    return jsonify(res)


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     "--config_path", type=str, default="Config/Testing/DQN_eval_config.yaml"
    # )
    # # Independent_D4PG_heterogeneous_network_eval_config
    # # heterogeneous_network_eval_config
    # args = parser.parse_args()
    # worker = OnlineAgent(args.config_path)
    # # test lordland
    # test_landlord_string = {
    #     "self_cards": "5,10,5,6,6,6",
    #     "self_out": "7,8,9,10,J,Q,K,J,X,2,2,2,A,A",
    #     "oppo_last_move": "",
    #     "oppo_out": "8,9,10,J,Q,K,A,5,6,7,8,9,7,Q",
    #     "self_win_card_num": 0,
    #     "oppo_win_card_num": 1,
    #     "oppo_left_cards":3,
    #     "bomb_num": 0,
    #     "history":{
    #         "self": ["7,8,9,10,J,Q,K", "", "", 'J','X', '2,2,2,A,A'],
    #         "oppo": ["8,9,10,J,Q,K,A", '5,6,7,8,9', '7', 'Q', "", ""]
    #     }
    #     }
    # # test farmer
    # test_farmer_string = {
    #     "self_cards": "5,10,5",
    #     "self_out": "8,9,10,J,Q,K,A,5,6,7,8,9,7,Q",
    #     "oppo_last_move": "6",
    #     "oppo_out": "7,8,9,10,J,Q,K,J,X,2,2,2,A,A",
    #     "self_win_card_num": 1,
    #     "oppo_win_card_num": 0,
    #     "oppo_left_cards":6,
    #     "bomb_num": 0,
    #     "history":{
    #         "self": ["8,9,10,J,Q,K,A", '5,6,7,8,9', '7', 'Q', "", ""] ,
    #         "oppo": ["7,8,9,10,J,Q,K", "", "", 'J','X', '2,2,2,A,A', '6']
    #     }
    #     }
    # landlord_output = worker.decision(test_landlord_string) 
    # print(landlord_output)

    # farmer_output = worker.decision(test_farmer_string)
    # print(farmer_output)
    app.run(host='172.17.16.17', port=2345)