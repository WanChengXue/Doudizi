from collections import Counter
import numpy as np

from Env.game import GameEnv
import copy
import random

Card2Column = {5: 0, 6: 1, 7: 2, 8: 3, 9: 4, 10: 5, 11: 6, 12: 7,
                13: 8, 14: 9, 17: 10}

NumOnes2Array = {0: np.array([0, 0, 0, 0]),
                1: np.array([1, 0, 0, 0]),
                2: np.array([1, 1, 0, 0]),
                3: np.array([1, 1, 1, 0]),
                4: np.array([1, 1, 1, 1])}
# --- 去掉了3，4，14表示A，17表示2，20表示小王，30表示大王 --
deck = []
for i in range(5, 15):
    deck.extend([i for _ in range(4)])
deck.extend([17 for _ in range(4)])
deck.extend([20, 30])

class Env:
    """
    Doudizhu multi-agent wrapper
    """
    def __init__(self):
        # Initialize the internal environment
        self._env = GameEnv()
        self.infoset = None
        self.jiaodizhu_dict = {str(1): 2, str(2): 4, str(3): 5, str(4): 6}

    def jiaodizhu_phase(self):
        # 进入叫地主阶段
        self.landlord_index = random.sample([0,1],1)[0]
        # 随机一个叫地主次数出来 
        self.jiaodizhu_turns = random.sample([1,2,3,4], 1)[0]
        self.jiaodizhu_beishu = self.jiaodizhu_dict[str(self.jiaodizhu_turns)]
        self._env.set_rangpaishu(self.jiaodizhu_turns)
        self.rangpaishus = self.jiaodizhu_turns

    def dipai_beishu_stage(self, dipai):
        # --- 判断底牌倍数 -----
        # 4、特殊底牌：当底牌出现顺子、同花、对子、大王、小王、双王、倍数*2
        # 4.1、特殊底牌：当底牌出现同花顺、三条，倍数*3
        if dipai[0] == dipai[1] == dipai[2]:
            self.dipai_beishu = 3
        elif dipai[1] == dipai[0] + 1 and dipai[1] == dipai[2] -1:
            self.dipai_beishu = 3
        else:
            self.dipai_beishu = 1

    def is_chuantian(self):
        # 春天只可能出现在地主端，判断农民是不是17张牌就可以了
        if self._env.get_farmer_card_num == 17:
            return True
        else:
            return False

    def reset(self):
        # init cards，返回一个两个值，第一个是一个二维列表，第二个是底牌
        cards = copy.deepcopy(deck)
        # ---- 随机抽九张牌作为废牌 ----
        feipai = random.sample(cards, 9)
        feipai.sort()
        # ---- 废牌信息后面使用guide oracle的训练方式的时候可能用的到 -----
        for x in feipai:
            cards.remove(x)
        # -- 从剩下的牌中抽三张作为地主牌 ---
        dipai = random.sample(cards, 3)
        dipai.sort()
        self.diapi = copy.deepcopy(dipai)
        self.feipai = copy.deepcopy(feipai)
        for x in dipai:
            cards.remove(x)
        random.shuffle(cards)
        player_cards = [cards[::2], cards[1::2]]
        self.jiaodizhu_phase()
        self.dipai_beishu_stage(dipai)
        landlord_cards = copy.deepcopy(player_cards[0]) + dipai
        card_play_data = {'landlord': landlord_cards,
                            'farmer': player_cards[1],
                            'three_landlord_cards': dipai,
                            'feipai': feipai
                        }
        assert len(card_play_data['landlord']) == 20
        assert len(card_play_data['farmer']) == 17
        for key in card_play_data:
            card_play_data[key].sort()

        # Initialize the cards
        try:
            self._env.card_play_init(card_play_data)
        except :
            print(card_play_data)
        self.infoset = self._game_infoset
        return get_obs(self.infoset)

    def step(self, action):
        assert action in self.infoset.legal_actions
        self._env.step(action)
        self.infoset = self._game_infoset
        done = False
        reward = {'landlord': 0.0, 'farmer':0.0}
        if self._game_over:
            done = True
            reward = self._get_reward()
            obs = None
        else:
            obs = get_obs(self.infoset)
        return obs, reward, done, {}

    def _get_reward(self):
        """
        This function is called in the end of each
        game. It returns either 1/-1 for win/loss,
        or ADP, i.e., every bomb will double the score.
        """
        winner = self._game_winner
        bomb_num = self._game_bomb_num
        if winner == 'landlord':
            if self.is_chuantian():
                landlord_reward =  self.jiaodizhu_beishu * self.dipai_beishu * (2.0**bomb_num) * 2
            else: 
                landlord_reward =  self.jiaodizhu_beishu * self.dipai_beishu * (2.0 ** bomb_num)
            
        else:
            # ----- 农民胜利 ------
            landlord_reward = -self.jiaodizhu_beishu * self.dipai_beishu * (2.0 ** bomb_num)
        reward_dict = {"landlord": landlord_reward, "farmer": -landlord_reward}
        return reward_dict
    @property
    def _game_infoset(self):
        """
        Here, inforset is defined as all the information
        in the current situation, incuding the hand cards
        of all the players, all the historical moves, etc.
        That is, it contains perferfect infomation. Later,
        we will use functions to extract the observable
        information from the views of the three players.
        """
        return self._env.game_infoset

    @property
    def _game_bomb_num(self):
        """
        The number of bombs played so far. This is used as
        a feature of the neural network and is also used to
        calculate ADP.
        """
        return self._env.get_bomb_num()

    @property
    def _game_winner(self):
        """ A string of landlord/peasants
        """
        return self._env.get_winner()

    @property
    def _acting_player_position(self):
        """
        The player that is active. It can be landlord,
        landlod_down, or landlord_up.
        """
        return self._env.acting_player_position

    @property
    def _game_over(self):
        """ Returns a Boolean
        """
        return self._env.game_over

def get_obs(infoset):
    """
    This function obtains observations with imperfect information
    from the infoset. It has three branches since we encode
    different features for different positions.
    
    This function will return dictionary named `obs`. It contains
    several fields. These fields will be used to train the model.
    One can play with those features to improve the performance.

    `position` is a string that can be landlord/landlord_down/landlord_up

    `x_batch` is a batch of features (excluding the hisorical moves).
    It also encodes the action feature

    `z_batch` is a batch of features with hisorical moves only.

    `legal_actions` is the legal moves

    `x_no_action`: the features (exluding the hitorical moves and
    the action features). It does not have the batch dim.

    `z`: same as z_batch but not a batch.
    """
    if infoset.player_position == 'landlord':
        return _get_obs_landlord(infoset)
    elif infoset.player_position == 'farmer':
        return _get_obs_farmer(infoset)
    else:
        raise ValueError('')

def _get_one_hot_array(num_left_cards, max_num_cards):
    """
    A utility function to obtain one-hot endoding
    """
    one_hot = np.zeros(max_num_cards)
    one_hot[num_left_cards - 1] = 1

    return one_hot

def _cards2array(list_cards):
    """
    A utility function that transforms the actions, i.e.,
    A list of integers into card matrix. Here we remove
    the six entries that are always zero and flatten the
    the representations.
    """
    if len(list_cards) == 0:
        return np.zeros(46, dtype=np.int8)

    matrix = np.zeros([4, 11], dtype=np.int8)
    jokers = np.zeros(2, dtype=np.int8)
    counter = Counter(list_cards)
    for card, num_times in counter.items():
        if card < 20:
            matrix[:, Card2Column[card]] = NumOnes2Array[num_times]
        elif card == 20:
            jokers[0] = 1
        elif card == 30:
            jokers[1] = 1
    return np.concatenate((matrix.flatten('F'), jokers))

def _action_seq_list2array(action_seq_list):
    """
    A utility function to encode the historical moves.
    We encode the historical 15 actions. If there is
    no 15 actions, we pad the features with 0. Since
    three moves is a round in DouDizhu, we concatenate
    the representations for each consecutive three moves.
    Finally, we obtain a 5x162 matrix, which will be fed
    into LSTM for encoding.
    """
    action_seq_array = np.zeros((len(action_seq_list), 46))
    for row, list_cards in enumerate(action_seq_list):
        action_seq_array[row, :] = _cards2array(list_cards)
    action_seq_array = action_seq_array.reshape(5, 138)
    return action_seq_array

# 这个位置将历史的15次move进行编码
def _process_action_seq(sequence, length=15):
    """
    A utility function encoding historical moves. We
    encode 15 moves. If there is no 15 moves, we pad
    with zeros.
    """
    sequence = sequence[-length:].copy()
    if len(sequence) < length:
        empty_sequence = [[] for _ in range(length - len(sequence))]
        empty_sequence.extend(sequence)
        sequence = empty_sequence
    return sequence

def _get_one_hot_bomb(bomb_num):
    """
    A utility function to encode the number of bombs
    into one-hot representation.
    """
    one_hot = np.zeros(13)
    one_hot[bomb_num] = 1
    return one_hot

def _get_obs_landlord(infoset):
    num_legal_actions = len(infoset.legal_actions) # 定义合法动作的数量
    my_handcards = _cards2array(infoset.player_hand_cards) # 将自己的手牌变成0-1向量
    my_handcards_batch = np.repeat(my_handcards[np.newaxis, :],
                                num_legal_actions, axis=0)
    # --- 将对手 + 牌山的牌编码成一个0-1向量，然后在action维度进行扩展 ---
    other_handcards = _cards2array(infoset.other_hand_cards)
    other_handcards_batch = np.repeat(other_handcards[np.newaxis, :],
                                    num_legal_actions, axis=0)
    # ---- 最近一次动作，然后编码成0-1向量，在动作维度进行拓展 ---
    last_action = _cards2array(infoset.last_move)
    last_action_batch = np.repeat(last_action[np.newaxis, :],
                                num_legal_actions, axis=0)
    # -- 将所有的合法动作变成0-1向量 --
    my_action_batch = np.zeros(my_handcards_batch.shape)
    for j, action in enumerate(infoset.legal_actions):
        my_action_batch[j, :] = _cards2array(action)

    # --- 将农民手牌数量进行one-hot编码，然后在action维度进行拓展 ---
    farmer_num_cards_left = _get_one_hot_array(
        infoset.num_cards_left_dict['farmer'], 17)
    farmer_num_cards_left_batch = np.repeat(
        farmer_num_cards_left[np.newaxis, :],
        num_legal_actions, axis=0)

    # --- 农民出过的牌进行编码 ---
    farmer_played_cards = _cards2array(infoset.played_cards['farmer'])
    farmer_played_cards_batch = np.repeat(farmer_played_cards[np.newaxis, :], num_legal_actions, axis=0)

    # --- 炸弹数量进行one-hot编码，然后动作维度进行拓展 ----
    bomb_num = _get_one_hot_bomb(infoset.bomb_num)
    bomb_num_batch = np.repeat(bomb_num[np.newaxis, :], num_legal_actions, axis=0)
    # 返回的x_batch是一个维度为legal_action_nums * (concatenate_dim)
    x_batch = np.hstack((my_handcards_batch,
                        other_handcards_batch,
                        last_action_batch,
                        farmer_played_cards_batch,
                        farmer_num_cards_left_batch,
                        bomb_num_batch,
                        my_action_batch))
    # --- 这里返回的就是一个向量了 ---
    x_no_action = np.hstack((my_handcards,
                            other_handcards,
                            last_action,
                            farmer_played_cards,
                            farmer_num_cards_left,
                            bomb_num))

    # 这个z是将过去的15次move进行编码
    z = _action_seq_list2array(_process_action_seq(
        infoset.card_play_action_seq))
    z_batch = np.repeat(
        z[np.newaxis, :, :],
        num_legal_actions, axis=0) # 重复一下，变成action_dim * 5 * 138的矩阵
    obs = {
            'position': 'landlord',
            'x_batch': x_batch.astype(np.float32), # 这个返回的是带有所有可能动作构成的0-1矩阵action_dim * 260
            'z_batch': z_batch.astype(np.float32), # 这个是历史的15次move构成的5*138的0-1矩阵，在action维度上进行了复制
            'legal_actions': infoset.legal_actions, # 这个是所有可能的动作构成的列表
            'x_no_action': x_no_action.astype(np.int8), # 这个是不带所有动作构成的0-1矩阵，action_dim * 214
            'z': z.astype(np.int8), # 这个z表示不进行动作维度扩展的历史移动矩阵
        }
    return obs

def _get_obs_farmer(infoset):
    """
    Obttain the landlord_up features. See Table 5 in
    https://arxiv.org/pdf/2106.06135.pdf
    """
    num_legal_actions = len(infoset.legal_actions)
    my_handcards = _cards2array(infoset.player_hand_cards)
    my_handcards_batch = np.repeat(my_handcards[np.newaxis, :],
                                num_legal_actions, axis=0)

    other_handcards = _cards2array(infoset.other_hand_cards)
    other_handcards_batch = np.repeat(other_handcards[np.newaxis, :],
                                    num_legal_actions, axis=0)

    last_action = _cards2array(infoset.last_move)
    last_action_batch = np.repeat(last_action[np.newaxis, :],
                                num_legal_actions, axis=0)

    my_action_batch = np.zeros(my_handcards_batch.shape)
    for j, action in enumerate(infoset.legal_actions):
        my_action_batch[j, :] = _cards2array(action)

    last_landlord_action = _cards2array(
        infoset.last_move_dict['landlord'])
    last_landlord_action_batch = np.repeat(
        last_landlord_action[np.newaxis, :],
        num_legal_actions, axis=0)
    landlord_num_cards_left = _get_one_hot_array(
        infoset.num_cards_left_dict['landlord'], 20)
    landlord_num_cards_left_batch = np.repeat(
        landlord_num_cards_left[np.newaxis, :],
        num_legal_actions, axis=0)

    landlord_played_cards = _cards2array(
        infoset.played_cards['landlord'])
    landlord_played_cards_batch = np.repeat(
        landlord_played_cards[np.newaxis, :],
        num_legal_actions, axis=0)


    bomb_num = _get_one_hot_bomb(
        infoset.bomb_num)
    bomb_num_batch = np.repeat(
        bomb_num[np.newaxis, :],
        num_legal_actions, axis=0)

    x_batch = np.hstack((my_handcards_batch,
                        other_handcards_batch,
                        landlord_played_cards_batch,
                        last_action_batch,
                        last_landlord_action_batch,
                        landlord_num_cards_left_batch,
                        bomb_num_batch,
                        my_action_batch))

    x_no_action = np.hstack((my_handcards,
                            other_handcards,
                            landlord_played_cards,
                            last_action,
                            last_landlord_action,
                            landlord_num_cards_left,
                            bomb_num))

    z = _action_seq_list2array(_process_action_seq(
        infoset.card_play_action_seq))
    z_batch = np.repeat(
        z[np.newaxis, :, :],
        num_legal_actions, axis=0)
    obs = {
            'position': 'farmer',
            'x_batch': x_batch.astype(np.float32),
            'z_batch': z_batch.astype(np.float32),
            'legal_actions': infoset.legal_actions,
            'x_no_action': x_no_action.astype(np.int8),
            'z': z.astype(np.int8),
        }
    return obs

