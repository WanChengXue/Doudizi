import logging

from rosea.envs.base_element import BaseElement
from rosea.feature import sumo_features
from rosea.utils.math import mean, root_mean_square
from rosea.utils.operator import dict_map


class Reward(BaseElement):
    def __init__(self, cfg):
        super(Reward, self).__init__(cfg)
        self.add_feature('action', sumo_features.ActionFeature)
        self.add_feature('queue_length', sumo_features.QueueLengthFeature)
        self.add_feature('occupancy', sumo_features.OccupancyFeature)
        self.add_feature('waiting_time', sumo_features.WaitingTimeFeature, selected_features='accumulated_waiting_time')
        self.add_feature('traffic_volume', sumo_features.TrafficVolumeFeature, selected_features=('accumulated_traffic_volume'))
        self.add_feature('signal', sumo_features.SignalFeature)
        self._logger = logging.getLogger(__name__)

    def produce_feature(self, info=None):
        '''
        info = {
            'action': {
                'cycle': 1,
                'tls': {
                    'htxdj_wjj': 1,
                    'haxl_wjj': 0,
                    'haxl_htxdj': -1,
                }
            },
            'sample_window': 120,
        }
        '''
        feature = dict()
        feature['queue_length'] = self['queue_length']('assembled_normalized_queue_length')
        feature['occupancy'] = self['occupancy']('assembled_occupancy')

        feature['traffic volume'] = self['traffic_volume']('accumulated_traffic_volume')
        feature['waiting time'] = self['waiting_time']('accumulated_waiting_time')

        raw_reward = dict()
        # accumulated reward
        raw_reward["max_queue_length"] = root_mean_square(self._reshaper.squeeze(dict_map(max, feature['queue_length'])))
        raw_reward["mean_queue_length"] = mean(self._reshaper.squeeze(dict_map(mean, feature['queue_length'])))
        raw_reward["max_occupancy"] = root_mean_square(self._reshaper.squeeze(dict_map(max, feature['occupancy'])))
        raw_reward["mean_occupancy"] = mean(self._reshaper.squeeze(dict_map(mean, feature['occupancy'])))
        raw_reward["min_occupancy"] = mean(self._reshaper.squeeze(dict_map(min, feature['occupancy'])))

        raw_reward["average_traffic_volume"] = mean(self._reshaper.squeeze(self._mean_of_time(feature['traffic volume'])))
        raw_reward["wait_time_per_vehicle"] = self['traffic_volume']._per_vehicle(feature['waiting time'], feature['traffic volume'], sum_up=True)

        # instnant reward
        raw_reward["low_green_time_punishment"] = self['signal']("low_green_time_punishment")

        if self['queue_length']._is_flared():
            feature['flared_queue_length'] = self['queue_length']('assembled_normalized_flared_queue_length')
            feature['flared_occupancy'] = self['occupancy']('assembled_flared_occupancy')

            raw_reward["max_flared_queue_length"] = root_mean_square(self._reshaper.squeeze(dict_map(max, feature['flared_queue_length'])))
            raw_reward["mean_flared_queue_length"] = mean(self._reshaper.squeeze(dict_map(mean, feature['flared_queue_length'])))
            raw_reward["max_flared_occupancy"] = root_mean_square(self._reshaper.squeeze(dict_map(max, feature['occupancy'])))
            raw_reward["mean_flared_occupancy"] = mean(self._reshaper.squeeze(dict_map(mean, feature['occupancy'])))
            raw_reward["min_flared_occupancy"] = mean(self._reshaper.squeeze(dict_map(min, feature['occupancy'])))

        if info:
            # action reward
            action = info['action']
            reward_cycle, reward_tls, reward_offset = self['action']('action_punishment', action=action)
            raw_reward["action_cycle"] = reward_cycle
            raw_reward["action_tls"] = reward_tls
        if info is not None:
            self._logger.debug('RAW_REWARD_WITH_INFO: {}, INFO: {}'.format(raw_reward, info))
        else:
            self._logger.debug('RAW_REWARD: {}'.format(raw_reward))
        return raw_reward

    def _movement(self, raw_reward):
        normalized_reward = {
            reward_name: reward_value + self._cfg['reward']['movement_param'][reward_name]
            for reward_name, reward_value in raw_reward.items()
        }
        return normalized_reward

    def produce(self, info):
        '''
        info = { 
            'action': {
                'cycle': 1,
                'tls': {
                    'htxdj_wjj': 1,
                    'haxl_wjj': 0,
                    'haxl_htxdj': -1,
                }
            },
            'sample_window': 120,
        }
        '''
        raw_reward = self.produce_feature(info)
        # !!NORMALIZE WAIT TIME REWARD
        # normalized_reward = {
        #     reward_name: reward_value + self._cfg['reward']['movement_param'][reward_name]
        #     for reward_name, reward_value in raw_reward.items()
        # }
        # return normalized_reward
        normalized_reward = self._movement(raw_reward)
        weighted_reward = self._weight(normalized_reward)

        reward = sum(weighted_reward.values())

        self._logger.debug('NORMALIZED_REWARD: {}'.format(normalized_reward))
        self._logger.debug('REWARD: {}'.format(reward))
        self._logger.debug('WEIGHTED_REWARD: {}'.format(weighted_reward))
        return reward
