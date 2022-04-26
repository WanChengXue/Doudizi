import logging

from rosea.envs.base_element import BaseElement
from rosea.feature import sumo_features
from rosea.utils.math import mean
from rosea.utils.operator import dict_map


class Observer(BaseElement):
    def __init__(self, cfg):
        super(Observer, self).__init__(cfg)
        self._feature_list = (
            'max_occupancy',
            'mean_occupancy',
            'min_occupancy',
            'max_flared_occupancy',
            'mean_flared_occupancy',
            'min_flared_occupancy',

            'traffic_volume_ratio',
            'average_traffic_volume',
            'normalized_durations',
        )
        self.add_feature('traffic_volume_ratio', sumo_features.TrafficVolumeRatioFeature, selected_features=('accumulated_traffic_volume', 'traffic_volume_ratio'))
        self.add_feature('signal', sumo_features.SignalFeature)
        self.add_feature('occupancy', sumo_features.OccupancyFeature)

    def produce_feature(self, info: dict = None):
        if info is None:
            info = dict()

        traffic_volume_ratio = self['traffic_volume_ratio']('traffic_volume_ratio', sample_cycle=self._simulation_time)
        occupancy = self['occupancy']('assembled_occupancy')
        features = dict()
        features['max_occupancy'] = self._reshaper.structurize(dict_map(max, occupancy), method='max', )
        features['mean_occupancy'] = self._reshaper.structurize(dict_map(mean, occupancy))
        features['min_occupancy'] = self._reshaper.structurize(dict_map(min, occupancy))

        features['traffic_volume_ratio'] = traffic_volume_ratio
        features['normalized_durations'] = self['signal']('normalized_durations', cycle=True)
        if self['occupancy']._is_flared():
            flared_occupancy = self['occupancy']('assembled_flared_occupancy')
            features['max_flared_occupancy'] = dict_map(max, flared_occupancy)
            features['mean_flared_occupancy'] = dict_map(mean, flared_occupancy)
            features['min_flared_occupancy'] = dict_map(min, flared_occupancy)
        logger = logging.getLogger(__name__)
        logger.debug('OBSERVATION_FEATURES: {}'.format(features))
        return features

    def produce(self, info=None):
        features = self.produce_feature(info=info)
        observation_list = [self._reshaper.squeeze(features[feature_name]) for feature_name in sorted(self._feature_list) if feature_name in features]
        observation = sum(observation_list, list())

        return observation
