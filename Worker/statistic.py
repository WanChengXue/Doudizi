class StatisticsUtils:
    def __init__(self):
        self.__data__ = {}
        self.list_max_size = 3000

    def set_value(self, key, value):
        self.__data__[key] = value

    def delete(self, key):
        if key in self.__data__:
            del self.__data__[key]

    def incrby(self, key, value):
        if key not in self.__data__:
            self.__data__[key] = value
        else:
            self.__data__[key] += value

    def include(self, key):
        return key in self.__data__.keys()

    def clear(self):
        self.__data__ = {}

    def is_empty(self):
        return len(self.__data__) == 0

    def append(self, key, value):
        if key not in self.__data__:
            self.__data__[key] = [value]
        else:
            self.__data__[key].append(value)
            if len(self.__data__[key]) > self.list_max_size:
                self.__data__[key] = self.__data__[key][1:]

    def get_key_value(self, key):
        if key in self.__data__:
            return self.__data__[key]
        else:
            return 0

    def iter_key_avg_value(self):
        for key in self.__data__:
            yield key, self.get_avg_value(key)

    def get_avg_value(self, key):
        if key not in self.__data__ or len(self.__data__[key]) == 0:
            return 0
        else:
            return sum(self.__data__[key]) / len(self.__data__[key])

    def get_sum_value(self, key):
        if key not in self.__data__ or len(self.__data__[key]) == 1:
            return 0
        else:
            return sum(self.__data__[key])

    def get_max_value(self, key):
        if key not in self.__data__ or len(self.__data__[key]) == 1:
            return 0
        else:
            return max(self.__data__[key])

    def get_latest_avg_value(self, key):
        if key not in self.__data__ or len(self.__data__[key]) == 1:
            return 0
        else:
            # ---------- 这个地方是最近的五个值计算平均数 --------------
            return sum(self.__data__[key][-5:]) / len(self.__data__[key][-5:])
