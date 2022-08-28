import operator
import numpy as np


class SegmentTree(object):
    def __init__(self, capacity, operation, neutral_element):
        assert (
            capacity > 0 and capacity & (capacity - 1) == 0
        ), "capacity must be positive and a power of 2."
        # ----------- 线段树的长度必须是2的指数 -----------
        self._capacity = capacity
        # ---------- 这个value是线段树被构建出来的初值 -------
        self._value = [neutral_element for _ in range(2 * capacity - 1)]
        # ------- 一个完整二叉树的容量为2^{n+1} - 1 -----------------
        self._operation = operation
        # ------- 这个operation可以是sum，可以是min，可以是max，分别表示家和线段树，最小线段树以及最大线段树 -------

    def _reduce_helper(self, start, end, node, node_start, node_end):
        if start == node_start and end == node_end:
            # ----- start, end分别表示的是一个区间，对应于数据的开始和结束 --------
            # ----- 就直接返回value[node],这个node表示的就是这个区间operation的结果 ----
            return self._value[node]
        mid = (node_start + node_end) // 2
        if end <= mid:
            # ---- 如果说这个中间值大于传入的end值，则不需要考虑右边子树了 --------
            return self._reduce_helper(start, end, 2 * node, node_start, mid)
        else:
            if mid + 1 <= start:
                # ------ 如果说这个mid+1 是落在start的左边，则不考虑左边子树了 -----
                return self._reduce_helper(start, end, 2 * node + 1, mid + 1, node_end)
            else:
                # ------- 最后一个种情况，mid落在start和end中间，需要拆分区间 -------
                return self._operation(
                    self._reduce_helper(start, mid, 2 * node, node_start, mid),
                    self._reduce_helper(mid + 1, end, 2 * node + 1, mid + 1, node_end),
                )

    def reduce(self, start=0, end=None):
        if end is None:
            end = self._capacity
        if end < 0:
            end += self._capacity
        end -= 1  # 下标是从零开始的
        # --------- 这个_reduce_helper的参数含义分别是：start, end, node, node_start, node_end
        return self._reduce_helper(start, end, 1, 0, self._capacity - 1)

    def __setitem__(self, idx, val):
        # ----- 这个函数是根据idx设置叶子节点的val ------
        idx += self._capacity
        # ----- 开了2*capacity长度的list，然后传入的idx是在0-(capacity-1)之间，落在线段树上的位置就是idx+capacity,因为0号位置不用 -----
        self._value[idx] = val
        idx //= 2
        while idx >= 1:
            self._value[idx] = self._operation(
                self._value[2 * idx], self._value[2 * idx + 1]
            )
            idx //= 2
        # ------ 这个循环是用来设置所有父亲节点的value ------

    def __getitem__(self, idx):
        assert 0 <= idx < self._capacity
        # -------- 给定index 获取叶子节点上面的值 -------
        return self._value[self._capacity + idx]


class SumSegmentTree(SegmentTree):
    def __init__(self, capacity):
        super(SumSegmentTree, self).__init__(
            capacity=capacity, operation=operator.add, neutral_element=0.0
        )
        # --------- 设置加法线段树 -------

    def sum(self, start=0, end=None):
        """Returns arr[start] + ... + arr[end]"""
        # ----------- 返回整个buffer里面的权重的和 -----------
        return super(SumSegmentTree, self).reduce(start, end)

    def find_prefixsum_idx(self, prefixsum):
        # ------- 这个是给定了一个prefixsum，然后在叶子节点上面寻找一个index
        assert 0 <= prefixsum <= self.sum() + 1e-5
        idx = 1
        while idx < self._capacity:  # while non-leaf
            if self._value[2 * idx] > prefixsum:
                # ------ 如果说是从root开始，第一层左子树之和大于prefixsum，则说明这个样本是在左子树中，然后往下搜索 ----
                idx = 2 * idx
            else:
                # ------ 如果说这个子树在右边，则将这个prefixsum减去左子树所有node的权重和 ----------
                prefixsum -= self._value[2 * idx]
                idx = 2 * idx + 1
        #  ------- 如果到了叶子节点上面，就具体到了某一个样本了，直接返回 --------
        return idx - self._capacity


class MinSegmentTree(SegmentTree):
    def __init__(self, capacity):
        super(MinSegmentTree, self).__init__(
            capacity=capacity, operation=min, neutral_element=float("inf")
        )
        #  ------- 默认将最小线段树上面的值都初始化为inf ------

    def min(self, start=0, end=None):
        """Returns min(arr[start], ...,  arr[end])"""
        # ------- 这个函数是返回整个buffer中的最小值 ------
        return super(MinSegmentTree, self).reduce(start, end)
