# 文件路径: utils/replay_buffer.py (已修正并确认)

from collections import namedtuple
import random

# 【修正1】: 统一并修正 Transition 的定义
# 字段名 'state_pool', 'next_state_pool' 等与主脚本和优化函数中的期望保持一致
Transition = namedtuple('Transition',
                        ('state_pool', 'state_subset', 'action', 'next_state_pool', 'next_state_subset', 'reward'))


class ReplayMemory(object):
    '''
    Class that encapsulates the experience replay buffer, the push and sampling method
    '''

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    # 【修正2】: 修正 push 方法的逻辑
    def push(self, state_pool, state_subset, action, next_state_pool, next_state_subset, reward):
        """Saves a single, complete transition."""
        # 此方法现在正确地接收6个独立的参数

        # 确保内存中有空间，如果满了就从头开始覆盖旧的
        if len(self.memory) < self.capacity:
            self.memory.append(None)

        # 直接使用传入的6个参数创建一个 Transition 对象并存储
        # 之前那个有问题的 for 循环被完全移除了
        self.memory[self.position] = Transition(state_pool, state_subset, action, next_state_pool, next_state_subset,
                                                reward)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)