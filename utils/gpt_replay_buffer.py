# Adapted from https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
from collections import namedtuple
import random

EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 10

# ============================================================
# ✅ 修正 Transition 定义：现在与 run_ebm_workflow.py 完全一致
#    明确区分 pool / subset 两种状态流
# ============================================================
Transition = namedtuple('Transition',
                        ('state_pool', 'state_subset', 'action',
                         'next_state_pool', 'next_state_subset', 'reward'))


class ReplayMemory(object):
    '''
    Experience Replay Buffer supporting dual-stream state representation
    (pool and subset embeddings for HAR Active Learning RL).
    保留原有显式内存释放逻辑，防止显存泄漏。
    '''

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    # ============================================================
    # ✅ 修改 push() 签名，使其与 run_ebm_workflow.py 对齐
    #    不再传递整个 current_state 字典，而是显式传入各字段
    # ============================================================
    def push(self, state_pool, state_subset, action,
             next_state_pool, next_state_subset, reward):
        """Saves a dual-stream transition."""
        # 如果 next_state_pool 为空，构造占位符，避免 unpack 错误
        if next_state_pool is None:
            next_state_pool = [None] * state_pool.size()[0]
        if next_state_subset is None:
            next_state_subset = None

        # 注意：这里保持与原实现一致的 for-loop 和 del 逻辑，
        # 确保大 batch 情况下逐元素写入以节省内存。
        for sp, a, nsp, r in zip(state_pool, action, next_state_pool, reward):
            if len(self.memory) < self.capacity:
                self.memory.append(None)

            # 构造 Transition，保证六元组字段对应一致
            self.memory[self.position] = None
            self.memory[self.position] = Transition(
                sp.unsqueeze(0),                         # 当前状态 pool embedding
                state_subset.unsqueeze(0),               # 当前状态 subset embedding
                a,                                       # 动作
                nsp.unsqueeze(0) if nsp is not None else nsp,  # 下一状态 pool embedding
                next_state_subset.unsqueeze(0) if next_state_subset is not None else next_state_subset,
                r                                        # 奖励
            )

            self.position = (self.position + 1) % self.capacity

            # 显式释放局部引用，保持与原逻辑一致
            del (sp)
            del (a)
            del (nsp)
            del (r)

        # 显式释放外层输入引用
        del (state_pool)
        del (state_subset)
        del (action)
        del (next_state_pool)
        del (next_state_subset)
        del (reward)

    def sample(self, batch_size):
        """随机采样一个 batch 的 Transition"""
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
