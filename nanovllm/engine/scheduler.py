from collections import deque

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager


class Scheduler:
    """
    Scheduler类负责管理序列的调度，包括等待队列、运行队列、KV缓存块的分配与回收。
    """

    def __init__(self, config: Config):
        # 最大并发序列数
        self.max_num_seqs = config.max_num_seqs
        # 最大批处理token数
        self.max_num_batched_tokens = config.max_num_batched_tokens
        # 终止token id
        self.eos = config.eos
        # KV缓存块管理器
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        # 等待队列
        self.waiting: deque[Sequence] = deque()
        # 运行队列
        self.running: deque[Sequence] = deque()

    def is_finished(self):
        """
        判断所有序列是否都已完成（等待和运行队列均为空）。
        """
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        """
        新增一个序列到等待队列。
        """
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], bool]:
        """
        调度方法，分为prefill和decode两个阶段。
        返回本轮调度的序列列表和是否为prefill阶段。
        """
        # prefill阶段
        scheduled_seqs = []  # 本轮被调度的序列
        num_seqs = 0         # 当前已调度的序列数
        num_batched_tokens = 0  # 当前已调度的token总数

        # 1. prefill阶段：优先从waiting队列中调度新序列
        while self.waiting and num_seqs < self.max_num_seqs:
            seq = self.waiting[0]  # 取队首序列
            # 判断token数和KV块是否足够
            if num_batched_tokens + len(seq) > self.max_num_batched_tokens or not self.block_manager.can_allocate(seq): # can_allocate prefill阶段的检查，用于判断是否有足够的空闲KV缓存块供初始化
                break  # 超出限制则停止调度
            num_seqs += 1
            self.block_manager.allocate(seq)  # 分配KV缓存块
            num_batched_tokens += len(seq) - seq.num_cached_tokens  # 统计本轮新处理的token数，如果某个token在缓存里面就不加入
            seq.status = SequenceStatus.RUNNING  # 状态设为运行中
            self.waiting.popleft()  # 从waiting队列移除
            self.running.append(seq)  # 加入running队列
            scheduled_seqs.append(seq)  # 记录本轮调度的序列

        if scheduled_seqs:
            # 如果有调度的序列，返回并标记为prefill阶段
            return scheduled_seqs, True

        # 2. decode阶段：对已在running队列中的序列做增量生成
        while self.running and num_seqs < self.max_num_seqs:
            seq = self.running.popleft()  # 取出一个运行中的序列
            # 如果KV块不足，抢占其他序列
            # while-else 含义为循环正常结束就会到else中，在这里就是循环到can_append满足了，就进入else中
            # 如果将seq自己抢占了，就直接break，不会进入else的调度逻辑中
            while not self.block_manager.can_append(seq): # can_append decode阶段的检查，用于追加下一个token判断
                if self.running:
                    self.preempt(self.running.pop())  # 抢占并回收最后一个序列的资源
                else:
                    self.preempt(seq)  # 如果只剩当前序列，也要抢占，抢占完了执行break，不会走下面else的调度逻辑
                    break
            else:
                num_seqs += 1
                self.block_manager.may_append(seq)  # 追加token时的block分配
                scheduled_seqs.append(seq)
        assert scheduled_seqs  # 保证本轮至少有一个序列被调度

        # 将本轮调度的序列重新放回运行队列（保持顺序）
        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False  # 返回本轮调度的序列和decode阶段标志

    def preempt(self, seq: Sequence):
        """
        抢占一个序列，将其状态设为WAITING并回收KV块，重新加入等待队列。
        需要注意，这里只是释放了KV缓存块，token_ids没有清空，这会保存已生成的token_ids，然后重新prefill一遍，和完全重新从prompt开始是不同的。
        """
        seq.status = SequenceStatus.WAITING
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int]) -> list[bool]:
        """
        后处理方法，将生成的token追加到序列，并判断是否终止（eos或达到最大token数）。
        如果终止则回收KV块并移出运行队列。
        """
        for seq, token_id in zip(seqs, token_ids):
            seq.append_token(token_id)
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)