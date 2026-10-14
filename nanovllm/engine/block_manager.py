from collections import deque
import xxhash
import numpy as np

from nanovllm.engine.sequence import Sequence


class Block:

    def __init__(self, block_id):
        self.block_id = block_id
        self.ref_count = 0
        self.hash = -1
        self.token_ids = []

    def update(self, hash: int, token_ids: list[int]):
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []


class BlockManager:

    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        self.hash_to_block_id: dict[int, int] = dict()
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        self.used_block_ids: set[int] = set()
        self.cache_hits = 0
        self.cache_evictions = 0
        self.peak_used_blocks = 0

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        h = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))
        h.update(np.asarray(token_ids, dtype=np.int64).tobytes())
        return h.intdigest()

    def _allocate_block(self) -> int:
        block_id = self.free_block_ids.popleft()
        block = self.blocks[block_id]
        assert block.ref_count == 0
        if block.hash != -1 and self.hash_to_block_id.get(block.hash) == block_id:
            del self.hash_to_block_id[block.hash]
            self.cache_evictions += 1
        block.reset()
        self.used_block_ids.add(block_id)
        self.peak_used_blocks = max(self.peak_used_blocks, len(self.used_block_ids))
        return block_id

    def _deallocate_block(self, block_id: int):
        assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    def cached_prefix_blocks(self, seq: Sequence) -> int:
        h = -1
        num_cached_blocks = 0
        for i in range(seq.num_blocks - 1):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block_id = self.hash_to_block_id.get(h, -1)
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                break
            num_cached_blocks += 1
        return num_cached_blocks

    def max_allocatable_tokens(self, seq: Sequence, num_cached_blocks: int = 0) -> int:
        """Token capacity for this sequence without evicting a referenced block."""
        if seq.block_table:
            blocks = len(seq.block_table) + len(self.free_block_ids)
        else:
            h = -1
            shared_blocks = 0
            for i in range(num_cached_blocks):
                h = self.compute_hash(seq.block(i), h)
                shared_blocks += self.hash_to_block_id[h] in self.used_block_ids
            blocks = shared_blocks + len(self.free_block_ids)
        return min(seq.num_tokens, blocks * self.block_size)

    def can_allocate(self, seq: Sequence) -> int:
        """Compatibility check for allocating the full sequence up front."""
        num_cached_blocks = self.cached_prefix_blocks(seq)
        if self.max_allocatable_tokens(seq, num_cached_blocks) < seq.num_tokens:
            return -1
        return num_cached_blocks

    def allocate(self, seq: Sequence, num_cached_blocks: int, target_tokens: int | None = None):
        assert not seq.block_table
        if target_tokens is None:
            target_tokens = seq.num_tokens
        target_blocks = (target_tokens + self.block_size - 1) // self.block_size
        assert num_cached_blocks <= target_blocks <= seq.num_blocks
        h = -1
        for i in range(num_cached_blocks):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block_id = self.hash_to_block_id[h]
            block = self.blocks[block_id]
            if block_id in self.used_block_ids:
                block.ref_count += 1
            else:
                block.ref_count = 1
                self.free_block_ids.remove(block_id)
                self.used_block_ids.add(block_id)
                self.peak_used_blocks = max(self.peak_used_blocks, len(self.used_block_ids))
            self.cache_hits += 1
            seq.block_table.append(block_id)
        for i in range(num_cached_blocks, target_blocks):
            seq.block_table.append(self._allocate_block())
        seq.num_cached_tokens = num_cached_blocks * self.block_size

    def append_blocks(self, seq: Sequence, target_tokens: int):
        target_blocks = (target_tokens + self.block_size - 1) // self.block_size
        assert len(seq.block_table) <= target_blocks <= seq.num_blocks
        assert target_blocks - len(seq.block_table) <= len(self.free_block_ids)
        for _ in range(len(seq.block_table), target_blocks):
            seq.block_table.append(self._allocate_block())

    def deallocate(self, seq: Sequence):
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        seq.num_cached_tokens = 0
        seq.block_table.clear()

    def can_append(self, seq: Sequence) -> bool:
        return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)

    def may_append(self, seq: Sequence):
        if len(seq) % self.block_size == 1:
            seq.block_table.append(self._allocate_block())

    def hash_blocks(self, seq: Sequence):
        start = seq.num_cached_tokens // self.block_size
        end = (seq.num_cached_tokens + seq.num_scheduled_tokens) // self.block_size
        if start == end: return
        h = self.blocks[seq.block_table[start - 1]].hash if start > 0 else -1
        for i in range(start, end):
            block = self.blocks[seq.block_table[i]]
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block.update(h, token_ids)
            self.hash_to_block_id[h] = block.block_id
