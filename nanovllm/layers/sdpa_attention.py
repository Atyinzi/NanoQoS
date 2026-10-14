"""Reference attention over the existing paged KV cache.

This backend gathers KV blocks per request. It is for correctness smoke tests,
not a performance substitute for FlashAttention's paged kernels.
"""

import torch
from torch.nn import functional as F


def gather_kv(cache: torch.Tensor, block_table: torch.Tensor, length: int) -> torch.Tensor:
    block_size = cache.shape[1]
    num_blocks = (length + block_size - 1) // block_size
    blocks = cache.index_select(0, block_table[:num_blocks].long())
    return blocks.reshape(-1, *cache.shape[2:])[:length]


def sdpa_prefill(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                 k_cache: torch.Tensor, v_cache: torch.Tensor,
                 cu_seqlens_q: list[int], cu_seqlens_k: list[int],
                 block_tables: torch.Tensor | None, scale: float) -> torch.Tensor:
    outputs = []
    for i in range(len(cu_seqlens_q) - 1):
        q_start, q_end = cu_seqlens_q[i:i + 2]
        key_len = cu_seqlens_k[i + 1] - cu_seqlens_k[i]
        query = q[q_start:q_end].transpose(0, 1).unsqueeze(0)
        if block_tables is None:
            key = k[q_start:q_end]
            value = v[q_start:q_end]
        else:
            key = gather_kv(k_cache, block_tables[i], key_len)
            value = gather_kv(v_cache, block_tables[i], key_len)
        key = key.transpose(0, 1).unsqueeze(0)
        value = value.transpose(0, 1).unsqueeze(0)
        # Non-square is_causal=True uses upper-left alignment. A chunk with
        # cached prefix needs lower-right alignment in absolute token space.
        positions = torch.arange(key_len - (q_end - q_start), key_len,
                                 device=q.device)
        causal = torch.arange(key_len, device=q.device) <= positions[:, None]
        output = F.scaled_dot_product_attention(
            query, key, value, attn_mask=causal, dropout_p=0.0,
            scale=scale, enable_gqa=q.shape[1] != key.shape[1])
        outputs.append(output.squeeze(0).transpose(0, 1))
    return torch.cat(outputs, dim=0)


def sdpa_decode(q: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor,
                context_lens: list[int], block_tables: torch.Tensor,
                scale: float) -> torch.Tensor:
    outputs = []
    for i, key_len in enumerate(context_lens):
        key = gather_kv(k_cache, block_tables[i], key_len)
        value = gather_kv(v_cache, block_tables[i], key_len)
        output = F.scaled_dot_product_attention(
            q[i].unsqueeze(0).unsqueeze(2),
            key.transpose(0, 1).unsqueeze(0),
            value.transpose(0, 1).unsqueeze(0),
            dropout_p=0.0, scale=scale,
            enable_gqa=q.shape[1] != key.shape[1])
        outputs.append(output.transpose(1, 2))
    return torch.cat(outputs, dim=0)
