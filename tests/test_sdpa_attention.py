import pytest

torch = pytest.importorskip("torch")
from torch.nn import functional as F

from nanovllm.layers.sdpa_attention import sdpa_decode, sdpa_prefill


def _reference(query, key, value, causal):
    output = F.scaled_dot_product_attention(
        query.transpose(0, 1).unsqueeze(0),
        key.transpose(0, 1).unsqueeze(0),
        value.transpose(0, 1).unsqueeze(0),
        is_causal=causal, enable_gqa=True)
    return output.squeeze(0).transpose(0, 1)


def test_sdpa_prefill_without_prefix_keeps_requests_isolated():
    torch.manual_seed(4)
    q = torch.randn(5, 4, 8)
    k = torch.randn(5, 2, 8)
    v = torch.randn(5, 2, 8)
    empty = torch.empty(0)
    result = sdpa_prefill(q, k, v, empty, empty,
                          [0, 2, 5], [0, 2, 5], None, 8 ** -0.5)
    expected = torch.cat((_reference(q[:2], k[:2], v[:2], True),
                          _reference(q[2:], k[2:], v[2:], True)))
    torch.testing.assert_close(result, expected)


def test_sdpa_chunked_prefill_uses_lower_right_causal_mask():
    torch.manual_seed(5)
    q = torch.randn(5, 4, 8)
    k = torch.randn(5, 2, 8)
    v = torch.randn(5, 2, 8)
    k_cache = torch.zeros(2, 4, 2, 8)
    v_cache = torch.zeros_like(k_cache)
    k_cache[1], v_cache[1] = k[:4], v[:4]
    k_cache[0, 0], v_cache[0, 0] = k[4], v[4]
    table = torch.tensor([[1, 0]], dtype=torch.int32)
    result = sdpa_prefill(q[3:], k[3:], v[3:], k_cache, v_cache,
                          [0, 2], [0, 5], table, 8 ** -0.5)
    expected = _reference(q, k, v, True)[3:]
    torch.testing.assert_close(result, expected)


def test_sdpa_decode_gathers_each_paged_context():
    torch.manual_seed(6)
    q = torch.randn(2, 4, 8)
    keys = [torch.randn(5, 2, 8), torch.randn(3, 2, 8)]
    values = [torch.randn(5, 2, 8), torch.randn(3, 2, 8)]
    k_cache = torch.zeros(3, 4, 2, 8)
    v_cache = torch.zeros_like(k_cache)
    k_cache[2], v_cache[2] = keys[0][:4], values[0][:4]
    k_cache[0, 0], v_cache[0, 0] = keys[0][4], values[0][4]
    k_cache[1, :3], v_cache[1, :3] = keys[1], values[1]
    table = torch.tensor([[2, 0], [1, -1]], dtype=torch.int32)
    result = sdpa_decode(q, k_cache, v_cache, [5, 3], table, 8 ** -0.5)
    expected = torch.stack([
        _reference(q[i:i + 1], keys[i], values[i], False)[0]
        for i in range(2)], dim=0).unsqueeze(1)
    torch.testing.assert_close(result, expected)
