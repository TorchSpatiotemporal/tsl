import torch

from tsl.nn.blocks.encoders.mlp_attention import MLPAttention
from tsl.nn.functional import (
    gated_tanh,
    scatter_sum,
    sparse_multi_head_attention,
    sparse_softmax,
)
from tsl.nn.layers.base.attention import MultiHeadAttention
from tsl.nn.layers.base.temporal_conv import GatedTemporalConv


def test_scatter_sum_allocates_output_and_accumulates_values():
    src = torch.tensor([1.0, 2.0, 3.0])
    index = torch.tensor([0, 1, 0])

    out = scatter_sum(src, index, dim=0, dim_size=3)

    assert torch.equal(out, torch.tensor([4.0, 2.0, 0.0]))


def test_gated_tanh_matches_definition_and_gradient():
    value = torch.tensor([[0.0, 1.0, 2.0, -1.0]], requires_grad=True)

    out = gated_tanh(value)

    expected = torch.tanh(value[:, :2]) * torch.sigmoid(value[:, 2:])
    assert torch.allclose(out, expected)
    assert torch.autograd.grad(out.sum(), value)[0] is not None


def test_gated_tanh_is_torch_compile_fullgraph_compatible():
    compiled = torch.compile(gated_tanh, backend='eager', fullgraph=True)
    value = torch.randn(2, 3, 8)

    assert torch.allclose(compiled(value, dim=2), gated_tanh(value, dim=2))


def test_gated_temporal_conv_is_torch_compile_fullgraph_compatible():
    layer = GatedTemporalConv(
        input_channels=2,
        output_channels=3,
        kernel_size=2,
        channel_last=True,
    )
    value = torch.randn(2, 5, 4, 2, requires_grad=True)
    expected = layer(value)

    compiled = torch.compile(layer, backend='eager', fullgraph=True)
    actual = compiled(value)

    assert torch.allclose(actual, expected)
    assert torch.autograd.grad(actual.sum(), value)[0] is not None


def test_sparse_softmax_matches_grouped_softmax():
    src = torch.tensor([1.0, 2.0, 3.0])
    index = torch.tensor([0, 0, 1])

    out = sparse_softmax(src, index, dim=0)

    assert torch.allclose(out, torch.tensor([0.26894143, 0.7310586, 1.0]))


def test_sparse_softmax_supports_csr_ptr_and_broadcast_dimensions():
    src = torch.tensor([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]])
    index = torch.tensor([0, 0, 1])
    ptr = torch.tensor([0, 2, 3])

    out = sparse_softmax(src, ptr=ptr, dim=0)
    out_with_both = sparse_softmax(src, index=index, ptr=ptr, dim=0)

    expected = torch.stack(
        [
            torch.softmax(src[:2], dim=0)[0],
            torch.softmax(src[:2], dim=0)[1],
            torch.ones(2),
        ]
    )
    assert torch.allclose(out, expected)
    assert torch.allclose(out_with_both, expected)


def test_sparse_softmax_supports_empty_groups():
    src = torch.empty(0, 2)
    index = torch.empty(0, dtype=torch.long)

    out = sparse_softmax(src, index, dim=0)

    assert out.shape == src.shape


def test_sparse_softmax_is_torch_compile_fullgraph_compatible():
    src = torch.tensor([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]])
    index = torch.tensor([0, 0, 1])
    compiled = torch.compile(sparse_softmax, backend='eager', fullgraph=True)

    actual = compiled(src, index, num_nodes=2, dim=0)

    assert torch.allclose(actual, sparse_softmax(src, index, num_nodes=2, dim=0))


def test_sparse_multi_head_attention_returns_normalized_weights():
    q = torch.ones(3, 2, 1, requires_grad=True)
    k = torch.ones(3, 2, 1, requires_grad=True)
    v = torch.ones(3, 2, 1, requires_grad=True)
    original_v = v.detach().clone()
    index = torch.tensor([0, 0, 1])

    out, alpha = sparse_multi_head_attention(q, k, v, index)

    assert out.shape == (2, 2, 1)
    assert torch.allclose(alpha[:2], torch.full((2, 2), 0.5))
    assert torch.allclose(alpha[2], torch.ones(2))
    assert torch.equal(v, original_v)
    assert all(grad is not None for grad in torch.autograd.grad(out.sum(), (q, k, v)))


def test_sparse_multi_head_attention_is_torch_compile_fullgraph_compatible():
    q = torch.randn(3, 2, 4)
    k = torch.randn(3, 2, 4)
    v = torch.randn(3, 2, 5)
    index = torch.tensor([0, 0, 1])
    compiled = torch.compile(
        sparse_multi_head_attention, backend='eager', fullgraph=True
    )

    actual = compiled(q, k, v, index, dim_size=2)
    expected = sparse_multi_head_attention(q, k, v, index, dim_size=2)

    assert all(torch.allclose(a, e) for a, e in zip(actual, expected))


def test_sparse_softmax_downstream_message_passing_is_compile_compatible():
    layer = MLPAttention(3, 4, reweigh='softmax', norm=False).eval()
    value = torch.randn(4, 3)
    edge_index = torch.tensor([[0, 1, 2, 3, 0], [1, 2, 3, 0, 0]])
    expected = layer(value, edge_index)

    compiled = torch.compile(layer, backend='eager', fullgraph=True)
    actual = compiled(value, edge_index)

    assert torch.allclose(actual, expected)


def test_causal_attention_is_torch_compile_fullgraph_compatible():
    layer = MultiHeadAttention(embed_dim=4, heads=2, causal=True).eval()
    value = torch.randn(2, 3, 5, 4)
    expected = layer(value, need_weights=False)[0]

    compiled = torch.compile(layer, backend='eager', fullgraph=True)
    actual = compiled(value, need_weights=False)[0]

    assert torch.allclose(actual, expected)
