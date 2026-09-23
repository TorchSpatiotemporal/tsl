import pytest
import torch

from tsl.nn.functional import (
    gated_tanh,
    scatter_sum,
    sparse_multi_head_attention,
    sparse_softmax,
)


def test_scatter_sum_allocates_output_and_accumulates_values():
    src = torch.tensor([1.0, 2.0, 3.0])
    index = torch.tensor([0, 1, 0])

    out = scatter_sum(src, index, dim=0, dim_size=3)

    assert torch.equal(out, torch.tensor([4.0, 2.0, 0.0]))


def test_gated_tanh_is_torchscript_compiled_and_matches_definition():
    value = torch.tensor([[0.0, 1.0, 2.0, -1.0]])

    out = gated_tanh(value)

    expected = torch.tanh(value[:, :2]) * torch.sigmoid(value[:, 2:])
    assert isinstance(gated_tanh, torch.jit.ScriptFunction)
    assert torch.allclose(out, expected)


@pytest.mark.torch_scatter
def test_sparse_softmax_is_torchscript_compiled_when_loaded():
    src = torch.tensor([1.0, 2.0, 3.0])
    index = torch.tensor([0, 0, 1])

    out = sparse_softmax(src, index, dim=0)

    assert torch.allclose(out, torch.tensor([0.26894143, 0.7310586, 1.0]))
    from tsl.nn import _functional_scatter

    assert isinstance(_functional_scatter.sparse_softmax, torch.jit.ScriptFunction)


@pytest.mark.torch_scatter
def test_sparse_multi_head_attention_returns_normalized_weights():
    q = k = v = torch.ones(3, 2, 1)
    index = torch.tensor([0, 0, 1])

    out, alpha = sparse_multi_head_attention(q, k, v, index)

    assert out.shape == (2, 2, 1)
    assert torch.allclose(alpha[:2], torch.full((2, 2), 0.5))
    assert torch.allclose(alpha[2], torch.ones(2))
