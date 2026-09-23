import pytest
import torch

from tsl.nn.utils import (
    broadcast,
    expand_then_cat,
    get_functional_activation,
    get_layer_activation,
    maybe_cat_exog,
)


def test_activation_factories():
    assert get_functional_activation() is not None
    assert get_layer_activation() is torch.nn.Identity
    assert get_layer_activation('relu') is torch.nn.ReLU
    with pytest.raises(ValueError, match='not valid'):
        get_functional_activation('unknown')
    with pytest.raises(ValueError, match='not valid'):
        get_layer_activation('unknown')


def test_expand_then_cat_broadcasts_non_concatenated_dimensions():
    left = torch.ones(2, 1, 3)
    right = torch.full((1, 4, 2), 2.0)

    out = expand_then_cat([left, right], dim=-1)

    assert out.shape == (2, 4, 5)
    assert torch.equal(out[..., :3], torch.ones(2, 4, 3))
    assert torch.equal(out[..., 3:], torch.full((2, 4, 2), 2.0))


def test_maybe_cat_exog_handles_global_exogenous_variables():
    x = torch.ones(2, 3, 4, 1)
    u = torch.full((2, 3, 2), 2.0)

    out = maybe_cat_exog(x, u)

    assert out.shape == (2, 3, 4, 3)
    assert torch.equal(out[..., :1], x)
    assert torch.equal(out[..., 1:], torch.full((2, 3, 4, 2), 2.0))
    assert maybe_cat_exog(x, None) is x


def test_broadcast_expands_index_to_match_tensor():
    index = torch.tensor([0, 1, 0])
    values = torch.zeros(2, 3, 4)

    out = broadcast(index, values, dim=1)

    assert out.shape == values.shape
    assert torch.equal(out[0, :, 0], index)
    assert torch.equal(out[1, :, 3], index)
