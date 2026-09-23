import pytest
import torch

from tsl.data import Data
from tsl.transforms import (
    MaskedSubgraph,
    MaskInput,
    NodeThenTime,
    Rearrange,
    SubgraphTransform,
)


def test_mask_input_masks_values_and_is_instantiable():
    data = Data(
        input={'x': torch.tensor([[1.0], [2.0]])},
        mask=torch.tensor([[True], [False]]),
    )

    out = MaskInput()(data)

    assert out is not data
    assert torch.equal(data.input['x'], torch.tensor([[1.0], [2.0]]))
    assert torch.equal(out.x, torch.tensor([[1.0], [0.0]]))


def test_rearrange_applies_declared_patterns():
    data = Data(input={'x': torch.arange(6).reshape(2, 3)}, pattern={'x': 't n'})

    out = Rearrange({'x': 'n t'})(data)

    assert out is not data
    assert data.input['x'].shape == (2, 3)
    assert out.x.shape == (3, 2)
    assert torch.equal(out.x, torch.tensor([[0, 3], [1, 4], [2, 5]]))
    assert out.pattern['x'] == 'n t'


def test_node_then_time_reorders_time_node_data():
    data = Data(input={'x': torch.arange(6).reshape(2, 3)}, pattern={'x': 't n'})

    out = NodeThenTime({'x': 't n'})(data)

    assert out is not data
    assert out.pattern['x'] == 'n t'
    assert torch.equal(out.x, torch.tensor([[0, 3], [1, 4], [2, 5]]))


def test_masked_subgraph_accepts_explicit_node_mask_and_preserves_indices():
    data = Data(
        input={'x': torch.ones(1, 3, 1)},
        edge_index=torch.tensor([[0, 1, 2], [1, 2, 0]]),
        pattern={'x': 't n f', 'edge_index': '2 e'},
    )

    out = SubgraphTransform(torch.tensor([True, False, True]), add_node_idx=True)(data)

    assert out is data
    assert out.node_idx.tolist() == [0, 2]
    assert out.pattern['node_idx'] == 'n'


def test_subgraph_transform_reduces_node_data_without_connectivity():
    data = Data(input={'x': torch.arange(6).reshape(1, 3, 2)}, pattern={'x': 't n f'})

    out = SubgraphTransform(torch.tensor([True, False, True]))(data)

    assert out is data
    assert out.edge_index is None
    assert out.x.shape == (1, 2, 2)
    assert torch.equal(out.x, torch.tensor([[[0, 1], [4, 5]]]))


def test_masked_subgraph_warns_and_derives_nodes_from_data_mask():
    data = Data(
        input={'x': torch.arange(3).reshape(1, 3, 1)},
        edge_index=torch.tensor([[0, 1, 2], [1, 2, 0]]),
        mask=torch.tensor([[[True], [False], [True]]]),
        pattern={'x': 't n f', 'mask': 't n f', 'edge_index': '2 e'},
    )

    with pytest.deprecated_call(match='MaskedSubgraph is deprecated'):
        out = MaskedSubgraph()(data)

    assert out is data
    assert out.x.shape == (1, 2, 1)
    assert torch.equal(out.mask, torch.tensor([[[True], [True]]]))
    assert torch.equal(out.edge_index, torch.tensor([[1], [0]]))
