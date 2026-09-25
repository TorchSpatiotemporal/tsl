import inspect

import pytest
import torch

from tsl.engines import Predictor
from tsl.nn import models
from tsl.nn.models import BaseModel

BATCH, STEPS, NODES, FEATURES, HORIZON = 2, 6, 6, 1, 2


def _inputs():
    x = torch.randn(BATCH, STEPS, NODES, FEATURES)
    mask = torch.rand(BATCH, STEPS, NODES, FEATURES) > 0.25
    u = torch.randn(BATCH, STEPS, NODES, FEATURES)
    nodes = torch.arange(NODES)
    edge_index = torch.stack([nodes, nodes.roll(-1)])
    edge_weight = torch.rand(edge_index.size(1))
    return x, mask, u, edge_index, edge_weight


def _model_case(name):
    x, mask, u, edge_index, edge_weight = _inputs()
    common = dict(input_size=FEATURES, output_size=FEATURES, horizon=HORIZON)
    cases = {
        'AGCRNModel': lambda: (
            models.AGCRNModel(**common, n_nodes=NODES, hidden_size=4, emb_size=2),
            (x,),
        ),
        'ARModel': lambda: (
            models.ARModel(**common, temporal_order=3),
            (x,),
        ),
        'BiRNNImputerModel': lambda: (
            models.BiRNNImputerModel(FEATURES, hidden_size=4),
            (x, mask),
        ),
        'DCRNNModel': lambda: (
            models.DCRNNModel(**common, hidden_size=4, ff_size=4),
            (x, edge_index, edge_weight),
        ),
        'EvolveGCNModel': lambda: (
            models.EvolveGCNModel(
                FEATURES, 4, FEATURES, HORIZON, 0, 1, 'mean', True, False
            ),
            (x, edge_index, edge_weight),
        ),
        'GRINModel': lambda: (
            models.GRINModel(
                FEATURES,
                hidden_size=4,
                ff_size=4,
                embedding_size=2,
                n_nodes=NODES,
                kernel_size=1,
                decoder_order=1,
            ),
            (x, edge_index, edge_weight, mask),
        ),
        'GatedGraphNetworkModel': lambda: (
            models.GatedGraphNetworkModel(
                FEATURES,
                STEPS,
                HORIZON,
                NODES,
                4,
                output_size=FEATURES,
                full_graph=False,
            ),
            (x, edge_index),
        ),
        'GraphWaveNetModel': lambda: (
            models.GraphWaveNetModel(
                **common,
                hidden_size=4,
                ff_size=4,
                n_layers=1,
                learned_adjacency=True,
                n_nodes=NODES,
                norm='none',
                dropout=0,
            ),
            (x, edge_index, edge_weight),
        ),
        'RNNImputerModel': lambda: (
            models.RNNImputerModel(FEATURES, hidden_size=4),
            (x, mask),
        ),
        'SPINHierarchicalModel': lambda: (
            models.SPINHierarchicalModel(
                FEATURES,
                4,
                4,
                NODES,
                exog_size=FEATURES,
                n_layers=1,
                eta=1,
            ),
            (x, u, mask, edge_index),
        ),
        'SPINModel': lambda: (
            models.SPINModel(
                FEATURES,
                4,
                NODES,
                exog_size=FEATURES,
                n_layers=1,
                eta=1,
                temporal_self_attention=False,
            ),
            (x, u, mask, edge_index),
        ),
        'STCNModel': lambda: (
            models.STCNModel(
                FEATURES,
                0,
                4,
                4,
                FEATURES,
                1,
                HORIZON,
                2,
                1,
                temporal_convs_layer=1,
                spatial_convs_layer=1,
            ),
            (x, edge_index, edge_weight),
        ),
        'STIDModel': lambda: (
            models.STIDModel(
                FEATURES,
                NODES,
                STEPS,
                HORIZON,
                n_exog_emb=[],
                hidden_size=4,
                n_layers=1,
                dropout=0,
            ),
            (x,),
        ),
        'TCNModel': lambda: (
            models.TCNModel(
                **common,
                hidden_size=4,
                ff_size=4,
                n_layers=1,
                n_convs_layer=1,
                norm='none',
            ),
            (x,),
        ),
        'TransformerModel': lambda: (
            models.TransformerModel(
                **common, hidden_size=4, ff_size=4, n_heads=1, n_layers=1
            ),
            (x,),
        ),
        'VARModel': lambda: (
            models.VARModel(**common, temporal_order=3, n_nodes=NODES),
            (x,),
        ),
    }
    return cases[name]()


COMPILE_COMPATIBLE_MODELS = {
    'AGCRNModel',
    'ARModel',
    'BiRNNImputerModel',
    'DCRNNModel',
    'EvolveGCNModel',
    'GRINModel',
    'GatedGraphNetworkModel',
    'GraphWaveNetModel',
    'RNNImputerModel',
    'SPINHierarchicalModel',
    'SPINModel',
    'STCNModel',
    'STIDModel',
    'TCNModel',
    'TransformerModel',
    'VARModel',
}

NON_COMPILE_COMPATIBLE_MODELS = {
    'DisjointSTGNN',
    'FCRNNModel',
    'GRUGCNModel',
    'RNNEncGCNDecModel',
    'RNNModel',
    'STGNN',
    'TimeThenSpace',
}


@pytest.fixture(autouse=True)
def _reset_compiler():
    yield
    torch.compiler.reset()


def _assert_output_close(actual, expected):
    actual_flat, actual_spec = torch.utils._pytree.tree_flatten(actual)
    expected_flat, expected_spec = torch.utils._pytree.tree_flatten(expected)
    assert actual_spec == expected_spec
    for actual_value, expected_value in zip(actual_flat, expected_flat):
        if isinstance(actual_value, torch.Tensor):
            torch.testing.assert_close(actual_value, expected_value)
        else:
            assert actual_value == expected_value


@pytest.mark.parametrize('model_name', sorted(COMPILE_COMPATIBLE_MODELS))
def test_model_compile_fullgraph(model_name):
    model, args = _model_case(model_name)
    model.eval()

    with torch.no_grad():
        expected = model(*args)
    model.compile(backend='eager', fullgraph=True)
    with torch.no_grad():
        actual = model(*args)

    _assert_output_close(actual, expected)


def test_all_models_declare_expected_compile_support():
    discovered = {
        name
        for name in models.__all__
        if inspect.isclass(getattr(models, name))
        and issubclass(getattr(models, name), BaseModel)
        and getattr(models, name) is not BaseModel
    }
    assert discovered == (COMPILE_COMPATIBLE_MODELS | NON_COMPILE_COMPATIBLE_MODELS)
    assert all(
        getattr(models, name).can_be_compiled for name in COMPILE_COMPATIBLE_MODELS
    )
    assert not any(
        getattr(models, name).can_be_compiled for name in NON_COMPILE_COMPATIBLE_MODELS
    )


def test_predictor_compile_model_uses_module_compile():
    model, args = _model_case('ARModel')
    predictor = Predictor(model=model)
    expected = predictor.model(*args)

    compiled = predictor.compile_model(backend='eager', fullgraph=True)
    actual = predictor.model(*args)

    assert compiled is predictor.model
    torch.testing.assert_close(actual, expected)


def test_predictor_rejects_model_not_marked_compile_compatible():
    predictor = Predictor(model=models.RNNModel(FEATURES, FEATURES, HORIZON))

    with pytest.raises(RuntimeError, match='not marked as compatible'):
        predictor.compile_model(backend='eager')
