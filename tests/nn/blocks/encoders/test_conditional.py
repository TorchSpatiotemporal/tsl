import pytest
import torch

from tsl.nn.blocks.encoders import ConditionalEncoder


@pytest.mark.parametrize(
    ('u_shape', 'v_shape', 'emb_shape'),
    [
        ((2, 3, 3), (2, 4), (4, 5)),
        ((2, 3, 4, 3), (2, 4, 4), (2, 4, 5)),
        (None, None, None),
    ],
    ids=['global_features', 'node_features', 'missing_features'],
)
def test_conditional_encoder_broadcasts_optional_features(u_shape, v_shape, emb_shape):
    batch, steps, nodes, input_size, output_size = 2, 3, 4, 2, 7
    encoder = ConditionalEncoder(
        input_size=input_size,
        output_size=output_size,
        exog_size=3,
        static_size=4,
        emb_size=5,
    )
    x = torch.randn(batch, steps, nodes, input_size)
    u = torch.randn(*u_shape) if u_shape is not None else None
    v = torch.randn(*v_shape) if v_shape is not None else None
    emb = torch.randn(*emb_shape) if emb_shape is not None else None

    out = encoder(x, u=u, v=v, emb=emb)

    assert out.shape == (batch, steps, nodes, output_size)


def test_conditional_encoder_accepts_each_optional_feature_independently():
    batch, steps, nodes = 2, 3, 4
    encoder = ConditionalEncoder(
        input_size=2,
        output_size=7,
        exog_size=3,
        static_size=4,
        emb_size=5,
    )
    x = torch.randn(batch, steps, nodes, 2)

    for kwargs in (
        {'u': torch.randn(batch, steps, 3)},
        {'v': torch.randn(batch, nodes, 4)},
        {'emb': torch.randn(nodes, 5)},
    ):
        assert encoder(x, **kwargs).shape == (batch, steps, nodes, 7)


def test_conditional_encoder_rejects_unconfigured_feature_group():
    encoder = ConditionalEncoder(input_size=2, output_size=7)
    x = torch.randn(2, 3, 4, 2)

    with pytest.raises(ValueError, match='configured feature size is zero'):
        encoder(x, u=torch.randn(2, 3, 3))


def test_conditional_encoder_rejects_invalid_feature_layout():
    encoder = ConditionalEncoder(input_size=2, output_size=7, exog_size=3)
    x = torch.randn(2, 3, 4, 2)

    with pytest.raises(ValueError, match='broadcastable'):
        encoder(x, u=torch.randn(2, 3))
