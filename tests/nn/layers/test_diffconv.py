import torch

from tsl.nn.layers import DiffConv
from tsl.ops.connectivity import convert_torch_connectivity


def test_diff_conv():
    b, t, n, f = 1, 12, 4, 8
    x = torch.randn(b, t, n, f)
    edge_index = torch.tensor([[0, 0, 0, 1, 1, 2, 2, 3],
                               [1, 2, 3, 0, 2, 0, 3, 0]])
    adj_bin = convert_torch_connectivity(edge_index, 'sparse', num_nodes=n)

    edge_weight = torch.rand(edge_index.size(1))
    adj = convert_torch_connectivity((edge_index, edge_weight),
                                     'sparse',
                                     num_nodes=n)

    out_f = 16
    k = 2

    for root_weight in [True, False]:
        for add_backward in [True, False]:
            conv = DiffConv(in_channels=f,
                            out_channels=out_f,
                            k=k,
                            root_weight=root_weight,
                            add_backward=add_backward)

            hidden_f = k * (2 if add_backward else 1)
            hidden_f += 1 if root_weight else 0
            hidden_f *= f
            assert conv.filters.weight.size() == (out_f, hidden_f)

            out1 = conv(x, edge_index)
            assert out1.size() == (b, t, n, out_f)
            assert torch.allclose(conv(x, adj_bin), out1, atol=1e-6)

            out2 = conv(x, edge_index, edge_weight)
            assert out2.size() == (b, t, n, out_f)
            assert torch.allclose(conv(x, adj), out2, atol=1e-6)


def test_normalization():
    b, t, n, f = 1, 12, 4, 1
    x_i = torch.ones(b, t, 1, f)
    x = torch.cat([x_i * i for i in range(n)], dim=-2)

    x_0 = x[:, :, :1]
    x0_neighbors = x[:, :, 1:]
    assert torch.allclose(x_0, torch.zeros_like(x_0))

    edge_index = torch.tensor([[1, 2, 3], [0, 0, 0]])
    adj_bin = convert_torch_connectivity(edge_index, 'sparse', num_nodes=n)

    edge_weight = torch.tensor([5, 3, 2], dtype=torch.float32)
    adj = convert_torch_connectivity((edge_index, edge_weight),
                                     'sparse',
                                     num_nodes=n)

    out_f = 1
    k = 1

    for add_backward in [True, False]:
        conv = DiffConv(in_channels=f,
                        out_channels=out_f,
                        k=k,
                        root_weight=False,
                        bias=False,
                        add_backward=add_backward)
        with torch.no_grad():
            conv.filters.weight.data.fill_(1.)

        hidden_f = k * (2 if add_backward else 1)
        hidden_f *= f
        assert conv.filters.weight.size() == (out_f, hidden_f)

        out1 = conv(x, edge_index)
        assert out1.size() == (b, t, n, out_f)
        assert torch.allclose(conv(x, adj_bin), out1, atol=1e-6)

        out1_0 = out1[:, :, 0]
        x0_neigh_mean = torch.mean(x0_neighbors, dim=-2)
        assert torch.allclose(out1_0, x0_neigh_mean)

        out2 = conv(x, edge_index, edge_weight)
        assert out2.size() == (b, t, n, out_f)
        assert torch.allclose(conv(x, adj), out2, atol=1e-6)

        out2_0 = out2[:, :, 0]
        ew_norm = edge_weight / edge_weight.sum()
        assert torch.allclose(ew_norm, torch.tensor([.5, .3, .2]))

        x0_neigh_weighted_mean = (x0_neighbors * ew_norm[:, None]).sum(-2)
        assert torch.allclose(out2_0, x0_neigh_weighted_mean)
