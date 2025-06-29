import os

from tsl.datasets import PeakWeather

def test_peakweather():
    pass

    # TODO, change for tmp_path fixture of pytest
    # after debugging
    data_path = os.path.join(os.getcwd(), 'data')

    ds = PeakWeather(root=data_path)

    # TODO finish
    edge_index, edge_weight = ds.get_connectivity(threshold=0.7, 
                                                  theta=None, 
                                                  include_self=False, 
                                                  knn=8)