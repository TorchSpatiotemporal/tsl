import numpy as np

from tsl.data import SpatioTemporalDataset
from tsl.data.utils import WINDOW
from tsl.data.preprocessing import RobustScaler
from tsl.datasets import PemsBay

dataset = PemsBay()

data = dataset.numpy()

exogenous = dict(prev_speed=np.roll(data, -1, 0),
                 global_avg_speed=data.mean(1))

adj = dataset.get_connectivity(threshold=0.1, include_self=False, layout='coo')

attributes = dict(nodes_emb=np.random.random((dataset.n_nodes, 16)))

trend = dataset.df.groupby(dataset.index.month).transform(np.mean).values

scalers = dict(data=RobustScaler(0).fit(data).torch())

stds = SpatioTemporalDataset(*dataset.numpy(return_idx=True),
                             mask=dataset.mask,
                             connectivity=adj,
                             exogenous=exogenous,
                             attributes=attributes,
                             scalers=scalers,
                             trend=trend)
print(stds.keys)

stds.update_input_map(avg_speed=(['prev_speed', 'avg_speed'], WINDOW))
print(stds.keys)

stds.set_input_map(x=['data', 'avg_speed'],
                   avg_speed=['avg_speed', 'nodes_emb'])
print(stds.keys)

batch = stds.snapshot(list(range(32)))
