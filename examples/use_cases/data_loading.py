import numpy as np

from tsl.data import SpatioTemporalDataset
from tsl.data.preprocessing import StandardScaler
from tsl.datasets import PemsBay

# Let's load the well-known traffic dataset PemsBay as an example
dataset = PemsBay()

# Get tabular data as ndarray with 3 dimensions:
#   time (t)
#   nodes (n)
#   features (f)
data = dataset.numpy()
print(data.shape)

# Spatial relationship can be obtained with different (supported) methods,
# defined in dataset.similarity_options
print(dataset.similarity_options)
adj = dataset.get_connectivity(method='distance', threshold=0.1,
                               include_self=False,  # remove self-loops
                               layout='edge_index')

# We may want to catch and remove a trend in our target data.
# A simple method is to consider the hourly average in the month.
df = dataset.dataframe()
trend = df.groupby([dataset.index.month,
                    dataset.index.hour]).transform(np.mean).values

# We want to enrich the dataset with useful features. We consider
# 1. the average traffic speed in the network
avg_speed = data.mean(1)  # avg_speed.shape = [t f]
# 2. sinusoidal encoding of time (the dataset has hourly sampling rate) as
#   sin and cos of hour in the day
#   sin and cos of hour in the week
time = dataset.datetime_encoded(['day', 'week'])  # time.shape = [t f]
# 3. a 16-sized embedding for each of the sensors in the dataset
nodes_emb = np.random.random((dataset.n_nodes, 16))  # nodes_emb.shape = [n f]

# Borrowing the nomenclature of sklearn, tsl.data.preprocessing contains
# the most used scaler classes. Let's scale our data with a standard scaler.
scaler = StandardScaler(axis=0).fit(data)

# Now we set the windowing parameters. Usually, neural time series processing
# methods handle temporal data with the "sliding window" approach, i.e., a
# sample is made by a fixed-length window on the temporal axis of the dataset.
window = 12  # length of the lookback window
horizon = 1  # length of the forecasting horizon
delay = 1  # number of steps between end of window and start of horizon
stride = 1  # number of steps between a window and the next one

# With all the data defined, let's build our spatiotemporal dataset!
# It will be used in tsl to interact with the neural module.
stds = SpatioTemporalDataset(dataset.target,
                             index=dataset.index,
                             mask=dataset.mask,
                             connectivity=adj,
                             covariates=dict(avg_speed=avg_speed,
                                             time=time,
                                             nodes_emb=nodes_emb),
                             scalers=dict(target=scaler),
                             trend=trend,
                             window=window,
                             horizon=horizon,
                             delay=delay,
                             stride=stride)
print(stds.input_map)

stds.update_input_map(avg_speed=(['prev_speed', 'avg_speed'], 'window'))
print(stds.input_map)

stds.set_input_map(x=['target', 'avg_speed'],
                   avg_speed=['avg_speed', 'nodes_emb'])
print(stds.input_map)

batch = stds[:32]
