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
# 2. sinusoidal encoding of time (the dataset has 5-minutes sampling rate) as
#   sin and cos of time step in the day
#   sin and cos of time step in the week
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

# Understanding INPUT MAP #####################################################

# We can decide how to map all the data we added in the dataset to the "dataset
# item", i.e., the single sample that has to be fed to the model, by the
# "input_map". The default input map follows the standard formulation of
# autoregressive approaches:
print(stds.input_map)

# We can update the map if we want to add or change some key in the item,
# without altering the others. We can map to a single item's key multiple
# dataset's keys. In this case, the data indexed by the different keys will be
# expanded to the minimum common pattern (if any) and concatenated along the
# cat_dim dimension (default is -1) at get time. Thus, data indexed through
# the item's keys are not materialized until they are accessed.
stds.update_input_map(x=(['target', 'avg_speed'], 'window'))
print(stds.input_map)

# If the item's element is time-varying, it must be paired with information
# telling if data should be synchronized with the lookback window (e.g., for the
# target in autoregressive modeling) or the forecasting horizon (e.g., for
# exogenous variables available in advance). The default behaviour for
# time-varying covariates is to be synchronized with the window.

# Let's now set a brand-new mapping for our dataset, in which:
#   * x contains the past traffic measurements paired with the average speed
#   * u contains the temporal information of the forecasting horizon
#   * v contains the node identifiers
stds.set_input_map(x=(['target', 'avg_speed'], 'window'),
                   u=('time', 'horizon'),
                   v='nodes_emb')
# The patterns of the item will thus be:
#   * x : ['target' || 'avg_speed'] = ['t n f' || 't f'] = ['t n f']
#   * u : ['time'] = ['t f']
#   * v : ['nodes_emb'] = ['n f']
print(stds.input_map)

# Understanding SAMPLE & BATCH ################################################

# Get of first item
sample = stds[0]
# Equivalent to
sample = stds.get(0)

# SpatioTemporalDataset supports also slicing. A batch of the first 32 samples
# can be easily built as
batch = stds[:32]
