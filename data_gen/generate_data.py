"""
Generate and save Vector Autoregressive data

"""

import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data_gen.var_gen_functions import generate_constant_ar_coef, generate_var
from data_gen.utils import graph_to_edges, granger_causality_test, screen

data_path = './data/var'

N = 3
overlap = False

if overlap:
    # length of sequences
    num_time_steps = 200
    # number of training, val and test sequences
    num_train = 1000
    num_val = 1000
    num_test = 1000
    # num_sims is apparent number of simulations for the model
    num_sims = num_train + num_val + num_test
    # using num_time_steps as a burn in
    T = num_time_steps - 1 + num_train + num_val + num_test
else:
    # length of sequences
    num_time_steps = 200
    # number of training, val and test sequences
    num_train = int(1000)
    num_val = int(1000)
    num_test = int(1000)
    # num_sims is apparent number of simulations for the model
    num_sims = num_train + num_val + num_test
    T = num_time_steps * num_sims

ar_mask_list = [
    # 0 edge
    [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    # 1 edge
    [[1, 0, 0], [1, 1, 0], [0, 0, 1]],
    # 2 edge
    [[1, 1, 0], [1, 1, 0], [0, 0, 1]],
    [[1, 0, 0], [1, 1, 0], [0, 1, 1]],
    [[1, 0, 0], [1, 1, 1], [0, 0, 1]],
    # 3 edge
    [[1, 0, 1], [1, 1, 0], [0, 1, 1]],
    [[1, 0, 0], [1, 1, 1], [1, 0, 1]],
    [[1, 1, 0], [1, 1, 1], [0, 0, 1]],
    [[1, 1, 0], [1, 1, 0], [0, 1, 1]],
    # 4 edge
    [[1, 1, 1], [1, 1, 0], [0, 1, 1]],
    [[1, 1, 0], [1, 1, 0], [1, 1, 1]],
    [[1, 1, 1], [1, 1, 1], [0, 0, 1]],
    [[1, 1, 1], [1, 1, 0], [1, 0, 1]],
    # edge 5
    [[1, 1, 1], [1, 1, 1], [0, 1, 1]],
    # edge 6
    [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
]

ar_mask = np.array(ar_mask_list[8])

print('Adjacency mask (graph.T): ', ar_mask)
ar_coef = generate_constant_ar_coef(N, 1)
ar_coef_selected = ar_mask * ar_coef
# renormalising with the max eigenvalue
norm = np.abs(np.linalg.eigvals(ar_coef_selected)).max()
ar_constant = 1/norm * (1 - 0.1)
ar_coef_selected *= ar_constant
eps_scale = 1e0
dataset = generate_var(T=T, ar_coef=ar_coef_selected, eps_scale=eps_scale)
pd.DataFrame(dataset).plot(); plt.show()

# SNR
print('Signal-to-noise ratio: ', np.round(np.std(dataset, axis=0) / eps_scale, 2))

# accuracy of classic Granger causality testing on dataset of length num_time_steps with this ar_coef matrix
granger_causality_test(num_time_steps, ar_coef_selected, ar_mask, eps_scale)

# transposing masking matrix to match graph format used in model embedding
graph = ar_mask.T

# computing auto-covariance tertiary adjustments in posterior probabilities
screen_edges = screen(dataset, ar_constant, graph)
print('screening for difficult edges: ', np.round(screen_edges.sum(), 2), '\n', np.round(screen_edges, 2))

# getting edges matrix from masking matrix used in var generation
# note that in the graph embedding used by the model, G_ij = 1 means that there is edge from node i to node j
edges = graph_to_edges(graph, N)

# converting features to a tensor
dataset = torch.FloatTensor(dataset)
edges = torch.FloatTensor(edges)

# reshaping features to match expected shape for models
if overlap:
    dataset = dataset.unfold(0, num_time_steps, 1).transpose(2, 1).unsqueeze(-1)
else:
    dataset = dataset.unfold(0, num_time_steps, num_time_steps).transpose(2, 1).unsqueeze(-1)

# edges are constant throughout time
all_edges = edges.expand(num_sims * num_time_steps, -1)
all_edges = all_edges.reshape(num_sims, num_time_steps, -1)

train_data = dataset[:num_train]
valid_data = dataset[num_train:(num_train + num_val)]
test_data = dataset[(num_train + num_val):]

# saving time series data
train_path = os.path.join(data_path, 'train_feats')
torch.save(train_data, train_path)
val_path = os.path.join(data_path, 'val_feats')
torch.save(valid_data, val_path)
test_path = os.path.join(data_path, 'test_feats')
torch.save(test_data, test_path)

# saving edge data
train_edges = torch.FloatTensor(all_edges[:num_train])
val_edges = torch.FloatTensor(all_edges[num_train:num_train + num_val])
test_edges = torch.FloatTensor(all_edges[num_train + num_val:])
train_path = os.path.join(data_path, 'train_edges')
torch.save(train_edges, train_path)
val_path = os.path.join(data_path, 'val_edges')
torch.save(val_edges, val_path)
test_path = os.path.join(data_path, 'test_edges')
torch.save(test_edges, test_path)

# saving AR coefficients
ar_coef = torch.FloatTensor(ar_coef_selected)
ar_coef_path = os.path.join(data_path, 'ar_coef')
torch.save(ar_coef, ar_coef_path)
