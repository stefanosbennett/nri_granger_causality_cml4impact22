"""
Functions to generate VAR data based on latent graph

"""

import numpy as np

#%% Generate AR coefficient matrices


def generate_random_ar_coef(N, scale=1e-2):
    ar_coef = np.random.normal(size=(N, N), scale=scale)

    return ar_coef


def generate_constant_ar_coef(N, const=1):
    ar_coef = np.ones((N, N))
    ar_coef *= const

    return ar_coef


def graph_variable_selection(ar_coef, graph):
    return ar_coef * graph


#%% Generate VAR data


def generate_var(T, N=None, ar_coef=None, eps_scale=0.1):

    assert (N is not None) or (ar_coef is not None), 'either N or ar_coef must be non-NA'

    if N is None:
        N = ar_coef.shape[0]
    else:
        if ar_coef is None:
            ar_coef = generate_random_ar_coef(N)
        else:
            assert ar_coef.shape == (N, N), 'ar_coef shape does not match (N, N)'

    eps = np.random.normal(size=(T, N), scale=eps_scale)

    y = np.zeros((T, N))
    # initiate to random starting point
    y[0, :] = eps[[0], :]

    for t in range(1, T):
        y[t, :] = np.matmul(ar_coef, y[t - 1]) + eps[t, :]

    return y

