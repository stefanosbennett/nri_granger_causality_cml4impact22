import numpy as np
import statsmodels.api as sm
import pandas as pd
from itertools import permutations, product
from data_gen.var_gen_functions import generate_var


def graph_to_edges(graph, N):
    edges = []
    for sending, receiving in permutations(range(N), 2):
        edge = graph[sending, receiving]
        edges.append(edge)

    return edges


def edges_to_graph(edges, N, diag='fill'):

    graph = np.zeros((N, N))

    if diag == 'fill':
        np.fill_diagonal(graph, 1)

    for edge_id, (sending, receiving) in enumerate(permutations(range(N), 2)):
        edge = edges[edge_id]
        graph[sending, receiving] = edge

    return graph


def granger_causality_test(num_time_steps, ar_coef_selected, ar_mask, eps_scale):
    """
    Computes the accuracy of classic Granger causality testing on dataset of length num_time_steps with this ar_coef matrix

    """
    mc_samples = 100
    gc_acc = []
    N = ar_coef_selected.shape[0]
    for _ in range(mc_samples):
        dataset_mc = generate_var(T=num_time_steps, ar_coef=ar_coef_selected, eps_scale=eps_scale)
        var_model = sm.tsa.VAR(dataset_mc).fit(maxlags=1, method='ols', ic=None, trend='n')
        estimated_graph = (var_model.pvalues < 0.05 / (N * (N - 1))).T + 0
        gc_acc.append((1 - np.abs(ar_mask - estimated_graph).sum() / (N * (N - 1))) * 100)

    gc_acc = np.array(gc_acc)
    print('accuracy: {0}, [{1}, {2}]'.format(round(gc_acc.mean(), 0), round(np.quantile(gc_acc, 0.05), 0),
                                             round(np.quantile(gc_acc, 0.95), 0)), '%')


def lag(time_series):
    """
    Shift the time series forward by one step
    """
    time_series = np.concatenate(([0], time_series[:-1]))

    return time_series


def screen(dataset, ar_constant, graph):
    """
    Check whether an edge is missing in the true graph and the corresponding relative interaction effect size is large
    """

    n = dataset.shape[1]
    strength = np.zeros((n, n))

    for i, j in permutations(range(n), 2):
        interaction_effects = 0

        for k in range(n):
            if k in [i, j]:
                continue
            interaction_effects += ar_constant * (lag(dataset[:, i]) * lag(dataset[:, k])).mean() * graph[k, j]

        main_effects = 2 * (lag(dataset[:, i]) * dataset[:, j]).mean() \
                       - ar_constant * (lag(dataset[:, j]) * lag(dataset[:, i])).mean() \
                       - ar_constant * (lag(dataset[:, i]) * lag(dataset[:, i])).mean()

        strength[i, j] = np.abs(interaction_effects) / (np.abs(interaction_effects) + np.abs(main_effects))

    missing_edge = (graph == 0)
    difficult_edges = missing_edge * strength

    return difficult_edges
