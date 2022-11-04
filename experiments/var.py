"""

Script to train and evaluate models

"""


#%% TRAIN VAR MODEL
import argparse
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
import models.dnri.training.train as train
import models.dnri.training.train_utils as train_utils
import models.dnri.training.evaluate as evaluate
from seaborn import heatmap
from data_gen.dataset_class import TimeSeriesDataset
from models.dnri.utils import misc
from models import model_builder
from data_gen.utils import edges_to_graph

# command line option for setting mode and model
parser = argparse.ArgumentParser()
parser.add_argument('--mode')
parser.add_argument('--model_name')
args = parser.parse_args()

# setting model for loading yaml
if args.model_name is not None:
    model_name = args.model_name
else:
    model_name = ['var', 'nri'][0]

with open('./experiment_configs/' + model_name + '.yaml', "r") as stream:
    params = yaml.full_load(stream)

misc.seed(params['seed'])

# setting mode to train or eval
if args.mode is not None:
    params['mode'] = args.mode
else:
    params['mode'] = ['train', 'eval'][0]

# load data
train_data = TimeSeriesDataset(params['data_path'], 'train')
val_data = TimeSeriesDataset(params['data_path'], 'val')
test_data = TimeSeriesDataset(params['data_path'], 'test')
ar_coef = TimeSeriesDataset(params['data_path'], 'test').ar_coef


if params['mode'] == 'train':
    model = model_builder.build_model(params)
    with train_utils.build_writers(params['working_dir']) as (train_writer, val_writer):
        train.train(model, train_data, val_data, params, train_writer, val_writer)

elif params['mode'] == 'eval':

    params['load_best_model'] = True
    best_model = model_builder.build_model(params)

    if params['model_type'] == 'VAR':
        for name, param in best_model.named_parameters():
            print(name, param.data)

        # comparing with the estimated AR matrix
        estimated_ar_coef = best_model.get_parameter('layers.0.weight').detach()

        color_scale = torch.cat([ar_coef.flatten(), estimated_ar_coef.flatten()], 0).abs().max().item()
        v_scale = {'vmin': -color_scale, 'vmax': color_scale}
        heatmap(ar_coef, cmap="RdBu", **v_scale); plt.show(title='true')
        heatmap(estimated_ar_coef, cmap="RdBu", **v_scale); plt.show(title='estimated')

    # comparing predictive performance to that of the Bayes predictor
    with open('./experiment_configs/var.yaml', "r") as stream:
        bayes_model_params = yaml.full_load(stream)

    bayes_model_params['load_best_model'] = False
    bayes_model_params['bias'] = False
    bayes_model = model_builder.build_model(bayes_model_params)

    # setting weights and biases to true value
    bayes_model.layers[0].weight = torch.nn.Parameter(torch.tensor(ar_coef, dtype=torch.float32))

    model_dict = {'best_trained': best_model, 'bayes': bayes_model}
    test_mse_dict = {}
    for name, model in model_dict.items():
        forward_pred = 1
        test_mse = evaluate.eval_forward_prediction(model, test_data, params['test_burn_in_steps'], forward_pred, params)
        test_mse = test_mse[0].item()
        test_mse_dict[name] = test_mse

        print("FORWARD PRED RESULTS ", name, ':')
        print("\t1 STEP: ", test_mse)

    print('mse relative difference', round(100 * (test_mse_dict['best_trained'] - test_mse_dict['bayes'])
                                           / test_mse_dict['bayes'], 1), '%')

    f1, all_acc, acc_0, acc_1, edges, gt_edges = evaluate.eval_edges(best_model, test_data, params)
    print("Val Edge results:")
    print("\tF1: ", f1)
    print("\tAll predicted edge accuracy: ", round(all_acc * 100, 1), '%')
    print("\tEdge 0 Acc: ", round(acc_0 * 100, 1), '%')
    print("\tEdge 1 Acc: ", round(acc_1 * 100, 1), '%')
    print("Num predicted 0 edges: ", (1 - edges).sum().item())
    print("Num predicted 1 edges: ", edges.sum().item())
    print("\tNum 1 edges in each gt graph: ", gt_edges[0, 0].sum().item(), ' out of ', gt_edges[0, 0].numel(), ' total edges')
    if params['num_edge_types'] > 2 or not params['skip_first']:
        print("\tAdjusted accuracy for relabelling: ", max(all_acc, 1 - all_acc))

    # visualising modal true and predicted graph
    gt_edges_mode = gt_edges.reshape(-1, gt_edges.size(-1)).mode(dim=0)[0]
    edges_mode = edges.mode(dim=0)[0]
    print('True graph: \n', edges_to_graph(gt_edges_mode, params['num_vars']))
    print('Predicted graph: \n', edges_to_graph(edges_mode, params['num_vars']))

    # visualising misclassified edges
    gt_edges_static = gt_edges.mode(dim=1)[0]
    misclassification_rate_edges = ((gt_edges_static != edges) + 0.).mean(dim=0)
    print('Av misclassification rate for each edge: \n', np.round(100 * edges_to_graph(misclassification_rate_edges,
                                                                                       params['num_vars'], diag=None), 0))
