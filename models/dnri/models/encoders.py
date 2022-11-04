import os
import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F
import numpy as np
from .model_utils import encode_onehot, RefNRIMLP


class BaseEncoder(nn.Module):

    def __init__(self, num_vars, graph_type):
        super(BaseEncoder, self).__init__()
        self.num_vars = num_vars
        self.graph_type = graph_type
        self.dynamic = graph_type == 'dynamic'

        # complete graph with no self loops
        edges = np.ones(num_vars) - np.eye(num_vars)
        # 1D array with the id of a sender node for each edge
        self.send_edges = np.where(edges)[0]
        # 1D array with the id of a receiver node for each edge
        self.recv_edges = np.where(edges)[1]
        # matrix of size (N, num_edges) whose i,j-th element encodes whether node i is the receiving node for edge j
        self.edge2node_mat = nn.Parameter(torch.FloatTensor(encode_onehot(self.recv_edges).transpose()), requires_grad=False)
    
    def node2edge(self, node_embeddings):
        """
        Joins sending and receiving node embeddings to create edge embeddings

        For the static case:
        :param node_embeddings: [batch, num_nodes, embed_size]
        :return edge_embeddings: [batch, num_edges, 2 * embed_size],
        the i-th row of matrix [batch_id, :, :] gives the concatenation of the features of the sender and
        receiver node features for edge i
        """
        if self.dynamic:
            send_embed = node_embeddings[:, self.send_edges, :, :]
            recv_embed = node_embeddings[:, self.recv_edges, :, :]
            return torch.cat([send_embed, recv_embed], dim=3)
        else:
            send_embed = node_embeddings[:, self.send_edges, :]
            recv_embed = node_embeddings[:, self.recv_edges, :]
            return torch.cat([send_embed, recv_embed], dim=2)

    def edge2node(self, edge_embeddings):
        """
        Aggregates the edge embeddings being received by node
        :return the sum (normalised by the total number of nodes) of the embeddings of the sending nodes for a given
        receiving node
        """
        if self.dynamic:
            old_shape = edge_embeddings.shape
            tmp_embeddings = edge_embeddings.view(old_shape[0], old_shape[1], -1)
            incoming = torch.matmul(self.edge2node_mat, tmp_embeddings).view(old_shape[0], -1, old_shape[2], old_shape[3])
        else:
            incoming = torch.matmul(self.edge2node_mat, edge_embeddings)
        return incoming/(self.num_vars-1)

    def forward(self, inputs, state=None, return_state=False):
        raise NotImplementedError


class RefMLPEncoder(BaseEncoder):
    def __init__(self, params):
        num_vars = params['num_vars']
        inp_size = params['input_size'] * params['input_time_steps']
        hidden_size = params['encoder_hidden']
        num_edges = params['num_edge_types']
        self.factor = not params['encoder_no_factor']
        no_bn = False
        graph_type = params['graph_type']
        super(RefMLPEncoder, self).__init__(num_vars, graph_type)
        dropout = params['encoder_dropout']
        self.input_time_steps = params['input_time_steps']
        self.dynamic = (self.graph_type == 'dynamic')

        # mlp1 processes node features
        self.mlp1 = RefNRIMLP(inp_size, hidden_size, hidden_size, dropout, no_bn=no_bn)
        # mlp2 processes edge features
        self.mlp2 = RefNRIMLP(hidden_size * 2, hidden_size, hidden_size, dropout, no_bn=no_bn)
        # mlp3 processes either node or edge features depending on self.factor
        self.mlp3 = RefNRIMLP(hidden_size, hidden_size, hidden_size, dropout, no_bn=no_bn)
        if self.factor:
            self.mlp4 = RefNRIMLP(hidden_size * 3, hidden_size, hidden_size, dropout, no_bn=no_bn)
            print("Using factor graph MLP encoder.")
        else:
            self.mlp4 = RefNRIMLP(hidden_size * 2, hidden_size, hidden_size, dropout, no_bn=no_bn)
            print("Using MLP encoder.")

        # setting number of layers of the final edge MLP
        num_layers = params['encoder_mlp_num_layers']
        if num_layers == 1:
            self.fc_out = nn.Linear(hidden_size, num_edges)
        else:
            tmp_hidden_size = params['encoder_mlp_hidden']
            layers = [nn.Linear(hidden_size, tmp_hidden_size), nn.ELU(inplace=True)]
            for _ in range(num_layers - 2):
                layers.append(nn.Linear(tmp_hidden_size, tmp_hidden_size))
                layers.append(nn.ELU(inplace=True))
            layers.append(nn.Linear(tmp_hidden_size, num_edges))
            self.fc_out = nn.Sequential(*layers)

        self.init_weights()

    def node2edge(self, node_embeddings):
        send_embed = node_embeddings[:, self.send_edges, :]
        recv_embed = node_embeddings[:, self.recv_edges, :]
        return torch.cat([send_embed, recv_embed], dim=2)

    def edge2node(self, edge_embeddings):
        incoming = torch.matmul(self.edge2node_mat, edge_embeddings)
        return incoming/(self.num_vars-1)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)

    def merge_states(self, states):
        return torch.cat(states, dim=0)

    def forward(self, inputs, state=None, return_state=False):
        """
        Takes a MV time series and returns edge encodings
        :param inputs: tensor of shape (num_sims, num_timesteps, num_particles, particle_features)
        :param state: this is prepended to the input and returned, if it's none, it's set to be the input
        :return:
        """
        if inputs.size(1) > self.input_time_steps:
            # select the last self.input_time_steps
            inputs = inputs[:, -self.input_time_steps:]
        elif inputs.size(1) < self.input_time_steps:
            # if the number of time steps is less than self.input_time_steps, then we repeat the
            # first row of the data to make up for it
            begin_inp = inputs[:, 0:1].expand(-1, self.input_time_steps-inputs.size(1), -1, -1)
            inputs = torch.cat([begin_inp, inputs], dim=1)
        if state is not None:
            # prepend state to the inputs
            inputs = torch.cat([state, inputs], 1)[:, -self.input_time_steps:]

        # the features for each node become the features for its last self.input_time_steps values
        x = inputs.transpose(1, 2).contiguous().view(inputs.size(0), inputs.size(2), -1)
        # New shape: [num_sims, num_atoms, num_timesteps*num_dims]

        # process node features
        x = self.mlp1(x)  # 2-layer ELU net per node

        x = self.node2edge(x)
        # mlp2 processes edge features
        x = self.mlp2(x)
        x_skip = x

        if self.factor:
            x = self.edge2node(x)
            # mlp3 processes node features
            x = self.mlp3(x)
            x = self.node2edge(x)
            x = torch.cat((x, x_skip), dim=-1)  # Skip connection
            # mlp4 processes edge features and uses the skip connection
            x = self.mlp4(x)
        else:
            # mlp3 is used to process edge features again
            x = self.mlp3(x)
            x = torch.cat((x, x_skip), dim=-1)  # Skip connection
            # mlp4 processes edge features and uses the skip connection
            x = self.mlp4(x)

        result = self.fc_out(x)
        result_dict = {
            'logits': result,
            'state': inputs,
        }
        return result_dict


class FixedEncoder(BaseEncoder):
    def __init__(self, params):
        num_vars = params['num_vars']
        graph_type = params['graph_type']
        self.input_time_steps = params['input_time_steps']
        super(FixedEncoder, self).__init__(num_vars, graph_type)

        # Child classes should create self.edges and logits

    def edges_to_logits(self):
        prob_edge = torch.cat([1 - self.edges.reshape(-1, 1), self.edges.reshape(-1, 1)], dim=1)
        self.logit = torch.log(prob_edge)

    def forward(self, inputs, state=None, return_state=False):
        """
        Takes a MV time series and returns edge encodings
        :param inputs: tensor of shape (num_sims, num_timesteps, num_particles, particle_features)
        :param state: this is prepended to the input and returned, if it's none, it's set to be the input
        :return:
        """
        if inputs.size(1) > self.input_time_steps:
            # select the last self.input_time_steps
            inputs = inputs[:, -self.input_time_steps:]
        elif inputs.size(1) < self.input_time_steps:
            # if the number of time steps is less than self.input_time_steps, then we repeat the
            # first row of the data to make up for it
            begin_inp = inputs[:, 0:1].expand(-1, self.input_time_steps - inputs.size(1), -1, -1)
            inputs = torch.cat([begin_inp, inputs], dim=1)
        if state is not None:
            # prepend state to the inputs
            inputs = torch.cat([state, inputs], 1)[:, -self.input_time_steps:]

        result = self.logit.unsqueeze(0)
        result = result.expand(inputs.size(0), result.size(1), result.size(2))
        result = result.contiguous()

        result_dict = {
            'logits': result,
            'state': inputs,
        }
        return result_dict


class GroundTruth(FixedEncoder):
    def __init__(self, params):
        super(GroundTruth, self).__init__(params)

        # load ground truth edges
        path = os.path.join(params['data_path'], 'train_edges')
        self.edges = torch.load(path)
        self.edges = self.edges[0, 0]

        # convert to logits
        self.edges_to_logits()


class CompleteGraph(FixedEncoder):
    def __init__(self, params):
        super(CompleteGraph, self).__init__(params)

        # load ground truth edges
        self.edges = torch.ones(self.num_vars * (self.num_vars - 1))

        # convert to logits
        self.edges_to_logits()


class EmptyGraph(FixedEncoder):
    def __init__(self, params):
        super(EmptyGraph, self).__init__(params)

        # load ground truth edges
        self.edges = torch.zeros(self.num_vars * (self.num_vars - 1))

        # convert to logits
        self.edges_to_logits()


class UnsharedEncoder(BaseEncoder):
    def __init__(self, params):
        """
        Each edge gets a separate probability
        """
        num_vars = params['num_vars']
        graph_type = params['graph_type']
        num_edge_types = params['num_edge_types']
        self.input_time_steps = params['input_time_steps']
        super(UnsharedEncoder, self).__init__(num_vars, graph_type)
        num_edges = num_vars * (num_vars - 1)
        # unnormalised logits
        self.edge_logit = nn.Parameter(-1.e-1 * torch.ones((num_edges, num_edge_types)))

    def forward(self, inputs, state=None, return_state=False):
        """
        Takes a MV time series and returns edge encodings
        :param inputs: tensor of shape (num_sims, num_timesteps, num_particles, particle_features)
        :param state: this is prepended to the input and returned, if it's none, it's set to be the input
        :return:
        """
        if inputs.size(1) > self.input_time_steps:
            # select the last self.input_time_steps
            inputs = inputs[:, -self.input_time_steps:]
        elif inputs.size(1) < self.input_time_steps:
            # if the number of time steps is less than self.input_time_steps, then we repeat the
            # first row of the data to make up for it
            begin_inp = inputs[:, 0:1].expand(-1, self.input_time_steps - inputs.size(1), -1, -1)
            inputs = torch.cat([begin_inp, inputs], dim=1)
        if state is not None:
            # prepend state to the inputs
            inputs = torch.cat([state, inputs], 1)[:, -self.input_time_steps:]

        result = self.edge_logit.unsqueeze(0)
        result = result.expand(inputs.size(0), result.size(1), result.size(2))
        result = result.contiguous()

        result_dict = {
            'logits': result,
            'state': inputs,
        }
        return result_dict
