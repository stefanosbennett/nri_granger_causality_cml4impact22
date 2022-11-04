import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F
import numpy as np

from .model_utils import encode_onehot 

#%% Linear decoder

class LinearDecoder(nn.Module):
    def __init__(self, params):
        super(LinearDecoder, self).__init__()
        self.embedder = None
        self.num_vars = params['num_vars']
        self.input_size = params['input_size']
        self.gpu = params['gpu']
        self.num_edge_types = params['num_edge_types']
        self.skip_first_edge_type = params['skip_first']
        self.dropout_prob = params['decoder_dropout']

        # separate linear map for computing hidden message for each edge type
        self.linear = nn.ModuleList(
            [nn.Linear(self.input_size, self.input_size, bias=False) for _ in range(self.num_edge_types)]
        )

        # linear map for a node's prediction using its own past value
        self.self_linear = nn.Linear(self.input_size, self.input_size, bias=False)

        print('Using learned recurrent interaction net decoder.')

        edges = np.ones(self.num_vars) - np.eye(self.num_vars)
        self.send_edges = np.where(edges)[0]
        self.recv_edges = np.where(edges)[1]

        # edge2node_mat is of size [num_edges, num_vars] and encodes which is the receiving node for each edge
        # it is a convenience matrix for multiplications since it encodes a complete graph
        # the message passing/learned graph structure is encoded through rel_type/sampled_edges as this tells us what edge type we have for each edge
        # in the single_step_forward and forward methods
        self.edge2node_mat = torch.FloatTensor(encode_onehot(self.recv_edges))
        if self.gpu:
            self.edge2node_mat = self.edge2node_mat.cuda(non_blocking=True)

    def single_step_forward(self, inputs, rel_type):
        """
        Computes linear message for each edge using rel_type (the sampled edge type for each edges)
        and the inputs of each variable.
        The messages to each node are aggregated and summed.

        :param inputs: time series input features for each var at the current time.
        :param rel_type: tensor encoding the sampled edge types
        :return: predictions for the next step
        """
        # Inputs: [batch, num_atoms, num_dims]
        # Hidden: [batch, num_atoms, msg_out]
        # rel_type: [batch_size, num_atoms*(num_atoms-1), num_edge_types]

        # pre_msg: [batch, num_edges, 2*msg_out]
        # get the sending features for each edge
        pre_msg = inputs[:, self.send_edges, :]

        if inputs.is_cuda:
            all_msgs = torch.cuda.FloatTensor(pre_msg.size(0), pre_msg.size(1),
                                              self.input_size).fill_(0.)
        else:
            # size [num_batches, num_edges, msg_out]
            all_msgs = torch.zeros(pre_msg.size(0), pre_msg.size(1),
                                   self.input_size)

        if self.skip_first_edge_type:
            start_idx = 1
        else:
            start_idx = 0

        # For each edge type, we take the hidden state of the receiving and sending messages
        # then, we pass them through an MLP to produce a message for each edge
        # We run a separate MLP for every edge type
        # NOTE: to exclude one edge type, simply offset range by 1
        for i in range(start_idx, self.num_edge_types):
            msg = self.linear[i](pre_msg)
            msg = F.dropout(msg, p=self.dropout_prob)

            # this line masks the message for edge type i if the edge in the graph is not of type i
            # rel_type[:, :, i:i+1] is 1 if an edge is of type i and otherwise it is 0
            msg = msg * rel_type[:, :, i:i + 1]
            all_msgs += msg

        # This step sums all the messages to each node
        # all_msgs.transpose(-2, -1) is of size [batch, msg_out, num_edges]
        # self.edge2node_mat is of size [num_edges, num_vars]
        # the multiplication, for each dimension, adds together all the messages sent to each var
        agg_msgs = all_msgs.transpose(-2, -1).matmul(self.edge2node_mat).transpose(-2, -1)
        # agg_msgs is of size [batch, num_vars, num_features]
        agg_msgs = agg_msgs.contiguous()

        # component of prediction due to node's own past values
        self_pred = self.self_linear(inputs).view(inputs.size(0), self.num_vars, -1)

        pred = self_pred + agg_msgs

        return pred

    def forward(self, inputs, sampled_edges, teacher_forcing=False, teacher_forcing_steps=-1, return_state=False,
                prediction_steps=-1, state=None, burn_in_masks=None):

        """
        prediction_steps is the number of steps into the future to predict from the first time row of inputs.
        Uses recursive prediction in general (for prediction_steps > 1) unless teacher_forcing is on,
        in which case, the first teacher_forecasts number of rows are shown as inputs to the decoder

        :param inputs: has shape [batch_size, num_timesteps, num_atoms, num_feats]
        :return preds: has shape [batch_size, prediction_steps, num_vars, num_feats]
        """

        time_steps = inputs.size(1)

        if prediction_steps > 0:
            pred_steps = prediction_steps
        else:
            # if prediction_steps <= -1, then produces forecasts based on all the timesteps in the inputs
            pred_steps = time_steps

        if len(sampled_edges.shape) == 3:
            # reshape to a 4D tensor to account for num_timesteps (at dim 1) where we repeat the sampled_edges across
            # the timesteps.
            sampled_edges = sampled_edges.unsqueeze(1).expand(sampled_edges.size(0), pred_steps, sampled_edges.size(1),
                                                              sampled_edges.size(2))
        # sampled_edges has shape:
        # [batch_size, num_time_steps, num_atoms*(num_atoms-1), num_edge_types]
        # represents the sampled edges in the graph

        if teacher_forcing_steps == -1:
            # uses all the inputs as input to the decoder
            teacher_forcing_steps = inputs.size(1)

        pred_all = []
        for step in range(0, pred_steps):
            if burn_in_masks is not None and step != 0:
                current_masks = burn_in_masks[:, step, :]
                ins = inputs[:, step, :] * current_masks + pred_all[-1] * (1 - current_masks)
            elif step == 0 or (teacher_forcing and step < teacher_forcing_steps):
                # use the last timestep value as the input to the decoder
                ins = inputs[:, step, :]
            else:
                # uses last prediction as the input to the next step
                ins = pred_all[-1]

            edges = sampled_edges[:, step, :]
            pred = self.single_step_forward(ins, edges)

            pred_all.append(pred)

        preds = torch.stack(pred_all, dim=1)

        if return_state:
            return preds, preds[:, -1]
        else:
            return preds


#%% MLPDecoder from ACD repo


class MLPDecoder(nn.Module):
    """Based on https://github.com/ethanfetaya/NRI (MIT License)."""

    def __init__(
        self,
        params,
    ):
        super(MLPDecoder, self).__init__()

        self.num_vars = params['num_vars']
        self.input_size = params['input_size']
        self.num_edge_types = params['num_edge_types']
        self.skip_first_edge_type = params['skip_first']
        # number of hidden units for message passing MLP
        self.msg_hid = params['decoder_hidden']
        self.msg_out = params['decoder_hidden']
        # number of hidden units in output MLP
        self.n_hid = params['decoder_hidden']
        self.dropout_prob = params['decoder_dropout']
        self.gpu = params['gpu']

        self.msg_fc1 = nn.ModuleList(
            [nn.Linear(2 * self.input_size, self.msg_hid) for _ in range(self.num_edge_types)]
        )
        self.msg_fc2 = nn.ModuleList(
            [nn.Linear(self.msg_hid, self.msg_out) for _ in range(self.num_edge_types)]
        )

        self.out_fc1 = nn.Linear(self.input_size + self.msg_out, self.n_hid)
        self.out_fc2 = nn.Linear(self.n_hid, self.n_hid)
        self.out_fc3 = nn.Linear(self.n_hid, self.input_size)

        print("Using learned interaction net decoder.")

        edges = np.ones(self.num_vars) - np.eye(self.num_vars)
        self.send_edges = np.where(edges)[0]
        self.recv_edges = np.where(edges)[1]

        # edge2node_mat is of size [num_edges, num_vars] and encodes which is the receiving node for each edge
        # it is a convenience matrix for multiplications since it encodes a complete graph
        # the message passing/learned graph structure is encoded through rel_type/sampled_edges as this tells us what edge type we have for each edge
        # in the single_step_forward and forward methods
        self.edge2node_mat = torch.FloatTensor(encode_onehot(self.recv_edges))
        if self.gpu:
            self.edge2node_mat = self.edge2node_mat.cuda(non_blocking=True)

    def single_step_forward(self, inputs, rel_type):
        """
        Computes MLP message for each edge using rel_type (the sampled edge type for each edges)
        and the inputs of each variable.
        The messages to each node are aggregated and summed.

        :param inputs: time series input features for each var at the current time.
        :param rel_type: tensor encoding the sampled edge types
        :return: predictions for the next step
        """
        # Inputs: [batch, num_atoms, num_dims]
        # Hidden: [batch, num_atoms, msg_out]
        # rel_type: [batch_size, num_atoms*(num_atoms-1), num_edge_types]

        # pre_msg: [batch, num_edges, 2*num_dims]
        # get the sending features for each edge
        receivers = inputs[:, self.recv_edges, :]
        senders = inputs[:, self.send_edges, :]
        pre_msg = torch.cat([senders, receivers], dim=-1)

        if inputs.is_cuda:
            all_msgs = torch.cuda.FloatTensor(pre_msg.size(0), pre_msg.size(1),
                                              self.msg_out).fill_(0.)
        else:
            # size [num_batches, num_edges, msg_out]
            all_msgs = torch.zeros(pre_msg.size(0), pre_msg.size(1),
                                   self.msg_out)

        if self.skip_first_edge_type:
            start_idx = 1
        else:
            start_idx = 0

        # For each edge type, we take the hidden state of the receiving and sending messages
        # then, we pass them through an MLP to produce a message for each edge
        # We run a separate MLP for every edge type
        # NOTE: to exclude one edge type, simply offset range by 1
        for i in range(start_idx, self.num_edge_types):
            msg = F.relu(self.msg_fc1[i](pre_msg))
            msg = F.dropout(msg, p=self.dropout_prob)
            msg = F.relu(self.msg_fc2[i](msg))
            # this line masks the message for edge type i if the edge in the graph is not of type i
            # rel_type[:, :, i:i+1] is 1 if an edge is of type i and otherwise it is 0
            msg = msg * rel_type[:, :, i:i + 1]
            all_msgs += msg

        # This step sums all the messages to each node
        # all_msgs.transpose(-2, -1) is of size [batch, msg_out, num_edges]
        # self.edge2node_mat is of size [num_edges, num_vars]
        # the multiplication, for each dimension, adds together all the messages sent to each var
        agg_msgs = all_msgs.transpose(-2, -1).matmul(self.edge2node_mat).transpose(-2, -1)
        # agg_msgs is of size [batch, num_vars, num_features]
        agg_msgs = agg_msgs.contiguous()

        # concatenating a node's features with the messages it has received.
        aug_inputs = torch.cat([inputs, agg_msgs], dim=-1)

        # Output MLP
        pred = F.dropout(F.relu(self.out_fc1(aug_inputs)), p=self.dropout_prob)
        pred = F.dropout(F.relu(self.out_fc2(pred)), p=self.dropout_prob)
        pred = self.out_fc3(pred)

        # predict increment
        pred = inputs + pred

        return pred

    def forward(self, inputs, sampled_edges, teacher_forcing=False, teacher_forcing_steps=-1, return_state=False,
                prediction_steps=-1, state=None, burn_in_masks=None):

        """
        prediction_steps is the number of steps into the future to predict from the first time row of inputs.
        Uses recursive prediction in general (for prediction_steps > 1) unless teacher_forcing is on,
        in which case, the first teacher_forecasts number of rows are shown as inputs to the decoder

        :param inputs: has shape [batch_size, num_timesteps, num_atoms, num_feats]
        :return preds: has shape [batch_size, prediction_steps, num_vars, num_feats]
        """

        time_steps = inputs.size(1)

        if prediction_steps > 0:
            pred_steps = prediction_steps
        else:
            # if prediction_steps <= -1, then produces forecasts based on all the timesteps in the inputs
            pred_steps = time_steps

        if len(sampled_edges.shape) == 3:
            # reshape to a 4D tensor to account for num_timesteps (at dim 1) where we repeat the sampled_edges across
            # the timesteps.
            sampled_edges = sampled_edges.unsqueeze(1).expand(sampled_edges.size(0), pred_steps, sampled_edges.size(1),
                                                              sampled_edges.size(2))
        # sampled_edges has shape:
        # [batch_size, num_time_steps, num_atoms*(num_atoms-1), num_edge_types]
        # represents the sampled edges in the graph

        if teacher_forcing_steps == -1:
            # uses all the inputs as input to the decoder
            teacher_forcing_steps = inputs.size(1)

        pred_all = []
        for step in range(0, pred_steps):
            if burn_in_masks is not None and step != 0:
                current_masks = burn_in_masks[:, step, :]
                ins = inputs[:, step, :] * current_masks + pred_all[-1] * (1 - current_masks)
            elif step == 0 or (teacher_forcing and step < teacher_forcing_steps):
                # use the last timestep value as the input to the decoder
                ins = inputs[:, step, :]
            else:
                # uses last prediction as the input to the next step
                ins = pred_all[-1]

            edges = sampled_edges[:, step, :]
            pred = self.single_step_forward(ins, edges)

            pred_all.append(pred)

        preds = torch.stack(pred_all, dim=1)

        if return_state:
            return preds, preds[:, -1]
        else:
            return preds
