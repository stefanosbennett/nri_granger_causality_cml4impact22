"""
Vector autoregressive model

"""

import torch
from torch import nn
import numpy as np


class Baseline(nn.Module):
    """
    Module for implementing baseline multivariate AR models
    """

    def __init__(self, params):
        super(Baseline, self).__init__()
        self.num_vars = params['num_vars']
        self.teacher_forcing_steps = params.get('teacher_forcing_steps', -1)
        self.nll_loss_type = params.get('nll_loss_type', 'crossent')
        self.prior_variance = params.get('prior_variance')
        self.normalize_nll = params.get('normalize_nll', False)
        self.normalize_nll_per_var = params.get('normalize_nll_per_var', False)
        self.anneal_teacher_forcing = params.get('anneal_teacher_forcing', False)
        self.val_teacher_forcing_steps = params.get('val_teacher_forcing_steps', -1)
        self.kl_coef = 0
        self.steps = 0

    def single_step_forward(self, inputs):
        raise NotImplementedError

    def normalize_inputs(self, inputs):
        return inputs

    def calculate_loss(self, inputs, is_train=False, teacher_forcing=True, return_logits=False, use_prior_logits=False,
                       normalized_inputs=None):
        num_time_steps = inputs.size(1)
        all_predictions = []

        if not is_train:
            teacher_forcing_steps = self.val_teacher_forcing_steps
        else:
            teacher_forcing_steps = self.teacher_forcing_steps

        for step in range(num_time_steps - 1):
            if (teacher_forcing and (teacher_forcing_steps == -1 or step < teacher_forcing_steps)) or step == 0:
                current_inputs = inputs[:, :(step + 1)]
            else:
                current_inputs = torch.cat([inputs, torch.stack(all_predictions.unsqueeze(1), dim=1)], dim=1)

            predictions = self.single_step_forward(current_inputs)
            all_predictions.append(predictions)

        all_predictions = torch.stack(all_predictions, dim=1)
        target = inputs[:, 1:, :, :]
        loss_nll = self.nll(all_predictions, target)
        loss_kl = torch.zeros_like(loss_nll)
        loss = loss_nll.mean()
        if return_logits:
            return loss, loss_nll, loss_kl, None, all_predictions
        else:
            return loss, loss_nll, loss_kl

    def predict_future(self, inputs, prediction_steps, return_everything=False):
        burn_in_timesteps = inputs.size(1)
        all_predictions = []
        for step in range(burn_in_timesteps - 1):
            current_inputs = inputs[:, :(step + 1)]
            predictions = self.single_step_forward(current_inputs)
            if return_everything:
                all_predictions.append(predictions)

        current_inputs = inputs[:, :(burn_in_timesteps + 1)]
        for step in range(prediction_steps):
            predictions = self.single_step_forward(current_inputs)
            all_predictions.append(predictions)
            current_inputs = torch.cat([current_inputs, predictions.unsqueeze(1)], dim=1)

        predictions = torch.stack(all_predictions, dim=1)
        return predictions

    def copy_states(self, state):
        if isinstance(state, tuple) or isinstance(state, list):
            current_state = (state[0].clone(), state[1].clone())
        else:
            current_state = state.clone()
        return current_state

    def predict_future_fixedwindow(self, inputs, burn_in_steps, prediction_steps, batch_size):

        for step in range(burn_in_steps - 1):
            current_inputs = inputs[:, :(step + 1)]
            predictions = self.single_step_forward(current_inputs)

        all_timestep_preds = []

        for window_ind in range(burn_in_steps - 1, inputs.size(1) - 1, batch_size):
            current_batch_preds = []

            # predicting on in-sample data
            for step in range(batch_size):
                if window_ind + step >= inputs.size(1):
                    break
                current_inputs = inputs[:, :(window_ind + step + 1)]
                predictions = self.single_step_forward(current_inputs)
                current_batch_preds.append(predictions)

            current_batch_preds = torch.cat(current_batch_preds, 0)
            current_timestep_preds = [current_batch_preds]

            # predicting on out-of-sample data
            current_inputs = torch.cat([current_inputs, predictions.unsqueeze(1)], dim=1)
            for step in range(prediction_steps - 1):
                current_batch_preds = self.single_step_forward(current_inputs)
                current_timestep_preds.append(current_batch_preds)
                current_inputs = torch.cat([current_inputs, predictions.unsqueeze(1)], dim=1)

            all_timestep_preds.append(torch.stack(current_timestep_preds, dim=1))

        results = torch.cat(all_timestep_preds, dim=0)
        return results.unsqueeze(0)

    def nll(self, preds, target):
        if self.nll_loss_type == 'crossent':
            return self.nll_crossent(preds, target)
        elif self.nll_loss_type == 'gaussian':
            return self.nll_gaussian(preds, target)
        elif self.nll_loss_type == 'poisson':
            return self.nll_poisson(preds, target)

    def nll_gaussian(self, preds, target, add_const=False):
        neg_log_p = ((preds - target) ** 2 / (2 * self.prior_variance))
        const = 0.5 * np.log(2 * np.pi * self.prior_variance)
        # neg_log_p += const
        if self.normalize_nll_per_var:
            return neg_log_p.sum() / (target.size(0) * target.size(2))
        elif self.normalize_nll:
            return (neg_log_p.sum(-1) + const).view(preds.size(0), -1).mean(dim=1)
        else:
            return neg_log_p.view(target.size(0), -1).sum() / (target.size(1))

    def nll_crossent(self, preds, target):
        if self.normalize_nll:
            return nn.BCEWithLogitsLoss(reduction='none')(preds, target).view(preds.size(0), -1).mean(dim=1)
        else:
            return nn.BCEWithLogitsLoss(reduction='none')(preds, target).view(preds.size(0), -1).sum(dim=1)

    def nll_poisson(self, preds, target):
        if self.normalize_nll:
            return nn.PoissonNLLLoss(reduction='none')(preds, target).view(preds.size(0), -1).mean(dim=1)
        else:
            return nn.PoissonNLLLoss(reduction='none')(preds, target).view(preds.size(0), -1).sum(dim=1)

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))


class VAR(Baseline):
    def __init__(self, params):
        """
        Vector autoregresive model
        :param num_vars: number of time series
        :param num_lags: order of AR
        """
        super().__init__(params)

        self.num_vars = params['num_vars']
        self.num_lags = params['num_lags']
        self.bias = params['bias']
        self.layers = nn.ModuleList(nn.Linear(self.num_vars, self.num_vars, bias=False) for _ in range(self.num_lags))
        self.bias = nn.Parameter(torch.zeros(self.num_vars)) if self.bias else None

    def single_step_forward(self, inputs):
        """
        :param inputs: [batch, num_timesteps, num_vars, 1]
        :return: size [batch, num_timesteps, num_vars, 1]
        """
        batch_size = inputs.size(0)
        num_timesteps = inputs.size(1)
        num_vars = inputs.size(2)
        num_features = inputs.size(3)

        y = torch.zeros((batch_size, num_vars))

        if num_timesteps < self.num_lags:
            # repeat first tensor to backward fill
            first = inputs[:, [0]]
            num_repeats = self.num_lags - inputs.size(1)
            filler = torch.tile(first, dims=(batch_size, num_repeats, num_vars, num_features))
            inputs = torch.concat([filler, inputs], 1)

        # removes final dim
        inputs = inputs.squeeze(-1)

        for i in range(1, self.num_lags + 1):
            try:
                y += self.layers[i - 1](inputs[:, -i])
            except RuntimeError:
                print('error')

        if self.bias is not None:
            y += self.bias

        # adds final dimension
        y = y.unsqueeze(-1)

        return y
