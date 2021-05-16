"""
Most of the code in this file is taken from https://gist.github.com/AAnoosheh/b3cce037ae24a86d89505907ec098d62.
Adaptations of the code are performed by Vadym Gryshchuk (vadym.gryshchuk@protonmail.com).

Implementation of Phased LSTM.
"""

import math

import torch
import torch.nn as nn


class PhasedLSTMCell(nn.Module):
    """A phased LSTM cell.
    Paper:
    https://papers.nips.cc/paper/6310-phased-lstm-accelerating-recurrent-network-training-for-long-or-event-based-sequences
    """

    def __init__(
            self,
            hidden_size,
            leak_rate=0.001,
            ratio_on=0.05,
            period_init_min=1.0,
            period_init_max=1000.0
    ):
        """
        :param hidden_size: the number of features in a hidden state.
        :param leak_rate: a scalar. Leak applied only during training to propagate gradient information.
            See page 3, Section 2.
        :param ratio_on: a scalar. Ratio of the period during which the gates are open. See page 3, Section 2.
        :param period_init_min: a scalar (>= 1).  Minimum value of the initialized period.
            The period values are initialized by drawing from the distribution:
            e^U(log(period_init_min), log(period_init_max)), where U(.,.) is the uniform distribution.
        :param period_init_max: a scalar (> period_init_min). Maximum value of the initialized period.
        """
        super().__init__()

        self.hidden_size = hidden_size
        self.ratio_on = ratio_on
        self.leak = leak_rate

        # Initialize period uniformly from exponential space:
        period = torch.exp(
            torch.Tensor(hidden_size).uniform_(
                math.log(period_init_min), math.log(period_init_max)
            )
        )
        self.tau = nn.Parameter(period)

        # Phase shift of the oscillation:
        phase = torch.Tensor(hidden_size).uniform_() * period  # interval [0, tau]. See page 4, Section Results.
        self.phase = nn.Parameter(phase)

    def _compute_phi(self, t):
        """
        Calculates the phase inside the rhythmic cycle. See Eq. 6.
        :param t: Synchronous or asynchronous timestamps. Tensor of shape (batch_size)
        :return: Phase.
        """
        t_ = t.view(-1, 1).repeat(1, self.hidden_size)  # transform to shape (batch_size, num_hidden_units)
        phase_ = self.phase.view(1, -1).repeat(t.shape[0], 1)  # transform to shape (batch_size, num_hidden_units)
        tau_ = self.tau.view(1, -1).repeat(t.shape[0], 1)  # transform to shape (batch_size, num_hidden_units)

        # Eq. 6:
        phi = torch.fmod((t_ - phase_), tau_).detach()
        phi = torch.abs(phi) / tau_
        return phi

    def set_state(self, c, h):
        """
        Set cell and hidden states.
        :param c: a cell state.
        :param h: a hidden state.
        """
        self.c0 = c
        self.h0 = h

    def forward(self, c_s, h_s, t):
        """
        Forward path of a Phased LSTM cell.
        :param c_s: a cell state of shape (1, batch_size, hidden_size).
        :param h_s: a hidden state of shape (1, batch_size, hidden_size).
        :param t: timestamps of shape (batch_size).
        :return: new cell and hidden states.
        """
        phi = self._compute_phi(t)

        # Eq. 6:
        k_up = 2 * phi / self.ratio_on
        k_down = 2 - k_up
        k_closed = self.leak * phi

        k = torch.where(phi < self.ratio_on, k_down, k_closed)
        k = torch.where(phi < 0.5 * self.ratio_on, k_up, k)
        k = k.view(c_s.shape[0], t.shape[0], -1)

        c_s_new = k * c_s + (1 - k) * self.c0
        h_s_new = k * h_s + (1 - k) * self.h0

        return c_s_new, h_s_new


class PhasedLSTM(nn.Module):
    """
    Applies a multi-layer Phased long short-term memory (LSTM) RNN to an input sequence.
    For each element in the input sequence, each layer computes the Eqs. 1 - 5 und 6 - 10 in the paper.
    """

    def __init__(self,
                 input_size,
                 hidden_size,
                 leak_rate,
                 ratio_on,
                 period_init_min,
                 period_init_max,
                 bidirectional=False):
        """

        :param input_size: the number of features in input data.
        :param hidden_size: the number of units in a Phased LSTM.
        :param leak_rate: a scalar. Leak applied only during training to propagate gradient information.
            See page 3, Section 2.
        :param ratio_on: a scalar. Ratio of the period during which the gates are open. See page 3, Section 2.
        :param period_init_min: a scalar (>= 1).  Minimum value of the initialized period.
            The period values are initialized by drawing from the distribution:
            e^U(log(period_init_min), log(period_init_max)), where U(.,.) is the uniform distribution.
        :param period_init_max: a scalar (> period_init_min). Maximum value of the initialized period.
        :param bidirectional: If `True`, becomes a bidirectional Phased LSTM. Default: `False`.
        """
        super().__init__()
        self.name = "plstm"
        self.feature_size = hidden_size
        self.hidden_size = hidden_size

        # 5% of the most active cell neurons are habituated:
        self.top_act_c_neurons = 1 if math.floor(0.05 * hidden_size) < 1 else math.floor(0.05 * hidden_size)

        # Initialize LSTM:
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            bidirectional=bidirectional,
            batch_first=True
        )

        # If bidirectional, scale the number of hidden units by two:
        self.bi = 2 if bidirectional else 1

        self.phased_cell = PhasedLSTMCell(self.bi * hidden_size, leak_rate, ratio_on, period_init_min, period_init_max)

    def forward(self, data):
        """
        A forward path of a Phased LSTM.
        :param inputs: The input sequence data of shape (batch_size, seq_length, n_features).
        :return: Hidden states for each time step.
        """

        inputs, sequences_starts = data
        # Initialize cell and hidden states with zeros:
        c0 = inputs.new_zeros((self.bi, inputs.size(0), self.hidden_size))
        h0 = inputs.new_zeros((self.bi, inputs.size(0), self.hidden_size))
        self.phased_cell.set_state(c0, h0)

        outputs = []  # container for hidden outputs
        # Iterate over time dimension:
        for time_step in range(inputs.size(1)):

            inputs_t = inputs[:, time_step, [0, 1, 3]].unsqueeze(1)  # value(s)
            timestamps_t = inputs[:, time_step, 2]  # timestamps

            _, (h_t, c_t) = self.lstm(inputs_t, (h0, c0))  # get calculated hidden and cell states from LSTM

            mask = (time_step >= sequences_starts).float().unsqueeze(1).expand_as(h_t)  # create a mask
            h_t = h_t * mask + h0 * (1 - mask)  # mask irrelevant input
            c_t = c_t * mask + c0 * (1 - mask)  # mask irrelevant input
            (c_s, h_s) = self.phased_cell(c_t, h_t, timestamps_t)  # calculate new cell and hidden states

            self.phased_cell.set_state(c_s, h_s)  # set a new state of Phased LSTM
            c0, h0 = c_s, h_s  # rearrange references

            outputs.append(torch.transpose(h_s, 0, 1))  # transpose to (batch_size, time_step, hidden_size_dim)

        outputs = torch.cat(outputs, dim=1)  # concatenate all outputs from different time steps
        outputs = torch.squeeze(outputs[:, -1, :], dim=1)  # get the output at the last time step

        return outputs

