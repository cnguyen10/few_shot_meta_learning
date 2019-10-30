import torch
import numpy as np

import collections

class FCNet(torch.nn.Module):
    def __init__(self,
                dim_input=1,
                dim_output=1,
                num_hidden_units=(100, 100, 100),
                device=torch.device('cpu')
    ):

        super(FC_3_layers, self).__init__()
        self.dim_output = dim_output
        self.dim_input = dim_input
        self.num_hidden_units = num_hidden_units
        self.device = device

    def get_weight_shape(self):
        weight_shape = collections.OrderedDict()

        weight_shape['w1'] = (self.num_hidden_units[0], 1)
        weight_shape['b1'] = weight_shape['w1'][0]

        for i in range(len(self.num_hidden_units) - 1):
            weight_shape['w{0:d}'.format(i + 2)] = (self.num_hidden_units[i + 1], self.num_hidden_units[i])
            weight_shape['b{0:d}'.format(i + 2)] = weight_shape['w{0:d}'.format(i + 2)][0]

        weight_shape['w{0:d}'.format(len(self.num_hidden_units) + 1)] = (self.dim_output, self.num_hidden_units[len(self.num_hidden_units) - 1])
        weight_shape['b{0:d}'.format(len(self.num_hidden_units) + 1)] = self.dim_output

        return weight_shape
    
    def initialise_weights(self):
        w = {}
        weight_shape = self.get_weight_shape()
        for key in weight_shape.keys():
            if 'b' in key:
                w[key] = torch.zeros(weight_shape[key], device=self.device, requires_grad=True)
            else:
                w[key] = torch.empty(weight_shape[key], device=self.device)
                # torch.nn.init.xavier_normal_(tensor=w[key], gain=1.)
                torch.nn.init.kaiming_normal_(tensor=w[key], nonlinearity='relu')
                # torch.nn.init.normal_(tensor=w[key], mean=0., std=1.)
                w[key].requires_grad_()
        return w

    def forward(self, x, w, p_dropout=0):
        out = x

        for i in range(len(self.num_hidden_units) + 1):
            out = torch.nn.functional.linear(
                input=out,
                weight=w['w{0:d}'.format(i + 1)],
                bias=w['b{0:d}'.format(i + 1)]
            )

            if (i < len(self.num_hidden_units)):
                # out = torch.tanh(out)
                out = torch.nn.functional.relu(out)
                if p_dropout > 0:
                    out = torch.nn.functional.dropout(out, p_dropout)
        return out