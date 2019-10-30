import torch
import collections
import numpy as np

class FC640(torch.nn.Module):
    def __init__(self, dim_output, num_hidden_units, device):
        super(FC640, self).__init__()
        self.dim_output = dim_output
        self.dim_input = 640
        self.device = device

        self.num_hidden_units = num_hidden_units

        self.bn = {}
        for i, num_hidden_unit in enumerate(num_hidden_units):
            self.bn[i] = torch.nn.BatchNorm1d(
                num_features=num_hidden_unit,
                eps=0,
                momentum=1,
                affine=False,
                track_running_stats=False
            )
    
    def get_weight_shape(self):
        w_shape = collections.OrderedDict()
        num_hidden_units = [self.dim_input]
        num_hidden_units.extend(self.num_hidden_units)

        for i in range(len(self.num_hidden_units)):
            w_shape['w{0:d}'.format(i + 1)] = (num_hidden_units[i + 1], num_hidden_units[i])
            w_shape['b{0:d}'.format(i + 1)] = num_hidden_units[i + 1]

        w_shape['w{0:d}'.format(len(self.num_hidden_units) + 1)] = (self.dim_output, self.num_hidden_units[-1])
        w_shape['b{0:d}'.format(len(self.num_hidden_units) + 1)] = self.dim_output
        return w_shape

    def initialise_weights(self):
        w = {}

        # w['w1'] = torch.empty((self.dim_output, self.dim_input), device=self.device)
        # torch.nn.init.kaiming_normal_(tensor=w['w1'])
        # w['w1'].requires_grad_()
        # w['b1'] = torch.zeros(self.dim_output, requires_grad=True, device=self.device)

        w_shape = self.get_weight_shape()
        for key in w_shape.keys():
            if ('w' in key):
                w[key] = torch.empty(w_shape[key], device=self.device)
                torch.nn.init.kaiming_normal_(tensor=w[key])
                w[key].requires_grad_()
            if 'b' in key:
                w[key] = torch.zeros(w_shape[key], requires_grad=True, device=self.device)

        return w

    def forward(self, x, w, p_dropout=0):
        out = torch.nn.functional.dropout(input=x, p=p_dropout)
        # out = x
        for i in range(len(self.num_hidden_units)):
            out = torch.nn.functional.linear(
                input=out,
                weight=w['w{0:d}'.format(i + 1)],
                bias=w['b{0:d}'.format(i + 1)]
            )
            out = self.bn[i](out)
            out = torch.nn.functional.relu(out)
        # out = torch.nn.functional.leaky_relu(input=out, negative_slope=0.1)

        # out = torch.nn.functional.dropout(input=out, p=p_dropout)
        out = torch.nn.functional.linear(
            input=out,
            weight=w['w{0:d}'.format(len(self.num_hidden_units) + 1)],
            bias=w['b{0:d}'.format(len(self.num_hidden_units) + 1)]
        )
        return out