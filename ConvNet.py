import torch
import numpy as np

import collections

class ConvNet(torch.nn.Module):
    def __init__(
        self,
        dim_input,
        dim_output=None,
        num_filters=[32]*4,
        filter_size=(3, 3),
        bn_flag=True,
        device=torch.device('cpu')
    ):
        '''
        Inputs:
            - dim_input = (in_channels, H, W)
            - dim_output = None for metric learning (no fully connected layer at the end)
            - num_filters = a list of numbers of filters
            - filter_size = kernel size (Kh, Kw)
            - device = cpu or gpu
        '''
        super(ConvNet, self).__init__()
        self.dim_input = dim_input[1:]
        self.filter_size = filter_size
        self.dim_output = dim_output
        self.device = device
        self.bn_flag = bn_flag

        num_channels = [dim_input[0]]
        num_channels.extend(num_filters)
        self.num_channels = num_channels

        # auxiliary functions
        if bn_flag:
            self.bn = {}
            for i in range(len(num_filters)):
                self.bn[i + 1] = torch.nn.BatchNorm2d(
                    num_features=num_filters[i],
                    eps=0,
                    momentum=1,
                    affine=False,
                    track_running_stats=False
                )
    
    def get_dim_features(self):
        '''
        Get the dimensions at the output of each convolutional block
        '''
        dim_features = []

        dim_features.append(self.dim_input)

        # Convolution
        conv_padding = 1
        conv_stride = 1

        # Maxpooling
        pool_padding = 0
        pool_stride = 2

        for _ in range(len(self.num_channels) - 1):
            h_conv2d = np.floor((dim_features[-1][0] + 2*conv_padding - self.filter_size[0])/conv_stride + 1)
            w_conv2d = np.floor((dim_features[-1][1] + 2*conv_padding - self.filter_size[1])/conv_stride + 1)

            h_pool = np.floor((h_conv2d + 2*pool_padding - 2)/pool_stride + 1)
            w_pool = np.floor((w_conv2d + 2*pool_padding - 2)/pool_stride + 1)
            dim_features.append((h_pool, w_pool))
        return dim_features
    
    def get_weight_shape(self):
        w_shape = collections.OrderedDict()
        for i in range(len(self.num_channels) - 1):
            w_shape['conv{0:d}'.format(i + 1)] = (self.num_channels[i + 1], self.num_channels[i], self.filter_size[0], self.filter_size[1])
            w_shape['b{0:d}'.format(i + 1)] = self.num_channels[i + 1]
        
        if self.dim_output is not None:
            dim_features = self.get_dim_features()
            w_shape['w{0:d}'.format(len(self.num_channels))] = (self.dim_output, self.num_channels[-1]*np.prod(dim_features[-1], dtype=np.int16))
            w_shape['b{0:d}'.format(len(self.num_channels))] = self.dim_output
        
        return w_shape

    def initialize_weights(self):
        w = {}
        w_shape = self.get_weight_shape()
        for key in w_shape.keys():
            if 'b' in key:
                w[key] = torch.zeros(w_shape[key], device=self.device, requires_grad=True)
            else:
                w[key] = torch.empty(w_shape[key], device=self.device)
                # torch.nn.init.xavier_uniform_(tensor=w[key], gain=1.)
                torch.nn.init.kaiming_normal_(tensor=w[key], a=0, mode='fan_in', nonlinearity='relu')
                w[key].requires_grad_()
        return w
    
    def forward(self, x, w, p_dropout=0):
        out = x
        for i in range(len(self.num_channels) - 1):
            out = torch.nn.functional.conv2d(
                input=out,
                weight=w['conv{0:d}'.format(i + 1)],
                bias=w['b{0:d}'.format(i + 1)],
                stride=1,
                padding=1
            )
            
            if self.bn_flag:
                out = self.bn[i + 1](out)
            out = torch.nn.functional.relu(out)

            out = torch.nn.functional.max_pool2d(
                input=out,
                kernel_size=2,
                stride=2,
                padding=0
            )
        
        out = torch.flatten(input=out, start_dim=1, end_dim=-1)
        if p_dropout > 0:
            out = torch.nn.functional.dropout(input=out, p=p_dropout)

        if self.dim_output is not None:
            out = torch.nn.functional.linear(
                input=out,
                weight=w['w{0:d}'.format(len(self.num_channels))],
                bias=w['b{0:d}'.format(len(self.num_channels))]
            )
        return out
