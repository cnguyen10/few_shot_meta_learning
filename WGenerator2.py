import torch
import numpy as np
import collections

# num_hidden_units = (256, 1024)
# num_hidden_units = (128, 512)

class WeightGenerator(torch.nn.Module):
    def __init__(self, z_dim, dim_output, num_hidden_units, tanh_scale=1., device=torch.device('cpu')):
        super(WeightGenerator, self).__init__()
        self.z_dim = z_dim
        self.dim_output = dim_output
        self.device = device
        self.tanh_scale = tanh_scale

        self.num_hidden_units = num_hidden_units

    def get_generator_shape(self):
        generator_shape = {}

        num_net_units = [self.z_dim]
        for num_units in self.num_hidden_units:
            num_net_units.append(num_units)
        num_net_units.append(self.dim_output)

        for i in range(len(num_net_units) - 1):
            generator_shape['fc{0:d}'.format(i + 1)] = (num_net_units[i + 1], num_net_units[i])
        # print(generator_shape)
        return generator_shape
    
    def initialise_generator(self):
        w = {}

        generator_shape = self.get_generator_shape()
        for i in range(len(self.num_hidden_units) + 1):
            w['fc{0:d}_w'.format(i + 1)] = torch.empty(
                generator_shape['fc{0:d}'.format(i + 1)],
                device=self.device
            )
            torch.nn.init.kaiming_normal_(
                tensor=w['fc{0:d}_w'.format(i + 1)],
                mode='fan_in',
                nonlinearity='relu'
            )
            # torch.nn.init.xavier_normal_(
            #     tensor=w['fc{0:d}_w'.format(i + 1)],
            #     gain=1.
            # )
            w['fc{0:d}_w'.format(i + 1)].requires_grad_()

            w['fc{0:d}_b'.format(i + 1)] = torch.zeros(
                generator_shape['fc{0:d}'.format(i + 1)][0],
                device=self.device,
                requires_grad=True
            )
        return w

    def forward(self, z, w_generator, p_dropout=0): # generate weights for the targeted network
        w_flatten = z
        for i in range(len(self.num_hidden_units) + 1):
            w_flatten = torch.nn.functional.linear(
                input=w_flatten,
                weight=w_generator['fc{0:d}_w'.format(i + 1)],
                bias=w_generator['fc{0:d}_b'.format(i + 1)]
            )
            if (i < len(self.num_hidden_units)):
                w_flatten = torch.nn.functional.relu(w_flatten)
                # w_flatten = torch.nn.functional.leaky_relu(w_flatten, negative_slope=0.2)
                if p_dropout > 0:
                    w_flatten = torch.nn.functional.dropout(input=w_flatten, p=p_dropout)
        
        w_flatten = self.tanh_scale*torch.tanh(w_flatten/self.tanh_scale)
        
        return w_flatten
        

class WeightDiscriminator(torch.nn.Module):
    def __init__(self, z_dim, dim_input, num_hidden_units, device=torch.device('cpu')):
        super(WeightDiscriminator, self).__init__()
        self.z_dim = z_dim
        self.dim_input = dim_input
        self.device = device

        self.num_hidden_units = num_hidden_units

    def get_discriminator_shape(self):
        discriminator_shape = {}

        num_net_units = [self.dim_input]
        for num_units in reversed(self.num_hidden_units):
        # for num_units in self.num_hidden_units:
            num_net_units.append(num_units)
        num_net_units.append(self.z_dim)
        num_net_units.append(1)

        for i in range(len(num_net_units) - 1):
            discriminator_shape['fc{0:d}'.format(i + 1)] = (num_net_units[i + 1], num_net_units[i])
        # print(discriminator_shape)
        return discriminator_shape

    def initialise_discriminator(self):
        w = {}

        discriminator_shape = self.get_discriminator_shape()
        for i in range(len(self.num_hidden_units) + 2):
            w['fc{0:d}_w'.format(i + 1)] = torch.empty(
                discriminator_shape['fc{0:d}'.format(i + 1)],
                device=self.device
            )
            torch.nn.init.kaiming_normal_(
                tensor=w['fc{0:d}_w'.format(i + 1)],
                mode='fan_in',
                nonlinearity='leaky_relu'
            )
            # torch.nn.init.xavier_normal_(
            #     tensor=w['fc{0:d}_w'.format(i + 1)],
            #     gain=1.
            # )
            w['fc{0:d}_w'.format(i + 1)].requires_grad_()

            w['fc{0:d}_b'.format(i + 1)] = torch.zeros(
                discriminator_shape['fc{0:d}'.format(i + 1)][0],
                device=self.device,
                requires_grad=True
            )
        return w
    
    def forward(self, w_input, w_discriminator, p_dropout=0): # without sigmoid at the output
        out = w_input
        for i in range(len(self.num_hidden_units) + 2):
            out = torch.nn.functional.linear(
                input=out,
                weight=w_discriminator['fc{0:d}_w'.format(i + 1)],
                bias=w_discriminator['fc{0:d}_b'.format(i + 1)]
            )
            if (i + 1) < (len(self.num_hidden_units) + 1):
                out = torch.nn.functional.relu(input=out)
                # out = torch.nn.functional.leaky_relu(out, negative_slope=0.2)
                if p_dropout > 0:
                    out = torch.nn.functional.dropout(out, p=p_dropout)
        return out