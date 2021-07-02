import torch
import numpy as np
import typing

from _utils import intialize_parameters, vector_to_list_parameters

class IdentityNet(torch.nn.Module):
    """Identity hyper-net class for MAML"""
    def __init__(self, base_net: torch.nn.Module, **kwargs) -> None:
        super(IdentityNet, self).__init__()
        base_state_dict = base_net.state_dict()

        params = intialize_parameters(state_dict=base_state_dict)

        self.params = torch.nn.ParameterList([torch.nn.Parameter(p) \
            for p in params])
        self.identity = torch.nn.Identity()

    def forward(self) -> typing.List[torch.Tensor]:
        out = []
        for param in self.params:
            temp = self.identity(param)
            out.append(temp)
        return out

class NormalVariationalNet(torch.nn.Module):
    """A simple neural network that simulate the
    reparameterization trick. Its parameters are
    the mean and std-vector
    """
    def __init__(self, base_net: torch.nn.Module, **kwargs) -> None:
        """
        Args:
            base_net: the base network
        """
        super(NormalVariationalNet, self).__init__()

        # dict of parameters of based network
        base_state_dict = base_net.state_dict()

        mean = intialize_parameters(state_dict=base_state_dict)

        # initialize parameters
        self.mean = torch.nn.ParameterList([torch.nn.Parameter(m) \
            for m in mean])
        self.log_std = torch.nn.ParameterList([torch.nn.Parameter(torch.rand_like(v) - 4) \
            for v in base_state_dict.values()])

        self.num_base_params = np.sum([torch.numel(p) for p in self.mean])

    def forward(self) -> typing.List[torch.Tensor]:
        """Output the parameters of the base network in list format to pass into higher monkeypatch
        """
        out = []
        for m, log_s in zip(self.mean, self.log_std):
            eps_normal = torch.randn_like(m, device=m.device)
            temp = m + eps_normal * torch.exp(input=log_s)
            out.append(temp)
        return out

class EnsembleNet(torch.nn.Module):
    """A hyper-net class for BMAML that stores a set of parameters (known as particles)
    """
    def __init__(self, base_net: torch.nn.Module, **kwargs) -> None:
        """Initiate an instance of EnsembleNet

        Args:
            base_net: the network of interest
            num_particles: number of models
        """
        super().__init__()

        self.num_particles = kwargs["num_models"]

        if (self.num_particles <= 1):
            raise ValueError("Minimum number of particles is 2.")

        # dict of parameters of based network
        base_state_dict = base_net.state_dict()

        # add shapes of parameters into self
        self.parameter_shapes = []
        for param in base_state_dict.values():
            self.parameter_shapes.append(param.shape)

        self.params = torch.nn.ParameterList(parameters=None) # empty parameter list

        for _ in range(self.num_particles):
            params_list = intialize_parameters(state_dict=base_state_dict) # list of tensors
            params_vec = torch.nn.utils.parameters_to_vector(parameters=params_list) # flattened tensor
            self.params.append(parameter=torch.nn.Parameter(data=params_vec))
        
        self.num_base_params = np.sum([torch.numel(p) for p in self.params[0]])

    def forward(self, i: int) -> typing.List[torch.Tensor]:
        return vector_to_list_parameters(vec=self.params[i], parameter_shapes=self.parameter_shapes)