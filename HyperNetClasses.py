import torch
import numpy as np
import typing

from _utils import intialize_parameters

class IdentityNet(torch.nn.Module):
    """Identity hyper-net class for MAML"""
    def __init__(self, base_net: torch.nn.Module) -> None:
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
    def __init__(self, base_net: torch.nn.Module) -> None:
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