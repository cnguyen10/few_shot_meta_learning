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

        self.params = torch.nn.ParameterList([torch.nn.Parameter(p.float()) \
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

class PlatipusNet(torch.nn.Module):
    """A class to hold meta-parameters used in PLATIPUS algorithm

    Meta-parameters include:
        - mu_theta
        - log_sigma_theta
        - log_v_q - note that, here v_q is the "std", not the covariance as in the paper.
        One can simply square it and get the one in the paper.
        - learning rate: gamma_p
        - learning rate: gamma_q
    
    Note: since the package "higher" is designed to handle ParameterList,
    the implementation requires to keep the order of such parameters mentioned above.
    This is annoying, but hopefully, the authors of "higher" could extend to handle
    ParameterDict.
    """

    def __init__(self, base_net: torch.nn.Module, **kwargs) -> None:
        super().__init__()

        # dict of parameters of based network
        base_state_dict = base_net.state_dict()

        # add shapes of parameters into self
        self.parameter_shapes = []
        self.num_base_params = 0
        for param in base_state_dict.values():
            self.parameter_shapes.append(param.shape)
            self.num_base_params += np.prod(param.shape)

        # initialize ParameterList
        self.params = torch.nn.ParameterList(parameters=None)

        # add parameters into ParameterList
        self.params.append(parameter=torch.nn.Parameter(data=torch.randn(size=(self.num_base_params,))))
        self.params.append(parameter=torch.nn.Parameter(data=torch.randn(size=(self.num_base_params,)) - 4))
        self.params.append(parameter=torch.nn.Parameter(data=torch.randn(size=(self.num_base_params,)) - 4))
        # for _ in ("mu_theta", "log_sigma_theta", "log_v_q"):
        #     params_list = intialize_parameters(state_dict=base_state_dict)
        #     params_vec = torch.nn.utils.parameters_to_vector(parameters=params_list) - 4 # flattened tensor
        #     self.params.append(parameter=torch.nn.Parameter(data=params_vec))
        
        self.params.append(parameter=torch.nn.Parameter(data=torch.tensor(0.01))) # gamma_p
        self.params.append(parameter=torch.nn.Parameter(data=torch.tensor(0.01))) # gamma_q

    def forward(self) -> dict:
        """Generate a dictionary consisting of meta-paramters
        """
        meta_params = dict.fromkeys(("mu_theta", "log_sigma_theta", "log_v_q", "gamma_p", "gamma_q"))

        meta_params["mu_theta"] = vector_to_list_parameters(vec=self.params[0], parameter_shapes=self.parameter_shapes)
        meta_params["log_sigma_theta"] = vector_to_list_parameters(vec=self.params[1], parameter_shapes=self.parameter_shapes)
        meta_params["log_v_q"] = vector_to_list_parameters(vec=self.params[2], parameter_shapes=self.parameter_shapes)
        meta_params["gamma_p"] = self.params[3]
        meta_params["gamma_q"] = self.params[4]

        return meta_params