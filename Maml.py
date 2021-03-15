import torch
import higher
import typing
import os

from MLBaseClass import MLBaseClass
from _utils import IdentityNet
from CommonModels import CNN, ResNet18
from _utils import train_val_split

class Maml(MLBaseClass):
    def __init__(self, config: dict) -> None:
        super().__init__(config=config)

        if self.config['min_way'] != self.config['max_way']:
            raise ValueError('MAML works with a fixed number of ways only.')
        
        self.config['num_models'] = 1 # overwrite number of models for speed

        self.hyper_net_class = IdentityNet

    def load_model(self, resume_epoch: int = None, **kwargs) -> typing.Tuple[torch.nn.Module, typing.Optional[higher.patch._MonkeyPatchBase], torch.optim.Optimizer]:
        """Initialize or load the hyper-net and base-net models

        Args:
            hyper_net_class: point to the hyper-net class of interest: IdentityNet for MAML or NormalVariationalNet for VAMPIRE
            resume_epoch: the index of the file containing the saved model

        Returns: a tuple consisting of
            hypet_net: the hyper neural network
            base_net: the base neural network
            meta_opt: the optimizer for meta-parameter
        """
        return self.load_maml_like_model(resume_epoch=resume_epoch, **kwargs)

    def adapt_and_predict(self, model: typing.Tuple[torch.nn.Module, typing.Optional[higher.patch._MonkeyPatchBase], torch.optim.Optimizer], x_t: torch.Tensor, y_t: torch.Tensor, x_v: torch.Tensor, y_v: torch.Tensor) -> typing.Tuple[higher.patch._MonkeyPatchBase, typing.List[torch.Tensor]]:
        """Adapt and predict the labels of the queried data
        """
        # -------------------------
        # adapt on the support data
        # -------------------------
        f_base_net = self.torch_module_to_functional(torch_net=model[1])
        f_hyper_net = self.adapt_to_episode(x=x_t, y=y_t, hyper_net=model[0], f_base_net=f_base_net, train_flag=True)

        # -------------------------
        # predict labels of queried data
        # -------------------------
        logits = self.predict(x=x_v, f_hyper_net=f_hyper_net, f_base_net=f_base_net)

        return f_hyper_net, logits

    def loss_extra(self, **kwargs) -> typing.Union[torch.Tensor, float]:
        return 0.

    @staticmethod
    def KL_divergence(**kwargs) -> typing.Union[torch.Tensor, float]:
        return 0.
    
    def loss_prior(self, model: typing.Tuple[torch.nn.Module, typing.Optional[higher.patch._MonkeyPatchBase], torch.optim.Optimizer], **kwargs) -> typing.Union[torch.Tensor, float]:
        """Loss prior or regularization for the meta-parameter
        """
        return 0.