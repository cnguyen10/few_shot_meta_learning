import torch
import higher
import typing
import os

from MLBaseClass import MLBaseClass
from _utils import kl_divergence_gaussians, NormalVariationalNet
from CommonModels import CNN, ResNet18

class Vampire(MLBaseClass):
    def __init__(self, config: dict) -> None:
        super().__init__(config=config)

        if self.config['min_way'] != self.config['max_way']:
            raise ValueError('VAMPIRE works with a fixed number of ways only.')

        self.hyper_net_class = NormalVariationalNet

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
        if resume_epoch is None:
            resume_epoch = self.config['resume_epoch']

        if self.config['network_architecture'] == 'CNN':
            base_net = CNN(
                dim_output=self.config['min_way'],
                image_size=self.config['image_size'],
                bn_affine=self.config['batchnorm']
            )
        elif self.config['network_architecture'] == 'ResNet18':
            base_net = ResNet18(
                input_channel=self.config['image_size'][0],
                dim_output=self.config['min_way'],
                bn_affine=self.config['batchnorm']
            )
        else:
            raise NotImplementedError('Network architecture is unknown. Please implement it in the CommonModels.py.')

        hyper_net = kwargs['hyper_net_class'](base_net=base_net)

        # move to device
        base_net.to(self.config['device'])
        hyper_net.to(self.config['device'])

        # optimizer
        meta_opt = torch.optim.Adam(params=hyper_net.parameters(), lr=self.config['meta_lr'])

        # load model if there is saved file
        if resume_epoch > 0:
            # path to the saved file
            checkpoint_path = os.path.join(self.config['logdir'], 'Epoch_{0:d}.pt'.format(resume_epoch))
            
            # load file
            saved_checkpoint = torch.load(
                f=checkpoint_path,
                map_location=lambda storage,
                loc: storage.cuda(self.config['device'].index) if self.config['device'].type == 'cuda' else storage
            )

            # load state dictionaries
            hyper_net.load_state_dict(state_dict=saved_checkpoint['hyper_net_state_dict'])
            meta_opt.load_state_dict(state_dict=saved_checkpoint['opt_state_dict'])

            # update learning rate
            for param_group in meta_opt.param_groups:
                if param_group['lr'] != self.config['meta_lr']:
                    param_group['lr'] = self.config['meta_lr']

        return hyper_net, base_net, meta_opt

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
        """
        """
        if ('p' not in kwargs) or ('q' not in kwargs):
            p = [p_ for p_ in kwargs['model'][0].parameters()]
            q = kwargs['f_hyper_net'].fast_params
        else:
            p = kwargs['p']
            q = kwargs['q']

        KL_div = kl_divergence_gaussians(p=p, q=q)

        return KL_div
    
    def loss_prior(self, model: typing.Tuple[torch.nn.Module, typing.Optional[higher.patch._MonkeyPatchBase], torch.optim.Optimizer], **kwargs) -> typing.Union[torch.Tensor, float]:
        """Loss prior or regularization for the meta-parameter
        """
        return 0.