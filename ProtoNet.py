import torch
import higher
import typing
import os

from MLBaseClass import MLBaseClass
from _utils import get_cls_prototypes, euclidean_distance, train_val_split
from CommonModels import CNN, ResNet18

class ProtoNet(MLBaseClass):
    def __init__(self, config: dict) -> None:
        super().__init__(config=config)

        self.hyper_net_class = None # dummy to match with MAML and VAMPIRE

    def load_model(self, resume_epoch: int = None, **kwargs) -> typing.Tuple[torch.nn.Module, typing.Optional[higher.patch._MonkeyPatchBase], torch.optim.Optimizer]:
        """Initialize or load the protonet and its optimizer

        Args:
            resume_epoch: the index of the file containing the saved model

        Returns: a tuple consisting of
            protonet: the prototypical network
            base_net: dummy to match with MAML and VAMPIRE
            opt: the optimizer for the prototypical network
        """
        if resume_epoch is None:
            resume_epoch = self.config['resume_epoch']

        if self.config['network_architecture'] == 'CNN':
            protonet = CNN(
                dim_output=None,
                bn_affine=self.config['batchnorm']
            )
        elif self.config['network_architecture'] == 'ResNet18':
            protonet = ResNet18(
                dim_output=None,
                bn_affine=self.config['batchnorm']
            )
        else:
            raise NotImplementedError('Network architecture is unknown. Please implement it in the CommonModels.py.')
        
        # ---------------------------------------------------------------
        # run a dummy task to initialize lazy modules defined in base_net
        # ---------------------------------------------------------------
        eps_data = kwargs['eps_generator'].generate_episode(episode_name=None)
        # split data into train and validation
        xt, _, _, _ = train_val_split(X=eps_data, k_shot=self.config['k_shot'], shuffle=True)
        # convert numpy data into torch tensor
        x_t = torch.from_numpy(xt).float()
        # run to initialize lazy modules
        protonet(x_t)

        # move to device
        protonet.to(self.config['device'])

        # optimizer
        opt = torch.optim.Adam(params=protonet.parameters(), lr=self.config['meta_lr'])

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
            protonet.load_state_dict(state_dict=saved_checkpoint['hyper_net_state_dict'])
            opt.load_state_dict(state_dict=saved_checkpoint['opt_state_dict'])

            # update learning rate
            for param_group in opt.param_groups:
                if param_group['lr'] != self.config['meta_lr']:
                    param_group['lr'] = self.config['meta_lr']

        return protonet, None, opt

    def adapt_and_predict(self, model: typing.Tuple[torch.nn.Module, typing.Optional[higher.patch._MonkeyPatchBase], torch.optim.Optimizer], x_t: torch.Tensor, y_t: torch.Tensor, x_v: torch.Tensor, y_v: torch.Tensor) -> typing.Tuple[higher.patch._MonkeyPatchBase, typing.List[torch.Tensor]]:
        """Adapt and predict the labels of the queried data
        """
        # -------------------------
        # adapt to task by calculating prototypes
        # -------------------------
        z_t = model[0].forward(x_t) # embed data into the latent space
        cls_prototypes = get_cls_prototypes(x=z_t, y=y_t)

        # -------------------------
        # predict labels of queried data
        # -------------------------
        z_v = model[0].forward(x_v)
        distance_matrix = euclidean_distance(matrixN=z_v, matrixM=cls_prototypes)
        logits = [-distance_matrix]

        return None, logits

    def loss_extra(self, **kwargs) -> typing.Union[torch.Tensor, float]:
        return 0.

    @staticmethod
    def KL_divergence(**kwargs) -> typing.Union[torch.Tensor, float]:
        return 0.
    
    def loss_prior(self, model: typing.Tuple[torch.nn.Module, typing.Optional[higher.patch._MonkeyPatchBase], torch.optim.Optimizer], **kwargs) -> typing.Union[torch.Tensor, float]:
        """Loss prior or regularization for the meta-parameter
        """
        return 0.