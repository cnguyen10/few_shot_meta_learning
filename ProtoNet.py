from numpy import mod
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

    def load_model(self, resume_epoch: int = None, **kwargs) -> dict:
        """Initialize or load the protonet and its optimizer

        Args:
            resume_epoch: the index of the file containing the saved model

        Returns: a dictionary consisting of
            protonet: the prototypical network
            optimizer: the optimizer for the prototypical network
        """
        model = dict.fromkeys((["hyper_net", "optimizer"]))

        if resume_epoch is None:
            resume_epoch = self.config['resume_epoch']

        if self.config['network_architecture'] == 'CNN':
            model["hyper_net"] = CNN(
                dim_output=None,
                bn_affine=self.config['batchnorm']
            )
        elif self.config['network_architecture'] == 'ResNet18':
            model["hyper_net"] = ResNet18(
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
        model["hyper_net"](x_t)

        params = torch.nn.utils.parameters_to_vector(parameters=model["hyper_net"].parameters())
        print('Number of parameters of the base network = {0:,}.\n'.format(params.numel()))

        # move to device
        model["hyper_net"].to(self.config['device'])

        # optimizer
        model["optimizer"] = torch.optim.Adam(params=model["hyper_net"].parameters(), lr=self.config['meta_lr'])

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
            model["hyper_net"].load_state_dict(state_dict=saved_checkpoint['hyper_net_state_dict'])
            model["optimizer"].load_state_dict(state_dict=saved_checkpoint['opt_state_dict'])

            # update learning rate
            for param_group in model["optimizer"].param_groups:
                if param_group['lr'] != self.config['meta_lr']:
                    param_group['lr'] = self.config['meta_lr']

        return model

    def adaptation(self, x: torch.Tensor, y: torch.Tensor, model: dict) -> higher.patch._MonkeyPatchBase:
        """Calculate the prototype of each class
        """
        z = model["hyper_net"].forward(x) # embed data into the latent space
        cls_prototypes = get_cls_prototypes(x=z, y=y)

        return cls_prototypes

    def prediction(self, x: torch.Tensor, adapted_hyper_net: torch.Tensor, model: dict) -> torch.Tensor:
        z = model["hyper_net"].forward(x)
        distance_matrix = euclidean_distance(matrixN=z, matrixM=adapted_hyper_net)
        logits = -distance_matrix

        return logits

    def validation_loss(self, x: torch.Tensor, y: torch.Tensor, adapted_hyper_net: torch.Tensor, model: dict) -> torch.Tensor:
        logits = self.prediction(x=x, adapted_hyper_net=adapted_hyper_net, model=model)
        loss = torch.nn.functional.cross_entropy(input=logits, target=y)

        return loss

    def evaluation(self, x_t: torch.Tensor, y_t: torch.Tensor, x_v: torch.Tensor, y_v: torch.Tensor, model: dict) -> typing.Tuple[float, float]:
        class_prototypes = self.adaptation(x=x_t, y=y_t, model=model)

        logits = self.prediction(x=x_v, adapted_hyper_net=class_prototypes, model=model)

        loss = torch.nn.functional.cross_entropy(input=logits, target=y_v)

        accuracy = (logits.argmax(dim=1) == y_v).float().mean().item()

        return loss.item(), accuracy * 100