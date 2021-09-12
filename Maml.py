import torch
import higher
import typing
import os

from MLBaseClass import MLBaseClass
from HyperNetClasses import IdentityNet
from CommonModels import CNN, ResNet18, FcNet
from _utils import train_val_split

class Maml(MLBaseClass):
    def __init__(self, config: dict) -> None:
        super().__init__(config=config)

        self.hyper_net_class = IdentityNet

    def load_model(self, resume_epoch: int, eps_dataloader: torch.utils.data.DataLoader, **kwargs) -> dict:
        """Initialize or load the hyper-net and base-net models

        Args:
            hyper_net_class: point to the hyper-net class of interest: IdentityNet for MAML or NormalVariationalNet for VAMPIRE
            resume_epoch: the index of the file containing the saved model
            eps_dataloader:

        Returns: a dictionray consisting of the following key-value pair:
            hypet_net: the hyper neural network
            f_base_net: the base neural network
            optimizer: the optimizer for the parameter of the hyper-net
        """
        # initialize a dictionary containing the parameters of interst
        model = dict.fromkeys(["hyper_net", "f_base_net", "optimizer"])

        if resume_epoch is None:
            resume_epoch = self.config['resume_epoch']

        if self.config['network_architecture'] == 'FcNet':
            base_net = FcNet(
                dim_output=self.config['num_ways'],
                num_hidden_units=(40, 40)
            )
        elif self.config['network_architecture'] == 'CNN':
            base_net = CNN(
                dim_output=self.config['num_ways'],
                bn_affine=self.config['batchnorm'],
                stride_flag=self.config['strided']
            )
        elif self.config['network_architecture'] == 'ResNet18':
            base_net = ResNet18(
                dim_output=self.config['num_ways'],
                bn_affine=self.config['batchnorm'],
                dropout_prob=self.config["dropout_prob"]
            )
        else:
            raise NotImplementedError('Network architecture is unknown. Please implement it in the CommonModels.py.')

        # ---------------------------------------------------------------
        # run a dummy task to initialize lazy modules defined in base_net
        # ---------------------------------------------------------------
        for eps_data in eps_dataloader:
            # split data into train and validation
            split_data = self.config['train_val_split_function'](eps_data=eps_data, k_shot=self.config['k_shot'])
            # run to initialize lazy modules
            base_net.forward(split_data['x_t'])
            break

        params = torch.nn.utils.parameters_to_vector(parameters=base_net.parameters())
        print('Number of parameters of the base network = {0:,}.\n'.format(params.numel()))

        model["hyper_net"] = kwargs["hyper_net_class"](base_net=base_net, num_models=self.config["num_models"])

        # move to device
        base_net.to(self.config['device'])
        model["hyper_net"].to(self.config['device'])

        # functional base network
        model["f_base_net"] = self.torch_module_to_functional(torch_net=base_net)

        # add running_mean and running_var for BatchNorm2d
        for m in model["f_base_net"].modules():
            if isinstance(m, torch.nn.BatchNorm2d):
                m.running_mean = None
                m.running_var = None

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
        """See MLBaseClass for the description"""
        # convert hyper_net to its functional form
        f_hyper_net = higher.patch.monkeypatch(
            module=model["hyper_net"],
            copy_initial_weights=False,
            track_higher_grads=self.config["train_flag"]
        )

        for _ in range(self.config['num_inner_updates']):
            q_params = f_hyper_net.fast_params # parameters of the task-specific hyper_net

            # generate task-specific parameter
            base_net_params = f_hyper_net.forward()

            # predict output logits
            logits = model["f_base_net"].forward(x, params=base_net_params)

            # calculate classification loss
            loss = self.config['loss_function'](input=logits, target=y)

            if self.config['first_order']:
                grads = torch.autograd.grad(
                    outputs=loss,
                    inputs=q_params,
                    retain_graph=True
                )
            else:
                grads = torch.autograd.grad(
                    outputs=loss,
                    inputs=q_params,
                    create_graph=True
                )

            new_q_params = []
            for param, grad in zip(q_params, grads):
                new_q_params.append(higher.optim._add(tensor=param, a1=-self.config['inner_lr'], a2=grad))

            f_hyper_net.update_params(new_q_params)

        return f_hyper_net

    def prediction(self, x: torch.Tensor, adapted_hyper_net: higher.patch._MonkeyPatchBase, model: dict) -> torch.Tensor:
        """See MLBaseClass for the description"""
        # generate task-specific parameter
        base_net_params = adapted_hyper_net.forward()

        logits = model["f_base_net"].forward(x, params=base_net_params)

        return logits

    def validation_loss(self, x: torch.Tensor, y: torch.Tensor, adapted_hyper_net: higher.patch._MonkeyPatchBase, model: dict) -> torch.Tensor:
        """See MLBaseClass for the description"""
        logits = self.prediction(x=x, adapted_hyper_net=adapted_hyper_net, model=model)

        loss = self.config['loss_function'](input=logits, target=y)

        return loss

    def evaluation(self, x_t: torch.Tensor, y_t: torch.Tensor, x_v: torch.Tensor, y_v: torch.Tensor, model: dict) -> typing.Tuple[float, float]:
        """See MLBaseClass for the description"""
        adapted_hyper_net = self.adaptation(x=x_t, y=y_t, model=model)
        
        logits = self.prediction(x=x_v, adapted_hyper_net=adapted_hyper_net, model=model)

        loss = self.config['loss_function'](input=logits, target=y_v)

        accuracy = (logits.argmax(dim=1) == y_v).float().mean().item()

        return loss.item(), accuracy * 100

    @staticmethod
    def torch_module_to_functional(torch_net: torch.nn.Module) -> higher.patch._MonkeyPatchBase:
        """Convert a conventional torch module to its "functional" form
        """
        f_net = higher.patch.make_functional(module=torch_net)
        f_net.track_higher_grads = False
        f_net._fast_params = [[]]

        return f_net