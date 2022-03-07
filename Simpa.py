import torch
import higher
from torch.utils.tensorboard import SummaryWriter

import numpy as np

import os
import typing
import logging

from MLBaseClass import MLBaseClass
from CommonModels import CNN, FcNet
from HyperNetClasses import IdentityNet
from _utils import torch_module_to_functional

logging.basicConfig(level=logging.INFO)

def vector_to_parameters(
    vec: torch.Tensor,
    parameter_shapes: typing.List[torch.Size]
) -> typing.List[torch.Tensor]:
    """convert a vector to model parameters

    Args:
        vec: an input 1d-vector
        parameter_shapes: a list of parameter shapes

    Returns:
        params: a list of tensors
    """
    params = []

    pointer = 0
    for parameter_shape in parameter_shapes:
        num_param = parameter_shape.numel()

        param = vec[pointer:pointer + num_param].view(parameter_shape)
        params.append(param)

        pointer += num_param
    
    return params

class FunctionalGenerator(torch.nn.Module):
    """a class for the generator"""

    def __init__(self, base_net: torch.nn.Module) -> None:
        """initialize the generator of interest

        Args:
            base_net: the neural network of interest to solve each task
                e.g. 4-layer CNN, Resnet18
        """
        super().__init__()

        # region SHAPE of BASE NETWORK
        sd = base_net.state_dict()

        num_base_net_params = 0
        self.param_shapes = []
        for param in sd.values():
            num_base_net_params += param.numel()
            self.param_shapes.append(param.shape)
        # endregion

        return None

    def forward_vector(self, z: torch.Tensor, w: typing.List[torch.Tensor]) -> torch.Tensor:
        """generate the parameter vector for the base neural network
        The generator here is designed as a 2-hidden-layer fully-connected network

        Args:
            z: a latent noise input
            w: the parameters of the generator

        Returns:
            param_vec: the (flattened) parameter vector of the base neural network
        """
        param_vec = torch.nn.functional.linear(
            input=z,
            weight=w[0],
            bias=w[1]
        )
        param_vec = torch.nn.functional.relu(input=param_vec)
        # param_vec = torch.nn.functional.leaky_relu(input=param_vec, negative_slope=0.1)
        # param_vec = torch.nn.functional.dropout(input=param_vec, p=0.25, training=train)

        param_vec = torch.nn.functional.linear(
            input=param_vec,
            weight=w[2],
            bias=w[3]
        )
        param_vec = torch.nn.functional.relu(input=param_vec)
        # param_vec = torch.nn.functional.leaky_relu(input=param_vec, negative_slope=0.1)
        # param_vec = torch.nn.functional.dropout(input=param_vec, p=0.25, training=train)

        param_vec = torch.nn.functional.linear(
            input=param_vec,
            weight=w[4],
            bias=w[5]
        )

        param_vec = torch.tanh(input=param_vec)

        return param_vec

    def forward(self, z: torch.Tensor, w: typing.List[torch.Tensor]) -> torch.Tensor:
        """generate the parameters for the base neural network

        Args:
            z: a latent noise that has a batch-size of 1
            w: the parameter of the generator

        Returns:
            params: the parameters of the base neural network
        """
        param_vec = self.forward_vector(z=z, w=w)

        params = vector_to_parameters(
            vec=param_vec[0],
            parameter_shapes=self.param_shapes
        )

        return params

class Simpa(MLBaseClass):
    """Implementation of Simpa"""

    def __init__(self, config: dict) -> None:
        config['s_theta'] = 1
        config['epsilon'] = 0.1

        # setting prior for task-specific parameter p(w)
        config['p_w'] = torch.distributions.normal.Normal(
            loc=torch.tensor(0., device=config['device']),
            scale=torch.tensor(1., device=config['device'])
        )

        # learning rates to train PHI network
        config['phi_inner_lr'] = 5e-4
        config['phi_lr'] = 1e-4

        super().__init__(config=config)

        return None

    def load_model(
        self,
        resume_epoch: int,
        eps_dataloader: torch.utils.data.DataLoader
    ) -> dict:
        """initialize/load the whole SImPa model

        Args:
            resume_epoch: the index of saved checkpoint to load model to continue training
            eps_dataloader: the episode/task dataloader

        Returns:
            model: a dictionary storing:
                hyper_net: the generator of interest (or meta-generator)
                f_genrator: the generator adapted/finetuned on a task
                f_base_net: the functional (skeleton) form of the base network
                optimizer: the optimizer of the generator
                phi_hyper_net: the meta-network of the phi network
                f_phi_base_net: the functional (skeleton) form of the base network for the PHI network
                phi_optimizer: the optimizer for the phi_hyper_net
        """
        # initialize a dictionary containing the parameters of interst
        model = dict.fromkeys(('hyper_net', 'f_generator', 'f_base_net', 'optimizer', 'phi_hyper_net', 'f_phi_base_net', 'phi_optimizer'))

        if resume_epoch is None:
            resume_epoch = self.config['resume_epoch']

        # region BASE-NET
        # construct the base network base on the input "network_architecture"
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
            # model['encoder'].forward(split_data['x_t'])
            break

        self.base_net_shapes = [param.shape for param in base_net.parameters()]
        params = torch.nn.utils.parameters_to_vector(parameters=base_net.parameters())
        self.base_net_num_params = params.numel()
        print('Number of parameters of the base network = {0:,}.\n'.format(self.base_net_num_params))

        # functional base network
        model['f_base_net'] = torch_module_to_functional(torch_net=base_net)

        # add running_mean and running_var for BatchNorm2d
        for m in model['f_base_net'].modules():
            if isinstance(m, torch.nn.BatchNorm2d):
                m.running_mean = None
                m.running_var = None
        # endregion

        # region GENERATOR
        model['f_generator'] = FunctionalGenerator(base_net=base_net)
        model['hyper_net'] = IdentityNet(
            base_net=torch.nn.Sequential(
                torch.nn.Linear(in_features=128, out_features=256),
                torch.nn.ReLU(),

                torch.nn.Linear(in_features=256, out_features=512),
                torch.nn.ReLU(),

                torch.nn.Linear(
                    in_features=512,
                    out_features=self.base_net_num_params
                )
            )
        )
        model['hyper_net'].to(device=self.config['device'])
        # endregion

        # region PHI NETWORK
        phi_base_net = torch.nn.Sequential(
            torch.nn.Linear(
                in_features=self.base_net_num_params,
                out_features=512
            ),
            torch.nn.BatchNorm1d(
                num_features=512,
                eps=0.,
                momentum=1.,
                affine=False,
                track_running_stats=False
            ),
            torch.nn.ReLU(),
            # torch.nn.Dropout(p=0.25),

            torch.nn.Linear(in_features=512, out_features=256),
            torch.nn.BatchNorm1d(
                num_features=256,
                eps=0.,
                momentum=1.,
                affine=False,
                track_running_stats=False
            ),
            torch.nn.ReLU(),
            # torch.nn.Dropout(p=0.25),

            torch.nn.Linear(in_features=256, out_features=128),
            torch.nn.BatchNorm1d(
                num_features=128,
                eps=0.,
                momentum=1.,
                affine=False,
                track_running_stats=False
            ),
            torch.nn.ReLU(),
            # torch.nn.Dropout(p=0.25),

            torch.nn.Linear(in_features=128, out_features=1)
        )
        model['f_phi_base_net'] = torch_module_to_functional(torch_net=phi_base_net)
        model['phi_hyper_net'] = IdentityNet(base_net=phi_base_net)
        model['phi_hyper_net'].to(device=self.config['device'])

        for m in model['f_phi_base_net'].modules():
            if isinstance(m, torch.nn.BatchNorm1d):
                m.running_mean = None
                m.running_var = None
        # endregion

        # region OPTIMIZER
        model['optimizer'] = torch.optim.Adam(
            params=model['hyper_net'].parameters(),
            lr=self.config['meta_lr']
        )
        model['phi_optimizer'] = torch.optim.Adam(
            params=model['phi_hyper_net'].parameters(),
            lr=self.config['phi_lr']
        )
        # endregion

        if resume_epoch > 0:
            # path to the saved file
            checkpoint_path = os.path.join(
                self.config['logdir'],
                'Epoch_{0:d}.pt'.format(resume_epoch)
            )
            
            # load file
            saved_checkpoint = torch.load(
                f=checkpoint_path,
                map_location=lambda storage, loc: storage.cuda(
                    device=self.config['device'].index
                ) if self.config['device'].type == 'cuda' else storage
            )

            # load state dictionaries
            model['hyper_net'].load_state_dict(
                state_dict=saved_checkpoint['hyper_net_state_dict']
            )
            model['optimizer'].load_state_dict(
                state_dict=saved_checkpoint['optimizer_state_dict']
            )

            model['phi_hyper_net'].load_state_dict(
                state_dict=saved_checkpoint['phi_hyper_net_state_dict']
            )
            model['phi_optimizer'].load_state_dict(
                state_dict=saved_checkpoint['phi_optimizer_state_dict']
            )

            # update learning rate
            for param_group in model['optimizer'].param_groups:
                param_group['lr'] = self.config['meta_lr']
            
            for param_group in model['phi_optimizer'].param_groups:
                param_group['lr'] = self.config['phi_lr']

        return model

    def train(
        self,
        train_dataloader: torch.utils.data.DataLoader,
        val_dataloader: typing.Optional[torch.utils.data.DataLoader]
    ) -> None:
        """training

        Args:
            train_dataloader: the dataloader of training tasks
            val_dataloader: the dataloader of validation tasks
        """
        logging.info(msg='Training is started.\nLog is stored at {0:s}.\n'.format(self.config['logdir']))

        # initialize/load model
        # Please see the load_model method implemented in each specific class for further information about the model
        model = self.load_model(
            resume_epoch=self.config['resume_epoch'],
            eps_dataloader=train_dataloader
        )
        model['optimizer'].zero_grad()

        try:
            for epoch_id in range(self.config['resume_epoch'], self.config['resume_epoch'] + self.config['num_epochs'], 1):
                loss_monitor = []
                loss_phi_monitor = []

                # initialize a tensorboard summary writer for logging
                tb_writer = SummaryWriter(
                    log_dir=self.config['logdir'],
                    purge_step=self.config['resume_epoch'] * self.config['num_episodes_per_epoch'] // self.config['minibatch_print'] if self.config['resume_epoch'] > 0 else None
                )
                for eps_count, eps_data in enumerate(train_dataloader):

                    if (eps_count >= self.config['num_episodes_per_epoch']):
                        break

                    # split data into train and validation
                    split_data = self.config['train_val_split_function'](eps_data=eps_data, k_shot=self.config['k_shot'])

                    # move data to GPU (if there is a GPU)
                    x_t = split_data['x_t'].to(device=self.config['device'])
                    y_t = split_data['y_t'].to(device=self.config['device'])
                    x_v = split_data['x_v'].to(device=self.config['device'])
                    y_v = split_data['y_v'].to(device=self.config['device'])

                    # -------------------------
                    # adaptation on training subset
                    # -------------------------
                    adapted_generator_params, phi_params = self.adaptation(
                        x=x_t,
                        y=y_t,
                        model=model
                    )

                    # train hyper-PHI network
                    loss_phi = -self.estimate_KL_lower_bound(
                        generator_params=adapted_generator_params,
                        phi_params=phi_params,
                        model=model
                    )
                    if torch.isnan(input=loss_phi):
                        # raise ValueError('Loss phi is NaN')
                        logging.info(msg='Loss PHI is NaN.')
                        continue

                    loss_phi_monitor.append(loss_phi.item())
                    loss_phi = loss_phi / self.config['minibatch']
                    loss_phi.backward(retain_graph=True)

                    # -------------------------
                    # loss on validation subset
                    # -------------------------
                    loss = self.validation_loss(
                        x=x_v,
                        y=y_v,
                        adapted_generator_params=adapted_generator_params,
                        model=model
                    )
                    if torch.isnan(input=loss):
                        raise ValueError('Loss is NaN.')

                    loss_monitor.append(loss.item())

                    with torch.no_grad():
                        loss.data = torch.clamp(input=loss.data, max=1)

                    # KL divergence on each task
                    KL_lower_bound = self.estimate_KL_lower_bound(
                        generator_params=adapted_generator_params,
                        phi_params=phi_params,
                        model=model
                    )
                    KL_loss = (KL_lower_bound + np.log(y_v.numel()) \
                        / self.config['epsilon']) / (2 * (y_v.numel() - 1))
                    KL_loss = torch.sqrt(input=KL_loss)

                    
                    if torch.isnan(KL_loss) or (KL_loss < 0):
                        KL_loss = 0.

                    loss = (loss + KL_loss) / self.config['minibatch']
                    loss.backward()
                        

                    # update meta-parameters
                    if ((eps_count + 1) % self.config['minibatch'] == 0):
                        KL_loss = self.KL_divergence_standard_normal(
                            p=[param for param in model['hyper_net'].parameters()]
                        )
                        KL_loss = KL_loss + self.config['minibatch'] * \
                            np.log(self.config['minibatch']) / self.config['epsilon']
                        KL_loss = KL_loss / (2 * (self.config['minibatch'] - 1))
                        KL_loss = torch.sqrt(input=KL_loss)
                        KL_loss.backward()

                        torch.nn.utils.clip_grad.clip_grad_norm_(
                            parameters=model['hyper_net'].parameters(),
                            max_norm=10
                        )

                        model['optimizer'].step()
                        model['optimizer'].zero_grad()

                        torch.nn.utils.clip_grad.clip_grad_norm_(
                            parameters=model['phi_hyper_net'].parameters(),
                            max_norm=10
                        )

                        model['phi_optimizer'].step()
                        model['phi_optimizer'].zero_grad()

                        # monitoring
                        if (eps_count + 1) % self.config['minibatch_print'] == 0:
                            # calculate step for Tensorboard Summary Writer
                            global_step = (epoch_id * self.config['num_episodes_per_epoch'] + eps_count + 1) // self.config['minibatch_print']

                            tb_writer.add_scalar(
                                tag='Loss/train',
                                scalar_value=np.mean(loss_monitor),
                                global_step=global_step
                            )
                            tb_writer.add_scalar(
                                tag='Loss/phi',
                                scalar_value=np.mean(loss_phi_monitor),
                                global_step=global_step
                            )

                            # reset monitoring variables
                            loss_monitor = []
                            loss_phi_monitor = []

                            # -------------------------
                            # Validation
                            # -------------------------
                            if val_dataloader is not None:
                                # turn on EVAL mode to disable dropout
                                model['f_base_net'].eval()
                                model['f_phi_base_net'].eval()
                                self.config['train_flag'] = False

                                loss_temp, accuracy_temp = self.evaluate(
                                    num_eps=self.config['num_episodes'],
                                    eps_dataloader=val_dataloader,
                                    model=model
                                )

                                tb_writer.add_scalar(
                                    tag='Loss/validation',
                                    scalar_value=np.mean(loss_temp),
                                    global_step=global_step
                                )
                                tb_writer.add_scalar(
                                    tag='Accuracy/validation',
                                    scalar_value=np.mean(accuracy_temp),
                                    global_step=global_step
                                )

                                model['f_base_net'].train()
                                model['f_phi_base_net'].train()
                                self.config['train_flag'] = True
                                del loss_temp
                                del accuracy_temp

                # save model
                checkpoint = {
                    'hyper_net_state_dict': model['hyper_net'].state_dict(),
                    'optimizer_state_dict': model['optimizer'].state_dict(),
                    'phi_hyper_net_state_dict': model['phi_hyper_net'].state_dict(),
                    'phi_optimizer_state_dict': model['phi_optimizer'].state_dict()
                }
                checkpoint_path = os.path.join(self.config['logdir'], 'Epoch_{0:d}.pt'.format(epoch_id + 1))
                torch.save(obj=checkpoint, f=checkpoint_path)
                print('State dictionaries are saved into {0:s}\n'.format(checkpoint_path))

                tb_writer.close()

            print('Training is completed.')
        finally:
            print('\nClose tensorboard summary writer')
            tb_writer.close()

        return None

    def adaptation(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        model: dict
    ) -> typing.Tuple[typing.List[torch.Tensor], typing.List[torch.Tensor]]:
        """
        """
        # generate the parameters of the generator
        generator_params = model['hyper_net'].forward()

        phi_params = model['phi_hyper_net'].forward()

        for _ in range(self.config['num_inner_updates']):
            grads_accum = [0] * len(generator_params) # accumulate gradients of Monte Carlo sampling

            for _ in range(self.config['num_models']):
                # generate parameter from task-specific hypernet
                base_net_params = model['f_generator'].forward(
                    z=torch.rand(size=(1, 128), device=self.config['device']),
                    w=generator_params
                )

                y_logits = model['f_base_net'].forward(x, params=base_net_params)

                cls_loss = self.config['loss_function'](input=y_logits, target=y)

                if torch.isnan(input=cls_loss):
                    raise ValueError('Adaptation loss is NaN')

                cls_loss = cls_loss / self.config['num_models']

                # with torch.no_grad():
                #     cls_loss.data = torch.clamp(input=cls_loss.data, max=1)

                if self.config['first_order']:
                    grads = torch.autograd.grad(
                        outputs=cls_loss,
                        inputs=generator_params,
                        retain_graph=True
                    )
                else:
                    grads = torch.autograd.grad(
                        outputs=cls_loss,
                        inputs=generator_params,
                        create_graph=True
                    )

                # accumulate gradients from Monte Carlo sampling and average out
                for i in range(len(grads)):
                    # if torch.isnan(input=grads[i]).any():
                    #     raise ValueError('Grad is NaN')
                    grads_accum[i] = grads_accum[i] + grads[i] / self.config['num_models']
            
            # loss related to KL[q(w; lambda) || p(w)]
            new_phi_params = self.train_phi(
                generator_params=generator_params,
                phi_params=phi_params,
                model=model
            )
            phi_params = [new_phi_param + 0. for new_phi_param in new_phi_params]
            KL_lower_bound = self.estimate_KL_lower_bound(
                generator_params=generator_params,
                phi_params=phi_params,
                model=model
            )
            KL_loss = (KL_lower_bound + np.log(y.numel()) / self.config['epsilon']) / (2 * (y.numel() - 1))
            KL_loss = torch.sqrt(input=KL_loss)
            if torch.isnan(input=KL_loss) or (KL_loss.item() < 0):
                KL_loss.data = torch.clamp(input=KL_loss.data, min=0)
            else:
                KL_grads = torch.autograd.grad(
                    outputs=KL_loss,
                    inputs=generator_params,
                    retain_graph=True
                )
                # accumulate gradients from Monte Carlo sampling and average out
                for i in range(len(KL_grads)):
                    # if torch.isnan(KL_grads[i]).any():
                    #     raise ValueError('KL grad is NaN')
                    grads_accum[i] = grads_accum[i] + KL_grads[i]

            # with torch.no_grad():
            #     for i in range(len(grads_accum)):
            #         grad_norm = torch.sqrt(input=torch.sum(input=torch.square(input=grads_accum[i].data)))
            #         # breakpoint()
            #         if (grad_norm > 1000):
            #             grads_accum[i].data = 1000 * grads_accum[i].data / grad_norm

            new_generator_params = [None] * len(generator_params)
            for i in range(len(generator_params)):
                new_generator_params[i] = generator_params[i] - self.config['inner_lr'] * grads_accum[i]

            generator_params = [new_generator_param + 0 for new_generator_param in new_generator_params]
        
        return generator_params, phi_params

    def prediction(
        self,
        x: torch.Tensor,
        adapted_generator_params: typing.List[torch.Tensor],
        model: dict
    ) -> typing.Union[torch.Tensor, typing.List[torch.Tensor]]:
        """
        """
        logits = [None] * self.config['num_models']
        for model_id in range(self.config['num_models']):
            # generate parameter from task-specific hypernet
            base_net_params = model['f_generator'].forward(
                z=torch.rand(size=(1, 128), device=self.config['device']),
                w=adapted_generator_params
            )

            logits_temp = model['f_base_net'].forward(x, params=base_net_params)

            logits[model_id] = logits_temp

        return logits

    def validation_loss(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        adapted_generator_params: typing.List[torch.Tensor],
        model: dict
    ) -> torch.Tensor:
        """
        """
        logits = self.prediction(
            x=x,
            adapted_generator_params=adapted_generator_params,
            model=model
        )

        cls_loss = 0

        # classification loss
        for logits_ in logits:
            cls_loss = cls_loss + self.config['loss_function'](input=logits_, target=y)
        
        cls_loss = cls_loss / len(logits)

        return cls_loss

    def evaluation(
        self,
        x_t: torch.Tensor,
        y_t: torch.Tensor,
        x_v: torch.Tensor,
        y_v: torch.Tensor,
        model: dict
    ) -> typing.Tuple[float, float]:
        """
        """
        adapted_generator_params, _ = self.adaptation(x=x_t, y=y_t, model=model)

        logits = self.prediction(
            x=x_v,
            adapted_generator_params=adapted_generator_params,
            model=model
        )

        cls_loss = self.validation_loss(
            x=x_v,
            y=y_v,
            adapted_generator_params=adapted_generator_params,
            model=model
        )

        y_pred = 0
        for logits_ in logits:
            y_pred = y_pred + torch.softmax(input=logits_, dim=1)
        
        y_pred = y_pred / len(logits)

        accuracy = (y_pred.argmax(dim=1) == y_v).float().mean().item()

        return cls_loss.item(), accuracy * 100

    def estimate_KL_lower_bound(
        self,
        generator_params: typing.List[torch.Tensor],
        phi_params: typing.List[torch.Tensor],
        model: dict
    ) -> torch.Tensor:
        """
        """
        num_samples = 512

        # parameters generated from the generator or q distribution
        param_vecs = model['f_generator'].forward_vector(
            z=torch.rand(size=(num_samples, 128), device=self.config['device']),
            w=generator_params
        )
        KL_lower_bound = torch.mean(input=model['f_phi_base_net'].forward(param_vecs, params=phi_params))

        # generate parameters from prior p
        param_vecs = self.config['p_w'].sample(sample_shape=(num_samples, self.base_net_num_params))
        KL_lower_bound = KL_lower_bound - \
            torch.logsumexp(
                input=model['f_phi_base_net'].forward(param_vecs, params=phi_params),
                dim=(0, 1)
            )
        
        KL_lower_bound = KL_lower_bound - np.log(num_samples)

        return KL_lower_bound
    
    def train_phi(
        self,
        generator_params: typing.List[torch.Tensor],
        phi_params: typing.List[torch.Tensor],
        model: dict
    ) -> typing.List[torch.Tensor]:
        """
        """
        # for _ in range(self.config['num_inner_updates']):
        for _ in range(1):
            KL_lower_bound = self.estimate_KL_lower_bound(
                generator_params=generator_params,
                phi_params=phi_params,
                model=model
            )

            KL_grads = torch.autograd.grad(
                outputs=KL_lower_bound,
                inputs=phi_params,
                retain_graph=True
            )

            new_phi_params = [None] * len(phi_params)
            for i in range(len(phi_params)):
                new_phi_params[i] = phi_params[i] + self.config['phi_inner_lr'] * KL_grads[i]
                # with torch.no_grad():
                #     new_phi_params[i].data = torch.clamp(input=new_phi_params[i].data, min=-10, max=10)
            
            phi_params = [new_phi_param + 0. for new_phi_param in new_phi_params]
        
        return phi_params

    @staticmethod
    def KL_divergence_standard_normal(p: typing.List[torch.Tensor]) -> typing.Union[torch.Tensor, float]:
        """Calculate KL divergence between a diagonal Gaussian with N(0, I)
        """
        KL_div = 0

        n = len(p) // 2

        for i in range(n):
            p_mean = p[i]
            p_log_std = p[n + i]

            KL_div = KL_div + torch.sum(input=torch.square(input=p_mean))
            KL_div = KL_div + torch.sum(input=torch.exp(input=2 * p_log_std))
            KL_div = KL_div - n
            KL_div = KL_div - 2 * torch.sum(input=p_log_std)

        KL_div = KL_div / 2

        return KL_div