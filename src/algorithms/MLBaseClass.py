import torch
import higher

#from torch.utils.tensorboard import SummaryWriter

import numpy as np
import typing
import os
import random
import sys

import abc

# --------------------------------------------------
# Default configuration
# --------------------------------------------------
config = {} # initialize a configuration dictionary

# Hardware
config['device'] = torch.device('cuda:0' if torch.cuda.is_available() \
    else torch.device('cpu'))

# Dataset
config['datasource'] = 'omniglot-py'
config['suffix'] = 'png' # extension of image file: png, jpg
config['image_size'] = (1, 64, 64)
config['ds_folder'] = './datasets' # path to the folder containing the dataset
config['load_images'] = True # load images on RAM for fast access. Set False for large dataset to avoid out-of-memory

# Meta-learning method
config['ml_algorithm'] = 'maml' # either: maml and vampire
config['first_order'] = True # applicable for MAML-like algorithms
config['num_models'] = 1 # number of models used in Monte Carlo to approximate expectation
config['KL_weight'] = 1e-4
config['dropout_prob'] = 0.2

# Task-related
config['max_way'] = 5
config['min_way'] = 5
config['k_shot'] = 1
config['v_shot'] = 15

# Training related parameters
config['network_architecture'] = 'CNN' # either CNN or ResNet18 specified in the CommonModels.py
config['batchnorm'] = False
config['num_inner_updates'] = 5
config['inner_lr'] = 0.1
config['meta_lr'] = 1e-3
config['minibatch'] = 20 # mini-batch of tasks
config['minibatch_print'] = np.lcm(config['minibatch'], 500)
config['num_episodes_per_epoch'] = 10000 # save model after every xx tasks
config['num_epochs'] = 1
config['resume_epoch'] = 0
# config['train_flag'] = True

# Testing
config['num_episodes'] = 100
config['episode_file'] = None # path to a csv file with row as episode name and column as list of classes that form an episode

# Log
config['logdir'] = os.path.join('/media/n10/Data', 'meta_learning', config['ml_algorithm'], config['datasource'], config['network_architecture'])

# --------------------------------------------------
# Meta-learning class
# --------------------------------------------------
class MLBaseClass(object):
    """Meta-learning class for MAML and VAMPIRE
    """
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __init__(self, config: dict = config) -> None:
        """Initialize an instance of a meta-learning algorithm
        """
        self.config = config
        return
    
    @abc.abstractmethod
    def load_model(self, resume_epoch: int, eps_dataloader: torch.utils.data.DataLoader, **kwargs) -> dict:
        """Load the model for meta-learning algorithm
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def adaptation(self, x: torch.Tensor, y: torch.Tensor, model: dict) -> typing.Union[higher.patch._MonkeyPatchBase, torch.Tensor]:
        """Task adaptation step that produces a task-specific model
        Args:
            x: training data of a task
            y: training labels of that task
            model: a dictionary consisting of
                - "hyper_net", "f_base_net", "optimizer" for MAML-like algorithms such as MAML, ABML, VAMPIRE
                - "protonet", "optimizer" for Prototypical Networks
        Returns: a task-specific model
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def prediction(self, x: torch.Tensor, adapted_hyper_net: typing.Union[torch.Tensor, higher.patch._MonkeyPatchBase], model: dict) -> typing.Union[torch.Tensor, typing.List[torch.Tensor]]:
        """Calculate logits of data

        Args:
            x: data of a task
            adapted_hyper_net: either the prototypes of classes or the adapted hypernet
            model: dictionary consisting of the model and its optimizer

        Returns: prediction logits of data x
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def validation_loss(self, x: torch.Tensor, y: torch.Tensor, adapted_hyper_net: typing.Union[torch.Tensor, higher.patch._MonkeyPatchBase], model: dict) -> torch.Tensor:
        """Calculate the validation loss to update the meta-paramter

        Args:
            x: data in the validation subset
            y: corresponding labels in the validation subset
            adapted_hyper_net: either the prototypes of classes or the adapted hypernet
            model: dictionary consisting of the model and its optimizer

        Return: loss on the validation subset (might also include some regularization such as KL divergence)
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def evaluation(self, x_t: torch.Tensor, y_t: torch.Tensor, x_v: torch.Tensor, y_v: torch.Tensor, model: dict) -> typing.Tuple[float, float]:
        """Calculate loss and accuracy of the given task

        Args:
            x_t, y_t, x_v, y_v: the data of task
            model:

        Returns: two scalars: loss and accuracy
        """
        raise NotImplementedError()

    def train(self, train_dataloader: torch.utils.data.DataLoader, val_dataloader: typing.Optional[torch.utils.data.DataLoader]) -> None:
        """Train meta-learning model

        Args:
            eps_dataloader: the generator that generate episodes/tasks
        """
        print('Training is started.\nLog is stored at {0:s}.\n'.format(self.config['logdir']))

        # initialize/load model. Please see the load_model method implemented in each specific class for further information about the model
        model = self.load_model(resume_epoch=self.config['resume_epoch'], hyper_net_class=self.hyper_net_class, eps_dataloader=train_dataloader)
        model["optimizer"].zero_grad()

        # initialize a tensorboard summary writer for logging
        # tb_writer = SummaryWriter(
        #     log_dir=self.config['logdir'],
        #     purge_step=self.config['resume_epoch'] * self.config['num_episodes_per_epoch'] // self.config['minibatch_print'] if self.config['resume_epoch'] > 0 else None
        # )

        try:
            for epoch_id in range(self.config['resume_epoch'], self.config['resume_epoch'] + self.config['num_epochs'], 1):
                loss_monitor = 0.
                for eps_count, eps_data in enumerate(train_dataloader):

                    if (eps_count >= self.config['num_episodes_per_epoch']):
                        break

                    # split data into train and validation
                    split_data = self.config['train_val_split_function'](eps_data=eps_data, k_shot=self.config['k_shot'])

                    # move data to GPU (if there is a GPU)
                    x_t = split_data['x_t'].to(self.config['device'])
                    y_t = split_data['y_t'].to(self.config['device'])
                    x_v = split_data['x_v'].to(self.config['device'])
                    y_v = split_data['y_v'].to(self.config['device'])

                    # -------------------------
                    # adaptation on training subset
                    # -------------------------
                    adapted_hyper_net = self.adaptation(x=x_t, y=y_t, model=model)

                    # -------------------------
                    # loss on validation subset
                    # -------------------------
                    loss_v = self.validation_loss(x=x_v, y=y_v, adapted_hyper_net=adapted_hyper_net, model=model)
                    loss_v = loss_v / self.config["minibatch"]

                    if torch.isnan(input=loss_v):
                        raise ValueError("Loss is NaN.")

                    # calculate gradients w.r.t. hyper_net's parameters
                    loss_v.backward()

                    loss_monitor += loss_v.item()

                    # update meta-parameters
                    if ((eps_count + 1) % self.config['minibatch'] == 0):

                        model["optimizer"].step()
                        model["optimizer"].zero_grad()

                        # monitoring
                        if (eps_count + 1) % self.config['minibatch_print'] == 0:
                            loss_monitor = loss_monitor * self.config["minibatch"] / self.config["minibatch_print"]

                            # calculate step for Tensorboard Summary Writer
                            global_step = (epoch_id * self.config['num_episodes_per_epoch'] + eps_count + 1) // self.config['minibatch_print']

                            #tb_writer.add_scalar(tag="Train_Loss", scalar_value=loss_monitor, global_step=global_step)

                            # reset monitoring variables
                            loss_monitor = 0.

                            # -------------------------
                            # Validation
                            # -------------------------
                            if val_dataloader is not None:
                                # turn on EVAL mode to disable dropout
                                model["f_base_net"].eval()

                                loss_temp, accuracy_temp = self.evaluate(
                                    num_eps=self.config['num_episodes'],
                                    eps_dataloader=val_dataloader,
                                    model=model
                                )

                                # tb_writer.add_scalar(tag="Val_NLL", scalar_value=np.mean(loss_temp), global_step=global_step)
                                # tb_writer.add_scalar(tag="Val_Accuracy", scalar_value=np.mean(accuracy_temp), global_step=global_step)

                                model["f_base_net"].train()
                                del loss_temp
                                del accuracy_temp

                # save model
                checkpoint = {
                    "hyper_net_state_dict": model["hyper_net"].state_dict(),
                    "opt_state_dict": model["optimizer"].state_dict()
                }
                checkpoint_path = os.path.join(self.config['logdir'], 'Epoch_{0:d}.pt'.format(epoch_id + 1))
                torch.save(obj=checkpoint, f=checkpoint_path)
                print('State dictionaries are saved into {0:s}\n'.format(checkpoint_path))

            print('Training is completed.')
        finally:
            print('\nClose tensorboard summary writer')
            # tb_writer.close()

        return None

    def evaluate(self, num_eps: int, eps_dataloader: torch.utils.data.DataLoader, model: dict) -> typing.Tuple[typing.List[float], typing.List[float]]:
        """Calculate loss and accuracy of tasks contained in the list 'eps'

        Args:
            num_eps: number of episodes to test
            eps_dataloader: receive an eps_name and output the data of that task
            model: a dictionary

        Returns: two lists: loss and accuracy
        """
        loss = [None] * num_eps
        accuracy = [None] * num_eps

        for eps_id, eps_data in enumerate(eps_dataloader):
            if eps_id >= num_eps:
                break

            # split data into train and validation
            split_data = self.config['train_val_split_function'](eps_data=eps_data, k_shot=self.config['k_shot'])

            # move data to GPU (if there is a GPU)
            x_t = split_data['x_t'].to(self.config['device'])
            y_t = split_data['y_t'].to(self.config['device'])
            x_v = split_data['x_v'].to(self.config['device'])
            y_v = split_data['y_v'].to(self.config['device'])

            loss[eps_id], accuracy[eps_id] = self.evaluation(x_t=x_t, y_t=y_t, x_v=x_v, y_v=y_v, model=model)

        return loss, accuracy
            

    def test(self, num_eps: int, eps_dataloader: torch.utils.data.DataLoader) -> None:
        """Evaluate the performance
        """
        print("Evaluation is started.\n")

        model = self.load_model(resume_epoch=self.config["resume_epoch"], hyper_net_class=self.hyper_net_class, eps_dataloader=eps_dataloader)

        loss, accuracy = self.evaluate(num_eps=num_eps, eps_dataloader=eps_dataloader, model=model)

        print('NLL = {0} +/- {1}'.format(np.mean(loss), 1.96 * np.std(loss) / np.sqrt(len(loss))))
        print("Accuracy = {0:.2f} +/- {1:.2f}\n".format(np.mean(accuracy), 1.96 * np.std(accuracy) / np.sqrt(len(accuracy))))
        return None