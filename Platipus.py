"""
Training PLATIPUS is quite time-consuming. One might need to train MAML, then load such paramters obtained from MAML as mu_theta to speed up the training of PLATIPUS.
"""

import torch
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import os
import random
import typing


from HyperNetClasses import PlatipusNet
from Maml import Maml
from EpisodeGenerator import OmniglotLoader, ImageFolderGenerator
from _utils import kl_divergence_gaussians, train_val_split, get_episodes

class Platipus(object):
    def __init__(self, config: dict) -> None:
        self.config = config

        self.hyper_net_class = PlatipusNet

    def load_model(self, resume_epoch: int, **kwargs) -> dict:
        maml_temp = Maml(config=self.config)
        return maml_temp.load_model(resume_epoch=resume_epoch, **kwargs)
    
    def adapt_params(self, x: torch.Tensor, y: torch.Tensor, params: typing.List[torch.Tensor], lr: torch.Tensor, model: dict) -> typing.List[torch.Tensor]:
        q_params = [p + 0. for p in params]
        
        for _ in range(self.config["num_inner_updates"]):
            # predict output logits
            logits = model["f_base_net"].forward(x, params=q_params)

            # calculate classification loss
            loss = torch.nn.functional.cross_entropy(input=logits, target=y)

            if self.config["first_order"]:
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

            for i in range(len(q_params)):
                q_params[i] = q_params[i] - lr * grads[i]
        
        return q_params
    
    def adaptation(self, x: torch.Tensor, y: torch.Tensor, model: dict) -> typing.List[typing.List[torch.Tensor]]:
        """Correspond to Algorithm 2 for testing
        """
        # initialize phi
        phi = [None] * self.config["num_models"]

        # get meta-parameters
        params_dict = model["hyper_net"].forward()

        # step 1 - prior distribution
        mu_theta_t = self.adapt_params(x=x, y=y, params=params_dict["mu_theta"], lr=params_dict["gamma_p"], model=model)

        for model_id in range(self.config["num_models"]):
            # sample theta
            theta = [None] * len(params_dict["mu_theta"])
            for i in range(len(theta)):
                theta[i] = mu_theta_t[i] + torch.randn_like(input=mu_theta_t[i], device=mu_theta_t[i].device) * torch.exp(input=params_dict["log_sigma_theta"][i])
            
            phi[model_id] = self.adapt_params(x=x, y=y, params=theta, lr=self.config["inner_lr"], model=model)

        return phi

    def prediction(self, x: torch.Tensor, phi: typing.List[typing.List[torch.Tensor]], model: dict) -> typing.List[torch.Tensor]:
        logits = [None] * self.config["num_models"]
        for model_id in range(self.config["num_models"]):
            logits[model_id] = model["f_base_net"].forward(x, params=phi[model_id])

        return logits
    
    def validation_loss(self, x_t: torch.Tensor, y_t: torch.Tensor, x_v: torch.Tensor, y_v: torch.Tensor, model: dict) -> torch.Tensor:
        params_dict = model["hyper_net"].forward()

        # adapt mu_theta - step 7 in PLATIPUS paper
        mu_theta_v = self.adapt_params(x=x_v, y=y_v, params=params_dict["mu_theta"], lr=params_dict["gamma_q"], model=model)

        phi = [None] * self.config["num_models"]
        for model_id in range(self.config["num_models"]):
            # step 7: sample theta from N(mu_theta, v_q^2)
            theta = [None] * len(params_dict["mu_theta"])
            for i in range(len(theta)):
                theta[i] = mu_theta_v[i] + \
                    torch.randn_like(input=mu_theta_v[i], device=mu_theta_v[i].device) * torch.exp(input=params_dict["log_v_q"][i])
            
            # steps 8 and 9
            phi[model_id] = self.adapt_params(x=x_t, y=y_t, params=theta, lr=self.config["inner_lr"], model=model)

        # step 10 - adapt mu_theta to training subset
        mu_theta_t = self.adapt_params(x=x_t, y=y_t, params=params_dict["mu_theta"], lr=params_dict["gamma_p"], model=model)

        # step 11 - validation loss
        loss = 0
        for i in range(len(phi)):
            logits = model["f_base_net"].forward(x_v, params=phi[i])
            loss_temp = torch.nn.functional.cross_entropy(input=logits, target=y_v)
            loss = loss + loss_temp
        
        loss = loss / len(phi)

        # KL loss
        KL_loss = kl_divergence_gaussians(p=[*mu_theta_v, *params_dict["log_v_q"]], q=[*mu_theta_t, *params_dict["log_sigma_theta"]])

        loss = loss + self.config["KL_weight"] * KL_loss

        return loss
    
    def evaluation(self, x_t: torch.Tensor, y_t: torch.Tensor, x_v: torch.Tensor, y_v: torch.Tensor, model: dict) -> typing.Tuple[float, float]:
        phi = self.adaptation(x=x_t, y=y_t, model=model)

        logits = self.prediction(x=x_v, phi=phi, model=model)

        # classification loss
        loss = 0
        for logits_ in logits:
            loss = loss + torch.nn.functional.cross_entropy(input=logits_, target=y_v)
        
        loss = loss / len(logits)

        y_pred = 0
        for logits_ in logits:
            y_pred = y_pred + torch.softmax(input=logits_, dim=1)
        
        y_pred = y_pred / len(logits)

        accuracy = (y_pred.argmax(dim=1) == y_v).float().mean().item()

        return loss.item(), accuracy * 100

    def train(self, eps_generator: typing.Union[OmniglotLoader, ImageFolderGenerator], **kwargs) -> None:
        """Train meta-learning model

        Args:
            eps_generator: the generator that generate episodes/tasks
        """
        print("Training is started.\nLog is stored at {0:s}.\n".format(self.config["logdir"]))

        # initialize/load model. Please see the load_model method implemented in each specific class for further information about the model
        model = self.load_model(resume_epoch=self.config["resume_epoch"], hyper_net_class=self.hyper_net_class, eps_generator=eps_generator)
        model["optimizer"].zero_grad()

        # get list of episode names, each episode name consists of classes
        # if no episode_file is specified, it will consist of None - corresponding to random task
        eps = get_episodes(episode_file_path=self.config["episode_file"])

        # initialize a tensorboard summary writer for logging
        tb_writer = SummaryWriter(
            log_dir=self.config["logdir"],
            purge_step=self.config["resume_epoch"] * self.config["num_episodes_per_epoch"] // self.config["minibatch_print"] if self.config["resume_epoch"] > 0 else None
        )

        try:
            for epoch_id in range(self.config["resume_epoch"], self.config["resume_epoch"] + self.config["num_epochs"], 1):
                loss_monitor = 0.
                for eps_count in range(self.config["num_episodes_per_epoch"]):
                    # -------------------------
                    # get eps from the given csv file or just random (None)
                    # -------------------------
                    eps_name = random.sample(population=eps, k=1)[0]

                    # -------------------------
                    # episode data
                    # -------------------------
                    eps_data = eps_generator.generate_episode(episode_name=eps_name)

                    # split data into train and validation
                    xt, yt, xv, yv = train_val_split(X=eps_data, k_shot=self.config["k_shot"], shuffle=True)

                    # move data to GPU (if there is a GPU)
                    x_t = torch.from_numpy(xt).float().to(self.config["device"])
                    y_t = torch.tensor(yt, dtype=torch.long, device=self.config["device"])
                    x_v = torch.from_numpy(xv).float().to(self.config["device"])
                    y_v = torch.tensor(yv, dtype=torch.long, device=self.config["device"])

                    # -------------------------
                    # loss on validation subset
                    # -------------------------
                    loss_v = self.validation_loss(x_t=x_t, y_t=y_t, x_v=x_v, y_v=y_v, model=model)
                    loss_v = loss_v / self.config["minibatch"]

                    if torch.isnan(input=loss_v):
                        raise ValueError("Loss is NaN.")

                    # calculate gradients w.r.t. hyper_net"s parameters
                    loss_v.backward()

                    loss_monitor += loss_v.item()

                    # update meta-parameters
                    if ((eps_count + 1) % self.config["minibatch"] == 0):

                        model["optimizer"].step()
                        model["optimizer"].zero_grad()

                        # monitoring
                        if (eps_count + 1) % self.config["minibatch_print"] == 0:
                            loss_monitor = loss_monitor * self.config["minibatch"] / self.config["minibatch_print"]

                            # calculate step for Tensorboard Summary Writer
                            global_step = (epoch_id * self.config["num_episodes_per_epoch"] + eps_count + 1) // self.config["minibatch_print"]

                            tb_writer.add_scalar(tag="Train_Loss", scalar_value=loss_monitor, global_step=global_step)

                            # reset monitoring variables
                            loss_monitor = 0.

                            # -------------------------
                            # Validation
                            # -------------------------
                            if ("eps_generator_val" in kwargs):
                                loss_temp, accuracy_temp = self.evaluate(
                                    eps=[None] * self.config["num_episodes"],
                                    eps_generator=kwargs["eps_generator_val"],
                                    model=model
                                )

                                tb_writer.add_scalar(tag="Val_NLL", scalar_value=np.mean(loss_temp), global_step=global_step)
                                tb_writer.add_scalar(tag="Val_Accuracy", scalar_value=np.mean(accuracy_temp), global_step=global_step)

                                del loss_temp
                                del accuracy_temp

                # save model
                checkpoint = {
                    "hyper_net_state_dict": model["hyper_net"].state_dict(),
                    "opt_state_dict": model["optimizer"].state_dict()
                }
                checkpoint_path = os.path.join(self.config["logdir"], "Epoch_{0:d}.pt".format(epoch_id + 1))
                torch.save(obj=checkpoint, f=checkpoint_path)
                print("State dictionaries are saved into {0:s}\n".format(checkpoint_path))

            print("Training is completed.")
        finally:
            print("\nClose tensorboard summary writer")
            tb_writer.close()

        return None

    def evaluate(self, eps: typing.Optional[typing.List[str]], eps_generator: typing.Union[OmniglotLoader, ImageFolderGenerator], model: dict) -> typing.Tuple[typing.List[float], typing.List[float]]:
        """Calculate loss and accuracy of tasks contained in the list "eps"

        Args:
            eps: a list of task names (list of strings) or a list of None for random tasks
            eps_generator: receive an eps_name and output the data of that task
            model: a dictionary

        Returns: two lists: loss and accuracy
        """
        loss = [None] * len(eps)
        accuracy = [None] * len(eps)

        for eps_id, eps_name in enumerate(eps):
            # -------------------------
            # episode data
            # -------------------------
            eps_data = eps_generator.generate_episode(episode_name=eps_name)

            # split data into train and validation
            xt, yt, xv, yv = train_val_split(X=eps_data, k_shot=self.config["k_shot"], shuffle=True)

            # move data to GPU (if there is a GPU)
            x_t = torch.from_numpy(xt).float().to(self.config["device"])
            y_t = torch.tensor(yt, dtype=torch.long, device=self.config["device"])
            x_v = torch.from_numpy(xv).float().to(self.config["device"])
            y_v = torch.tensor(yv, dtype=torch.long, device=self.config["device"])

            loss[eps_id], accuracy[eps_id] = self.evaluation(x_t=x_t, y_t=y_t, x_v=x_v, y_v=y_v, model=model)

        return loss, accuracy
            

    def test(self, eps_generator: typing.Union[OmniglotLoader, ImageFolderGenerator]) -> None:
        """Evaluate the performance
        """
        print("Evaluation is started.\n")

        model = self.load_model(resume_epoch=self.config["resume_epoch"], hyper_net_class=self.hyper_net_class, eps_generator=eps_generator)

        # get list of episode names, each episode name consists of classes
        eps = get_episodes(episode_file_path=self.config["episode_file"])

        _, accuracy = self.evaluate(eps=eps, eps_generator=eps_generator, model=model)

        print("Accuracy = {0:.2f} +/- {1:.2f}\n".format(np.mean(accuracy), 1.96 * np.std(accuracy) / np.sqrt(len(accuracy))))
        return None