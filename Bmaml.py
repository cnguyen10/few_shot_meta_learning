import torch
import numpy as np
import higher
import typing

from MLBaseClass import MLBaseClass
from HyperNetClasses import EnsembleNet
from Maml import Maml

class Bmaml(MLBaseClass):
    def __init__(self, config: dict) -> None:
        super().__init__(config=config)
        
        self.hyper_net_class = EnsembleNet

    def load_model(self, resume_epoch: int, eps_dataloader: torch.utils.data.DataLoader, **kwargs) -> dict:
        maml_temp = Maml(config=self.config)
        return maml_temp.load_model(resume_epoch=resume_epoch, eps_dataloader=eps_dataloader, **kwargs)

    def adaptation(self, x: torch.Tensor, y: torch.Tensor, model: dict) -> typing.List[higher.patch._MonkeyPatchBase]:
        """"""
        f_hyper_net = higher.patch.monkeypatch(
            module=model["hyper_net"],
            copy_initial_weights=False,
            track_higher_grads=self.config["train_flag"]
        )

        q_params = torch.stack(tensors=[p for p in model["hyper_net"].parameters()])

        for _ in range(self.config["num_inner_updates"]):
            distance_NLL = torch.empty(size=(self.config["num_models"], model["hyper_net"].num_base_params), device=self.config["device"])

            for particle_id in range(self.config["num_models"]):
                base_net_params = f_hyper_net.forward(i=particle_id)
                
                logits = model["f_base_net"].forward(x, params=base_net_params)

                loss_temp = self.config['loss_function'](input=logits, target=y)

                if self.config["first_order"]:
                    grads = torch.autograd.grad(
                        outputs=loss_temp,
                        inputs=f_hyper_net.fast_params[particle_id],
                        retain_graph=True
                    )
                else:
                    grads = torch.autograd.grad(
                        outputs=loss_temp,
                        inputs=f_hyper_net.fast_params[particle_id],
                        create_graph=True
                    )
                
                distance_NLL[particle_id, :] = torch.nn.utils.parameters_to_vector(parameters=grads)

            kernel_matrix, grad_kernel, _ = self.get_kernel(params=q_params)

            q_params = q_params - self.config["inner_lr"] * (torch.matmul(kernel_matrix, distance_NLL) - grad_kernel)

            # update hyper-net
            f_hyper_net.update_params(params=[q_params[i, :] for i in range(self.config["num_models"])])
        
        return f_hyper_net

    def prediction(self, x: torch.Tensor, adapted_hyper_net: higher.patch._MonkeyPatchBase, model: dict) -> typing.List[torch.Tensor]:
        """"""
        logits = [None] * self.config["num_models"]

        for particle_id in range(self.config["num_models"]):
            base_net_params = adapted_hyper_net.forward(i=particle_id)

            logits[particle_id] = model["f_base_net"].forward(x, params=base_net_params)
        
        return logits

    def validation_loss(self, x: torch.Tensor, y: torch.Tensor, adapted_hyper_net: higher.patch._MonkeyPatchBase, model: dict) -> torch.Tensor:
        """"""
        logits = self.prediction(x=x, adapted_hyper_net=adapted_hyper_net, model=model)

        loss = 0

        for logits_ in logits:
            loss_temp = self.config['loss_function'](input=logits_, target=y)
            loss = loss + loss_temp
        
        loss = loss / len(logits)

        return loss

    def evaluation(self, x_t: torch.Tensor, y_t: torch.Tensor, x_v: torch.Tensor, y_v: torch.Tensor, model: dict) -> typing.Tuple[float, float]:
        """
        """
        adapted_hyper_net = self.adaptation(x=x_t, y=y_t, model=model)

        logits = self.prediction(x=x_v, adapted_hyper_net=adapted_hyper_net, model=model)

        # classification loss
        loss = 0
        for logits_ in logits:
            loss = loss + self.config['loss_function'](input=logits_, target=y_v)
        
        loss = loss / len(logits)

        y_pred = 0
        for logits_ in logits:
            y_pred = y_pred + torch.softmax(input=logits_, dim=1)
        
        y_pred = y_pred / len(logits)

        accuracy = (y_pred.argmax(dim=1) == y_v).float().mean().item()

        return loss.item(), accuracy * 100

    def get_kernel(self, params: torch.Tensor) -> typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute the RBF kernel for the input
        
        Args:
            params: a tensor of shape (N, M)
        
        Returns: kernel_matrix = tensor of shape (N, N)
        """
        pairwise_d_matrix = self.get_pairwise_distance_matrix(x=params)

        median_dist = torch.quantile(input=pairwise_d_matrix, q=0.5)  # tf.reduce_mean(euclidean_dists) ** 2
        h = median_dist / np.log(self.config["num_models"])

        kernel_matrix = torch.exp(-pairwise_d_matrix / h)
        kernel_sum = torch.sum(input=kernel_matrix, dim=1, keepdim=True)
        grad_kernel = -torch.matmul(kernel_matrix, params)
        grad_kernel += params * kernel_sum
        grad_kernel /= h

        return kernel_matrix, grad_kernel, h

    @staticmethod
    def get_pairwise_distance_matrix(x: torch.Tensor) -> torch.Tensor:
        """Calculate the pairwise distance between each row of tensor x
        
        Args:
            x: input tensor
        
        Return: matrix of point-wise distances
        """
        n, m = x.shape

        # initialize matrix of pairwise distances as a N x N matrix
        pairwise_d_matrix = torch.zeros(size=(n, n), device=x.device)

        # num_particles = particle_tensor.shape[0]
        euclidean_dists = torch.nn.functional.pdist(input=x, p=2) # shape of (N)

        # assign upper-triangle part
        triu_indices = torch.triu_indices(row=n, col=n, offset=1)
        pairwise_d_matrix[triu_indices[0], triu_indices[1]] = euclidean_dists

        # assign lower-triangle part
        pairwise_d_matrix = torch.transpose(pairwise_d_matrix, dim0=0, dim1=1)
        pairwise_d_matrix[triu_indices[0], triu_indices[1]] = euclidean_dists

        return pairwise_d_matrix