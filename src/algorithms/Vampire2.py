import torch
import higher
import typing

from src.HyperNetClasses import NormalVariationalNet
from src.algorithms.MLBaseClass import MLBaseClass
from src.algorithms.Maml import Maml

class Vampire2(MLBaseClass):
    def __init__(self, config: dict) -> None:
        super().__init__(config=config)

        self.hyper_net_class = NormalVariationalNet

    def load_model(self, resume_epoch: int, eps_dataloader: torch.utils.data.DataLoader, **kwargs) -> dict:
        maml_temp = Maml(config=self.config)
        return maml_temp.load_model(resume_epoch=resume_epoch, eps_dataloader=eps_dataloader, **kwargs)

    def adaptation(self, x: torch.Tensor, y: torch.Tensor, model: dict) -> higher.patch._MonkeyPatchBase:
        # convert hyper_net to its functional form
        f_hyper_net = higher.patch.monkeypatch(
            module=model["hyper_net"],
            copy_initial_weights=False,
            track_higher_grads=self.config["train_flag"]
        )

        for _ in range(self.config['num_inner_updates']):
            q_params = f_hyper_net.fast_params # parameters of the task-specific hyper_net

            grads_accum = [0] * len(q_params) # accumulate gradients of Monte Carlo sampling

            # KL divergence
            KL_loss = self.KL_divergence_standard_normal(p=q_params)

            for _ in range(self.config['num_models']):
                # generate parameter from task-specific hypernet
                base_net_params = f_hyper_net.forward()

                y_logits = model["f_base_net"].forward(x, params=base_net_params)

                cls_loss = self.config['loss_function'](input=y_logits, target=y)

                loss = cls_loss + self.config['KL_weight'] * KL_loss

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

                # accumulate gradients from Monte Carlo sampling and average out
                for i in range(len(grads)):
                    grads_accum[i] = grads_accum[i] + grads[i] / self.config['num_models']

            new_q_params = []
            for param, grad in zip(q_params, grads_accum):
                new_q_params.append(higher.optim._add(tensor=param, a1=-self.config['inner_lr'], a2=grad))

            f_hyper_net.update_params(new_q_params)

        return f_hyper_net

    def prediction(self, x: torch.Tensor, adapted_hyper_net: higher.patch._MonkeyPatchBase, model: dict) -> typing.List[torch.Tensor]:

        logits = [None] * self.config['num_models']
        for model_id in range(self.config['num_models']):
            # generate parameter from task-specific hypernet
            base_net_params = adapted_hyper_net.forward()

            logits_temp = model["f_base_net"].forward(x, params=base_net_params)

            logits[model_id] = logits_temp

        return logits

    def validation_loss(self, x: torch.Tensor, y: torch.Tensor, adapted_hyper_net: higher.patch._MonkeyPatchBase, model: dict) -> torch.Tensor:

        logits = self.prediction(x=x, adapted_hyper_net=adapted_hyper_net, model=model)

        loss = 0

        # classification loss
        for logits_ in logits:
            loss = loss + self.config['loss_function'](input=logits_, target=y)
        
        loss = loss / len(logits)

        # KL loss
        KL_loss = self.KL_divergence_standard_normal(p=adapted_hyper_net.fast_params)
        loss = loss + self.config["KL_weight"] * KL_loss

        return loss

    def evaluation(self, x_t: torch.Tensor, y_t: torch.Tensor, x_v: torch.Tensor, y_v: torch.Tensor, model: dict) -> typing.Tuple[float, float]:
        
        adapted_hyper_net = self.adaptation(x=x_t, y=y_t, model=model)

        logits = self.prediction(x=x_v, adapted_hyper_net=adapted_hyper_net, model=model)

        loss = self.validation_loss(x=x_v, y=y_v, adapted_hyper_net=adapted_hyper_net, model=model)

        y_pred = 0
        for logits_ in logits:
            y_pred = y_pred + torch.softmax(input=logits_, dim=1)
        
        y_pred = y_pred / len(logits)

        accuracy = (y_pred.argmax(dim=1) == y_v).float().mean().item()

        return loss.item(), accuracy * 100

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