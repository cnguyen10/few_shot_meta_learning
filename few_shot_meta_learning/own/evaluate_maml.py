import torch
import numpy as np
import learn2learn as l2l
from mtutils.mtutils import split_tasks

def evaluate_maml(model, x, y, n_context, x_pred, n_adapt_steps, adapt_lr):
    """
    Evaluate RMSE of model on data x, y using n_context context_points.
    Also return predictions of adapted model for each task.
    """
    maml = l2l.algorithms.MAML(
        model,
        lr=adapt_lr,
        first_order=False,
        allow_unused=True,
    )
    loss_fn = torch.nn.MSELoss(reduction="mean")
    n_tasks = x.shape[0]

    # divide the data into context and target sets
    x_context, y_context, x_target, y_target = split_tasks(
        x=x,
        y=y,
        n_context=n_context,
    )
    x_context, y_context, x_target, y_target = (
        torch.tensor(x_context, dtype=torch.float32),
        torch.tensor(y_context, dtype=torch.float32),
        torch.tensor(x_target, dtype=torch.float32),
        torch.tensor(y_target, dtype=torch.float32),
    )

    mse = 0.0
    mse_context = 0.0
    cur_y_pred = []
    for l in range(n_tasks):
        learner = maml.clone()

        # adapt
        for _ in range(n_adapt_steps):  # adaptation_steps
            context_preds = learner(x_context[l: l + 1])
            context_loss = loss_fn(context_preds, y_context[l: l + 1])
            learner.adapt(context_loss)

        # predict
        cur_y_pred.append(learner.predict(x=x_pred[l: l + 1]))

        # mse
        mse += learner.mse(x=x_target[l: l + 1].numpy(),
                           y=y_target[l: l + 1].numpy())
        mse_context += learner.mse(
            x=x_context[l: l + 1].numpy(), y=y_context[l: l + 1].numpy()
        )

    y_pred = np.concatenate(cur_y_pred, axis=0)
    rmse = np.sqrt(mse / n_tasks)
    rmse_context = np.sqrt(mse_context / n_tasks)

    return rmse, rmse_context, y_pred