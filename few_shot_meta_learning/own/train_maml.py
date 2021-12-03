import torch
import learn2learn as l2l
from mtutils.mtutils import split_tasks

def train_maml(
    model, x_meta, y_meta, n_context, n_epochs, n_adapt_steps, adapt_lr, meta_lr, wandb_run
):
    """
    Train model using MAML on data x_meta, y_meta using n_context context points.
    """
    maml = l2l.algorithms.MAML(
        model,
        lr=adapt_lr,
        first_order=False,
        allow_unused=True,
    )
    optim = torch.optim.Adam(maml.parameters(), meta_lr)
    loss_fn = torch.nn.MSELoss(reduction="mean")
    n_tasks = x_meta.shape[0]

    # divide the data into context and target sets
    x_context, y_context, x_target, y_target = split_tasks(
        x=x_meta,
        y=y_meta,
        n_context=n_context,
    )
    x_context, y_context, x_target, y_target = (
        torch.tensor(x_context, dtype=torch.float32),
        torch.tensor(y_context, dtype=torch.float32),
        torch.tensor(x_target, dtype=torch.float32),
        torch.tensor(y_target, dtype=torch.float32),
    )

    # for each iteration
    learning_curve_meta = []
    for i in range(n_epochs):
        meta_train_loss = 0.0

        # for each task in the batch
        for l in range(n_tasks):
            learner = maml.clone()

            for _ in range(n_adapt_steps):  # adaptation_steps
                context_preds = learner(x_context[l: l + 1])
                context_loss = loss_fn(context_preds, y_context[l: l + 1])
                learner.adapt(context_loss)

            target_preds = learner(x_target[l: l + 1])
            target_loss = loss_fn(target_preds, y_target[l: l + 1])
            meta_train_loss += target_loss

        meta_train_loss = meta_train_loss / n_tasks
        learning_curve_meta.append(meta_train_loss.item())

        # log
        wandb_run.log(
            {
                f"meta_train/epoch": i,
                f"meta_train/loss_n_context_{n_context:03d}": meta_train_loss.item(),
            }
        )
        if i % 100 == 0 or i == n_epochs - 1:
            print(
                f"Epoch = {i:04d} | Meta Train Loss = {meta_train_loss.item():.4f}")

        optim.zero_grad()
        meta_train_loss.backward()
        optim.step()

    return learning_curve_meta