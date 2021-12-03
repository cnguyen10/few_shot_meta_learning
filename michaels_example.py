"""
Run the MAML benchmark.
"""

import os

import numpy as np
import torch
import wandb
from matplotlib import pyplot as plt
from mlp.mlp import MultiLayerPerceptron
from mtutils.mtutils import norm_area_under_curve
from mtutils.mtutils import print_headline_string as prinths

from mtbnn.plotting import plot_metrics
from mtmlp.plotting import plot_predictions

from src.own.create_benchmarks import create_extracted_benchmarks
from src.own.train_maml import train_maml
from src.own.evaluate_maml import evaluate_maml

def run_experiment(
    config,
    wandb_run,
):
    # define metrics for wandb logging
    wandb_run.define_metric(name="meta_train/epoch")
    wandb_run.define_metric(name="meta_train/*",
                            step_metric="meta_train/epoch")
    wandb_run.define_metric(name="adapt/epoch")
    wandb_run.define_metric(name="adapt/*", step_metric="adapt/epoch")
    wandb_run.define_metric(name="eval/n_context")
    wandb_run.define_metric(name="eval/*", step_metric="eval/n_context")

    # seeding
    torch.manual_seed(config["seed"])

    # create benchmarks
    x_meta, y_meta, x_test, y_test, x_pred_meta, x_pred_test = create_extracted_benchmarks(
        config)

    # create model
    model = MultiLayerPerceptron(
        d_x=x_meta.shape[2],
        d_y=y_meta.shape[2],
        hidden_units=config["hidden_units"],
        f_act=config["f_act"],
    )

    # meta training
    prinths("Performing Meta Training...")
    learning_curve_meta = train_maml(
        model=model,
        x_meta=x_meta,
        y_meta=y_meta,
        n_context=config["n_context_meta"],
        n_epochs=config["n_epochs"],
        n_adapt_steps=config["n_adapt_steps"],
        adapt_lr=config["adapt_lr"],
        meta_lr=config["meta_lr"],
        wandb_run=wandb_run,
    )

    # evaluate on meta tasks (for n_context = n_context_meta)
    rmse_meta, rmse_context_meta, y_pred_meta = evaluate_maml(
        model=model,
        x=x_meta,
        y=y_meta,
        x_pred=x_pred_meta,
        n_context=config["n_context_meta"],
        n_adapt_steps=config["n_adapt_steps"],
        adapt_lr=config["adapt_lr"],
    )
    y_preds_meta = y_pred_meta[None, :, :]

    # evaluate on test tasks (for varying n_context)
    rmses_test = np.zeros(len(config["n_contexts_pred"]))
    rmses_context_test = np.zeros(len(config["n_contexts_pred"]))
    y_preds_test = []
    for i, n_context in enumerate(config["n_contexts_pred"]):
        print(f"Adapting to tasks (n_context = {n_context:3d})...")

        cur_rmse_test, cur_rmse_context_test, cur_y_pred_test = evaluate_maml(
            model=model,
            x=x_test,
            y=y_test,
            x_pred=x_pred_test,
            n_context=n_context,
            n_adapt_steps=config["n_adapt_steps"],
            adapt_lr=config["adapt_lr"],
        )

        # log
        rmses_test[i] = cur_rmse_test
        rmses_context_test[i] = cur_rmse_context_test
        y_preds_test.append(cur_y_pred_test)
        wandb_run.log(
            {
                "eval/n_context": n_context,
                "eval/rmse": rmses_test[i],
                "eval/rmse_context": rmses_context_test[i],
            }
        )
    y_preds_test = np.stack(y_preds_test, axis=0)

    # log summaries
    wandb_run.summary["meta_train/rmse_meta_n_ctx_meta"] = rmse_meta
    wandb_run.summary["meta_train/rmse_context_meta_n_ctx_meta"] = rmse_context_meta
    wandb_run.summary["eval/rmse_test_n_ctx_meta"] = rmses_test[
        config["n_contexts_pred"].index(config["n_context_meta"])
    ]
    wandb_run.summary["eval/rmse_context_test_n_ctx_meta"] = rmses_context_test[
        config["n_contexts_pred"].index(config["n_context_meta"])
    ]
    wandb_run.summary["eval/rmse_test_auc"] = norm_area_under_curve(
        x=config["n_contexts_pred"], y=rmses_test
    )
    wandb_run.summary["eval/rmse_context_test_auc"] = norm_area_under_curve(
        x=config["n_contexts_pred"][1:], y=rmses_context_test[1:]
    )

    # plot predictions
    if config["plot"]:
        fig = plot_metrics(
            learning_curve_meta=learning_curve_meta,
            learning_curves_test=None,
            rmses=rmses_test,
            rmses_context=rmses_context_test,
            n_contexts=config["n_contexts_pred"],
        )
        fig = plot_predictions(
            x=x_meta,
            y=y_meta,
            x_pred=x_pred_meta,
            y_preds=y_preds_meta,
            n_contexts=[config["n_context_meta"]],
            max_tasks=config["max_tasks_plot"],
            n_contexts_plot=[config["n_context_meta"]],
            dataset_name="meta",
        )
        wandb_run.log({"predictions_meta_png": wandb.Image(fig)})
        fig = plot_predictions(
            x=x_test,
            y=y_test,
            x_pred=x_pred_test,
            y_preds=y_preds_test,
            n_contexts=config["n_contexts_pred"],
            max_tasks=config["max_tasks_plot"],
            n_contexts_plot=config["n_contexts_plot"],
            dataset_name="test",
        )
        wandb_run.log({"predictions_test_png": wandb.Image(fig)})

        if wandb_run.mode == "disabled":
            plt.show()


def main():
    # config
    wandb_mode = os.getenv("WANDB_MODE", "online")
    smoke_test = os.getenv("SMOKE_TEST", "False") == "True"
    print(f"wandb_mode={wandb_mode}")
    print(f"smoke_test={smoke_test}")
    config = dict(
        model="MAML",
        seed=123,
        # benchmarks
        bm="Affine1D",
        noise_stddev=0.01,
        n_tasks_meta=8,
        n_points_per_task_meta=16,
        n_tasks_test=128,
        n_points_per_task_test=128,
        seed_offset_train=1234,
        seed_offset_test=1235,
        normalize_bm=True,
        # model
        hidden_units=[8],
        f_act="relu",
        # training
        n_epochs=500 if not smoke_test else 100,
        adapt_lr=0.01,
        meta_lr=0.001,
        n_adapt_steps=5,
        n_context_meta=8,
        # evaluation
        n_points_pred=100,
        n_contexts_pred=(
            [0, 1, 2, 5, 8, 10, 15, 20, 30, 40, 50, 75, 100, 128]
            if not smoke_test
            else [0, 5, 8, 10, 50, 128]
        ),
        # plot
        plot=True,
        max_tasks_plot=4,
        n_contexts_plot=[0, 5, 8, 10, 50],
    )

    if wandb_mode != "disabled":
        wandb.login()
    with wandb.init(project="mtbnn_v0", mode=wandb_mode, config=config) as wandb_run:
        config = wandb_run.config
        run_experiment(config=config, wandb_run=wandb_run)


main()
