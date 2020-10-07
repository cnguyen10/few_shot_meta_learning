"""
python3 maml_regression.py --k-shot=5 --inner-lr=1e-2 --num-inner-updates=5 --meta-lr=1e-3 --minibatch=100 --decay-lr=0.9 --num-epochs=1 --resume-epoch=0

python3 maml_regression.py --k-shot=5 --inner-lr=1e-2 --num-inner-updates=5 --resume-epoch=1 --test --sine
"""

import torch
from torch.utils.tensorboard import SummaryWriter
import higher

import numpy as np
import os
import random
import sys

from matplotlib import pyplot as plt

import argparse
import typing as _typing

from EpisodeGenerator import SineLineGenerator
from _utils import train_val_split, _weights_init
from CommonModels import FcNet

# --------------------------------------------------
# SETUP INPUT PARSER
# --------------------------------------------------
parser = argparse.ArgumentParser(description='Setup variables')

parser.add_argument('--first-order', type=bool, default=True, help='First order or track higher gradients')

parser.add_argument('--num-inner-updates', type=int, default=5, help='The number of gradient updates for episode adaptation')
parser.add_argument('--inner-lr', type=float, default=1e-2, help='Learning rate of episode adaptation step')

parser.add_argument('--logdir', type=str, default='/media/n10/Data/', help='Folder to store model and logs')

parser.add_argument('--meta-lr', type=float, default=1e-3, help='Learning rate for meta-update')
parser.add_argument('--minibatch', type=int, default=5, help='Minibatch of episodes to update meta-parameters')
parser.add_argument('--decay-lr', type=float, default=1., help='Factor to decay meta learning rate')

parser.add_argument('--k-shot', type=int, default=5, help='Number of training examples per class')
parser.add_argument('--v-shot', type=int, default=95, help='Number of validation examples per class')

parser.add_argument('--num-episodes-per-epoch', type=int, default=10000, help='Save meta-parameters after this number of episodes')
parser.add_argument('--num-epochs', type=int, default=1, help='')
parser.add_argument('--resume-epoch', type=int, default=0, help='Resume')

parser.add_argument('--train', dest='train_flag', action='store_true')
parser.add_argument('--test', dest='train_flag', action='store_false')
parser.set_defaults(train_flag=True)

parser.add_argument('--sine', dest='mode', action='store_true')
parser.add_argument('--line', dest='mode', action='store_false')
parser.set_defaults(mode=True)

args = parser.parse_args()
print()
# --------------------------------------------------
# Setup CPU or GPU
# --------------------------------------------------
gpu_id = 0
device = torch.device('cuda:{0:d}'.format(gpu_id) \
    if torch.cuda.is_available() else torch.device('cpu'))

# --------------------------------------------------
# parse parameters
# --------------------------------------------------
logdir = os.path.join(args.logdir, 'MAML', 'sine_line')
if not os.path.exists(logdir):
    os.makedirs(logdir)

k_shot = args.k_shot
v_shot = args.v_shot

train_flag = args.train_flag

num_inner_updates = args.num_inner_updates
inner_lr = args.inner_lr

resume_epoch = args.resume_epoch

first_order = args.first_order

mode_ = 'sine' if args.mode else 'line'
# --------------------------------------------------
# Data loader
# --------------------------------------------------
eps_generator = SineLineGenerator(num_samples=k_shot + v_shot)

# --------------------------------------------------
# MAIN
# --------------------------------------------------
def main():
    if train_flag:
        train()
    else:
        evaluate()

# --------------------------------------------------
# TRAIN
# --------------------------------------------------
def train() -> None:
    """Train
    
    Args:

    Returns:
    """
    try:
        # parse training parameters
        meta_lr = args.meta_lr
        minibatch = args.minibatch
        minibatch_print = np.lcm(minibatch, 100)
        decay_lr = args.decay_lr

        num_episodes_per_epoch = args.num_episodes_per_epoch
        num_epochs = args.num_epochs

        # initialize/load model
        net, meta_optimizer, schdlr = load_model(epoch_id=resume_epoch, meta_lr=meta_lr, decay_lr=decay_lr)
        
        # zero grad
        meta_optimizer.zero_grad()

        # initialize a tensorboard summary writer for logging
        tb_writer = SummaryWriter(
            log_dir=logdir,
            purge_step=resume_epoch * num_episodes_per_epoch // minibatch_print if resume_epoch > 0 else None
        )

        for epoch_id in range(resume_epoch, resume_epoch + num_epochs, 1):
            episode_count = 0
            loss_monitor = 0
            
            while (episode_count < num_episodes_per_epoch):
                x, y_noisy, _ = eps_generator.generate_episode()
                
                # split into train and validation
                xt = x[:k_shot]
                yt = y_noisy[:k_shot]
                xv = x[k_shot:]
                yv = y_noisy[k_shot:]

                # move data to gpu
                x_t = torch.from_numpy(xt).float().to(device)
                y_t = torch.from_numpy(yt).float().to(device)
                x_v = torch.from_numpy(xv).float().to(device)
                y_v = torch.from_numpy(yv).float().to(device)

                # adapt on the support data
                fnet = adapt_to_episode(x=x_t, y=y_t, net=net)

                # evaluate on the query data
                logits_v = fnet.forward(x_v)
                cls_loss = torch.nn.functional.mse_loss(input=logits_v, target=y_v)
                loss_monitor += cls_loss.item()

                cls_loss = cls_loss / minibatch
                cls_loss.backward()

                episode_count += 1

                # update the meta-model
                if (episode_count % minibatch == 0):
                    meta_optimizer.step()
                    meta_optimizer.zero_grad()

                # monitor losses
                if (episode_count % minibatch_print == 0):
                    loss_monitor /= minibatch_print
                    global_step = (epoch_id * num_episodes_per_epoch + episode_count) // minibatch_print
                    tb_writer.add_scalar(
                        tag='Loss',
                        scalar_value=loss_monitor,
                        global_step=global_step
                    )
                    loss_monitor = 0

            # decay learning rate
            schdlr.step()

            # save model
            checkpoint = {
                'net_state_dict': net.state_dict(),
                'op_state_dict': meta_optimizer.state_dict(),
                'lr_schdlr_state_dict': schdlr.state_dict()
            }
            checkpoint_filename = 'Epoch_{0:d}.pt'.format(epoch_id + 1)
            torch.save(checkpoint, os.path.join(logdir, checkpoint_filename))
            checkpoint = 0
            print('SAVING parameters into {0:s}\n'.format(checkpoint_filename))

    except KeyboardInterrupt:
        pass
    else:
        pass
    finally:
        print('\nClose tensorboard summary writer')
        tb_writer.close()

# --------------------------------------------------
# EVALUATION
# --------------------------------------------------
def evaluate() -> None:
    assert resume_epoch > 0

    # load model
    net, _, _ = load_model(epoch_id=resume_epoch)

    x, y_noisy, y0 = eps_generator.generate_episode(episode_name=mode_)
            
    # split into train and validation
    xt = x[:k_shot]
    yt = y_noisy[:k_shot]
    xv = np.linspace(
        start=eps_generator.input_range[0],
        stop=eps_generator.input_range[1],
        num=k_shot + v_shot
    )
    xv = xv[:, np.newaxis]

    # move data to gpu
    x_t = torch.from_numpy(xt).float().to(device)
    y_t = torch.from_numpy(yt).float().to(device)
    x_v = torch.from_numpy(xv).float().to(device)

    # adapt on the support data
    fnet = adapt_to_episode(x=x_t, y=y_t, net=net)

    # evaluate on the query data
    y_logits = fnet.forward(x_v)

    plt.figure(figsize=(4, 4))
    plt.plot(eps_generator.x0, y0, linewidth=1, linestyle='--', color='C0', label='ground-truth')
    plt.plot(xv, y_logits.detach().cpu().numpy(), linewidth=1, color='C1', label='predict')
    plt.scatter(x=xt, y=yt, c='C2', marker='^', label='data')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(fname=os.path.join(logdir, 'regression.eps'), format='eps')
    plt.show()

    return None

# --------------------------------------------------
# Auxilliary
# --------------------------------------------------
def adapt_to_episode(x: torch.Tensor, y: torch.Tensor, net: torch.nn.Module) -> higher.patch._MonkeyPatchBase:
    """Also known as inner loop

    Args:
      x, y: training data and label
      net: the base network

    Return: a MonkeyPatch module
    """
    fnet = higher.patch.monkeypatch(
        module=net,
        copy_initial_weights=False,
        track_higher_grads=train_flag
    )

    for _ in range(num_inner_updates):
        y_logits = fnet.forward(x)
        cls_loss = torch.nn.functional.mse_loss(input=y_logits, target=y)

        params = fnet.fast_params # list of parameters/tensors

        if first_order:
            all_grads = torch.autograd.grad(
                outputs=cls_loss,
                inputs=params,
                retain_graph=train_flag
            )
        else:
            all_grads = torch.autograd.grad(
                outputs=cls_loss,
                inputs=params,
                create_graph=train_flag
            )

        new_params = []
        for param, grad in zip(params, all_grads):
            new_params.append(higher.optim._add(tensor=param, a1=-inner_lr, a2=grad))

        fnet.update_params(new_params)

    return fnet

def initialize_model(meta_lr: float, decay_lr: _typing.Optional[float] = 1.) -> _typing.Tuple[torch.nn.Module, torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
    """Initialize the model, optimizer and lr_scheduler
    The example here is to load ResNet18. You can write
    your own class of model, and specify here.

    Args:
        meta_lr: learning rate for meta-parameters
        decay_lr: decay factor of learning rate

    Returns:
        net:
        meta-optimizer:
        schdlr:
    """
    # define a neural network
    net = FcNet(dim_input=1, dim_output=1, num_hidden_units=(32, 32, 32))

    # initialize
    net.apply(_weights_init)

    # move to gpu
    net.to(device)

    meta_optimizer = torch.optim.Adam(params=net.parameters(), lr=meta_lr)
    schdlr = torch.optim.lr_scheduler.ExponentialLR(optimizer=meta_optimizer, gamma=decay_lr)

    return net, meta_optimizer, schdlr

def load_model(
    epoch_id: int,
    meta_lr: _typing.Optional[float] = None,
    decay_lr: _typing.Optional[float] = None
) -> _typing.Tuple[torch.nn.Module, torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
    """Initialize or load model

    Args:
        logdir: folder storing files
        n_way: number of classes in the classification
        nc: number of input channels (1 for BnW images, 3 for color images)
        epoch_id: id of the file to load
        meta_lr:
        decay_lr:

    Returns:
        net:
        meta-optimizer:
        schdlr:
    """
    net, meta_optimizer, schdlr = initialize_model(meta_lr=1e-3, decay_lr=decay_lr)

    if epoch_id > 0:
        checkpoint_filename = 'Epoch_{0:d}.pt'.format(epoch_id)
        checkpoint_fullpath = os.path.join(logdir, checkpoint_filename)
        if torch.cuda.is_available():
            saved_checkpoint = torch.load(
                checkpoint_fullpath,
                map_location=lambda storage,
                loc: storage.cuda(gpu_id)
            )
        else:
            saved_checkpoint = torch.load(
                checkpoint_fullpath,
                map_location=lambda storage,
                loc: storage
            )

        net.load_state_dict(state_dict=saved_checkpoint['net_state_dict'])
        meta_optimizer.load_state_dict(state_dict=saved_checkpoint['op_state_dict'])

        if meta_lr is not None:
            for param_g in meta_optimizer.param_groups:
                if param_g['lr'] != meta_lr:
                    param_g['lr'] = meta_lr

        schdlr.load_state_dict(state_dict=saved_checkpoint['lr_schdlr_state_dict'])
        if decay_lr is not None:
            if decay_lr != schdlr.gamma:
                schdlr.gamma = decay_lr

    return net, meta_optimizer, schdlr

if __name__ == '__main__':
    main()
