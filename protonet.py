"""
python3 protonet2.py --datasource=omniglot-py --ds-folder=/home/n10/Dropbox/ML/datasets --logdir=/media/n10/Data/ --k-shot=1 --v-shot=15 --meta-lr=1e-3 --minibatch=25 --decay-lr=0.9 --num-epochs=1 --resume-epoch=0
"""

import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
import higher

import numpy as np
import os
import random
import sys
import itertools

import csv

import argparse
import typing as _typing

from CommonModels import ConvNet, ResNet18
from _utils import train_val_split, _weights_init, euclidean_distance

# --------------------------------------------------
# SETUP INPUT PARSER
# --------------------------------------------------
parser = argparse.ArgumentParser(description='Setup variables')

parser.add_argument('--datasource', type=str, default='omniglot', help='Dataset: omniglot, ImageNet')

parser.add_argument('--ds-folder', type=str, default='./datasets', help='Folder to store model and logs')
parser.add_argument('--logdir', type=str, default='/media/n10/Data/', help='Folder to store model and logs')

parser.add_argument('--n-way', type=int, default=5, help='Number of classes of an episode')

parser.add_argument('--meta-lr', type=float, default=1e-3, help='Learning rate for meta-update')
parser.add_argument('--minibatch', type=int, default=5, help='Minibatch of episodes to update meta-parameters')
parser.add_argument('--decay-lr', type=float, default=1., help='Factor to decay meta learning rate')

parser.add_argument('--k-shot', type=int, default=1, help='Number of training examples per class')
parser.add_argument('--v-shot', type=int, default=15, help='Number of validation examples per class')

parser.add_argument('--num-episodes-per-epoch', type=int, default=10000, help='Save meta-parameters after this number of episodes')
parser.add_argument('--num-epochs', type=int, default=1, help='')
parser.add_argument('--resume-epoch', type=int, default=0, help='Resume')

parser.add_argument('--train', dest='train_flag', action='store_true')
parser.add_argument('--test', dest='train_flag', action='store_false')
parser.set_defaults(train_flag=True)

parser.add_argument('--num-episodes', type=int, default=100, help='Number of episodes used in testing')
parser.add_argument('--episode-file', type=str, default=None, help='Path to csv file: row = episode, columns = list of classes within the episode')

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
datasource = args.datasource
logdir = os.path.join(args.logdir, 'ProtoNet', datasource)
if not os.path.exists(logdir):
    os.makedirs(logdir)

k_shot = args.k_shot
v_shot = args.v_shot

train_flag = args.train_flag

resume_epoch = args.resume_epoch

n_way = args.n_way

ds_folder = os.path.join(args.ds_folder, datasource)
# --------------------------------------------------
# Data loader
# --------------------------------------------------
if datasource in ['omniglot-py']:
    from EpisodeGenerator import OmniglotLoader
    eps_generator = OmniglotLoader(
        root=ds_folder,
        images_background=train_flag,
        max_num_cls=5 if n_way is None else n_way,
        min_num_cls=20 if n_way is None else n_way,
        k_shot=k_shot + v_shot,
        expand_dim=False,
        load_images=True
    )
    nc = 1
elif datasource in ['miniImageNet']:
    from EpisodeGenerator import ImageFolderGenerator
    eps_generator = ImageFolderGenerator(
        root=ds_folder,
        train_subset=train_flag,
        suffix='.jpg',
        min_num_cls=n_way,
        max_num_cls=n_way,
        k_shot=k_shot + v_shot,
        expand_dim=False,
        load_images=True
    )
    nc = 3
else:
    raise ValueError('Unknown dataset')

# --------------------------------------------------
# MAIN
# --------------------------------------------------
def main():
    if train_flag:
        train()
    else:
        acc = evaluate()
        mean = np.mean(a=acc)
        std = np.std(a=acc)
        n = len(acc)
        print('Accuracy = {0:.4f} +/- {1:.4f}'.format(mean, 1.96 * std / np.sqrt(n)))

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

        episode_file = args.episode_file

        # initialize/load model
        net, meta_optimizer, schdlr = load_model(epoch_id=resume_epoch, meta_lr=meta_lr, decay_lr=decay_lr)
        
        # zero grad
        meta_optimizer.zero_grad()

        # get episode list if not None -> generator of episode names, each episode name consists of classes
        episodes = get_episodes(episode_file_path=episode_file, num_episodes=None)

        # initialize a tensorboard summary writer for logging
        tb_writer = SummaryWriter(
            log_dir=logdir,
            purge_step=resume_epoch * num_episodes_per_epoch // minibatch_print if resume_epoch > 0 else None
        )

        for epoch_id in range(resume_epoch, resume_epoch + num_epochs, 1):
            episode_count = 0
            loss_monitor = 0
            
            while (episode_count < num_episodes_per_epoch):
                # get episode from the given csv file, or just return None
                try:
                    episode_ = next(episodes)
                except StopIteration:
                    # if running out of episodes from the csv file, reset episode generator
                    episodes = get_episodes(episode_file_path=episode_file)
                finally:
                    episode_ = next(episodes)

                # randomly skip episode <=> shuffle episodes
                if random.random() < 0.5:
                    continue

                X = eps_generator.generate_episode(episode_name=episode_)
                
                # split into train and validation
                xt, yt, xv, yv = train_val_split(X=X, k_shot=k_shot, shuffle=True)

                # move data to gpu
                x_t = torch.from_numpy(xt).float().to(device)
                y_t = torch.tensor(yt, dtype=torch.long, device=device)
                x_v = torch.from_numpy(xv).float().to(device)
                y_v = torch.tensor(yv, dtype=torch.long, device=device)

                # adapt on the support data
                z_prototypes = adapt_to_episode(x=x_t, y=y_t, net=net)

                # evaluate on the query data
                z_v = net.forward(x_v)
                distance_matrix = euclidean_distance(matrixN=z_v, matrixM=z_prototypes)
                cls_loss = torch.nn.functional.cross_entropy(input=-distance_matrix, target=y_v)
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
def evaluate() -> _typing.List[float]:
    assert resume_epoch > 0

    episode_file = args.episode_file
    if episode_file is None:
        num_episodes = args.num_episodes
    else:
        num_episodes = None

    acc = []

    if (num_episodes is None) and (episode_file is None):
        raise ValueError('Expect exactly one of num_episodes and episode_file to be not None, receive both are None.')

    # load model
    net, _, _ = load_model(epoch_id=resume_epoch)
    episodes = get_episodes(episode_file_path=episode_file, num_episodes=num_episodes)
    for i, episode_ in enumerate(episodes):
        X = eps_generator.generate_episode(episode_name=episode_)
                
        # split into train and validation
        xt, yt, xv, yv = train_val_split(X=X, k_shot=k_shot, shuffle=True)

        # move data to gpu
        x_t = torch.from_numpy(xt).float().to(device)
        y_t = torch.tensor(yt, dtype=torch.long, device=device)
        x_v = torch.from_numpy(xv).float().to(device)
        y_v = torch.tensor(yv, dtype=torch.long, device=device)

        # adapt on the support data
        z_prototypes = adapt_to_episode(x=x_t, y=y_t, net=net)

        # evaluate on the query data
        z_v = net.forward(x_v)
        distance_matrix = euclidean_distance(matrixN=z_v, matrixM=z_prototypes)
        logits_v = -distance_matrix
        episode_acc = (logits_v.argmax(dim=1) == y_v).sum().item() / (eps_generator.k_shot * len(X))

        acc.append(episode_acc)

        sys.stdout.write('\033[F')
        print(i)
    
    return acc

    
# --------------------------------------------------
# Auxilliary
# --------------------------------------------------
def adapt_to_episode(x: torch.Tensor, y: torch.Tensor, net: torch.nn.Module) -> torch.Tensor:
    """Also known as inner loop

    Args:
      x, y: training data and label
      net: the base network
    
    Return: a MonkeyPatch module
    """
    # get embedding
    z = net.forward(x)

    z_prototypes = get_cls_prototypes(x=z, y=y)

    return z_prototypes

def get_cls_prototypes(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Calculate the prototypes/centroids

    Args:
        x: input data
        y: corresponding labels

    Returns: a tensor of prototypes with shape (C, d),
        where C is the number of classes, d is the embedding dimension
    """
    _, d = x.shape
    cls_idx = torch.unique(input=y, return_counts=False)
    C = cls_idx.shape[0]

    prototypes = torch.empty(size=(C, d), device=x.device)
    for c in range(C):
        prototypes[c, :] = torch.mean(input=x[y == cls_idx[c]], dim=0)

    return prototypes

def get_episodes(
    episode_file_path: _typing.Optional[str] = None,
    num_episodes: _typing.Optional[int] = None
) -> _typing.Generator:
    """Get episodes from a file

    Args:
        episode_file_path:
        num_episodes: dummy variable in training to create an infinite
            episode (str) generator. In testing, it defines how many
            episodes to evaluate

    Return: an episode (str) generator
    """
    # get episode list if not None
    if episode_file_path is not None:
        with open(file=episode_file_path, mode='r') as f_csv:
            csv_rd = csv.reader(f_csv, delimiter=',')
            episodes = (row for row in csv_rd) # generator from list
    elif num_episodes is not None:
        episodes = itertools.repeat(None, times=num_episodes)
    else:
        episodes = itertools.repeat(None)
    
    return episodes

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
    net = ResNet18(input_channel=nc, dim_output=n_way)
    # net = ConvNet(dim_output=None)

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

        schdlr = torch.optim.lr_scheduler.ExponentialLR(optimizer=meta_optimizer, gamma=decay_lr)
        schdlr.load_state_dict(state_dict=saved_checkpoint['lr_schdlr_state_dict'])
        if decay_lr is not None:
            if decay_lr != schdlr.gamma:
                schdlr.gamma = decay_lr

    return net, meta_optimizer, schdlr

# --------------------------------------------------
# 
# --------------------------------------------------
if __name__ == '__main__':
    main()
