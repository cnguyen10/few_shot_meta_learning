"""
python3 classification_meta_learning.py --datasource=omniglot-py --suffix=png --base-model=CNN --no-batchnorm --first-order --ds-folder=../datasets --min-way=5 --max-way=5 --k-shot=1 --v-shot=19 --minibatch=50 --num-epochs=1 --num-episodes-per-epoch=100000 --num-inner-updates=5 --inner-lr=0.01 --meta-lr=1e-3 --resume-epoch=0 --mode=maml --train
"""
import torch
from torch.utils.tensorboard import SummaryWriter
import higher

import numpy as np
import os
import random
import sys

import argparse
import typing as typing


from EpisodeGenerator import OmniglotLoader, ImageFolderGenerator
from CommonModels import CNN, ResNet18
from _utils import train_val_split, get_episodes, euclidean_distance, get_cls_prototypes, IdentityNet, NormalVariationalNet, kl_divergence_gaussians

# --------------------------------------------------
# SETUP INPUT PARSER
# --------------------------------------------------
parser = argparse.ArgumentParser(description='Setup variables')

parser.add_argument('--datasource', type=str, default='omniglot-py', help='Dataset: omniglot, ImageNet')
parser.add_argument('--suffix', type=str, default='png', help='Suffix of images, png for omniglot-py, jpg for ImageNet')
parser.add_argument('--load-images', type=bool, default=True, help='Load images on RAM (True) or on the fly (False)')

parser.add_argument('--mode', type=str, default='maml', help='Few-shot learning methods, including: maml, abml, vampire, protonet, bmaml')

parser.add_argument('--first-order', dest='first_order', action='store_true')
parser.add_argument('--no-first-order', dest='first_order', action='store_false')
parser.set_defaults(first_order=True)

parser.add_argument('--base-model', type=str, default='CNN', help='The base model used, including CNN and ResNet18 defined in CommonModels')

# Including learnable BatchNorm in the model or not learnable BN
parser.add_argument('--batchnorm', dest='batchnorm', action='store_true')
parser.add_argument('--no-batchnorm', dest='batchnorm', action='store_false')
parser.set_defaults(batchnorm=False)

parser.add_argument('--max-way', type=int, default=5, help='Maximum number of classes within an episode')
parser.add_argument('--min-way', type=int, default=5, help='Maximum number of classes within an episode')

parser.add_argument('--num-inner-updates', type=int, default=5, help='The number of gradient updates for episode adaptation')
parser.add_argument('--inner-lr', type=float, default=1e-2, help='Learning rate of episode adaptation step')

parser.add_argument('--ds-folder', type=str, default='./datasets', help='Folder to store model and logs')
parser.add_argument('--logdir', type=str, default='/media/n10/Data/', help='Folder to store model and logs')

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

parser.add_argument('--num-models', type=int, default=1, help='Number of base network sampled from the hyper-net')

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
# Setup parameters
# --------------------------------------------------
config = {}
for key in args.__dict__:
    config[key] = args.__dict__[key]

logdir = os.path.join(config['logdir'], config['mode'], config['base_model'], config['datasource'])
if not os.path.exists(path=logdir):
    from pathlib import Path
    Path(logdir).mkdir(parents=True, exist_ok=True)

kl_weight = 0.0001
minibatch_print = np.lcm(config['minibatch'], 100)

# --------------------------------------------------
# Data loader
# --------------------------------------------------
if config['datasource'] in ['omniglot-py']:
    EpisodeGeneratorClass = OmniglotLoader
    image_size = (1, 28, 28)
elif config['datasource'] in ['miniImageNet']:
    EpisodeGeneratorClass = ImageFolderGenerator
    image_size = (3, 84, 84)
elif config['datasource'] == 'miniImageNet_64':
    EpisodeGeneratorClass = ImageFolderGenerator
    image_size = (3, 64, 64)
else:
    raise ValueError('Unknown dataset')

eps_generator = EpisodeGeneratorClass(
    root=os.path.join(config['ds_folder'], config['datasource']),
    train_subset=config['train_flag'],
    suffix=config['suffix'],
    min_num_cls=config['min_way'],
    max_num_cls=config['max_way'],
    k_shot=config['k_shot'] + config['v_shot'],
    expand_dim=False,
    load_images=config['load_images']
)

# --------------------------------------------------
# Model-related
# --------------------------------------------------
def initialize_model(hyper_net_cls, meta_lr: float, decay_lr: float = 1.) -> typing.Tuple[torch.nn.Module, torch.nn.Module, torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
    """Initialize the model, optimizer and lr_scheduler
    The example here is to load ResNet18. You can write
    your own class of model, and specify here.

    Args:
        hyper_net_cls: a handler to refer to a hyper-net class
        meta_lr: learning rate for meta-parameters
        decay_lr: decay factor of learning rate

    Returns:
        net:
        meta-optimizer:
        schdlr:
    """
    if config['base_model'] in ['CNN']:
        base_net = CNN(
            dim_output=config['min_way'] if config['min_way'] == config['max_way'] else None,
            image_size=image_size,
            bn_affine=config['batchnorm']
        )
    elif config['base_model'] in ['ResNet18']:
        base_net = ResNet18(
            input_channel=image_size[0],
            dim_output=config['min_way'] if config['min_way'] == config['max_way'] else None,
            bn_affine=config['batchnorm']
        )
    else:
        raise NotImplementedError

    hyper_net = hyper_net_cls(base_net=base_net)

    # move to gpu
    base_net.to(device)
    hyper_net.to(device)

    meta_opt = torch.optim.Adam(params=hyper_net.parameters(), lr=meta_lr)
    schdlr = torch.optim.lr_scheduler.ExponentialLR(optimizer=meta_opt, gamma=decay_lr)

    return hyper_net, base_net, meta_opt, schdlr

def load_model(
    hyper_net_cls,
    epoch_id: int,
    meta_lr: typing.Optional[float] = None,
    decay_lr: typing.Optional[float] = None
) -> typing.Tuple[torch.nn.Module, torch.nn.Module, torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
    """Initialize or load model

    Args:
        hyper_net_cls: a handler to refer to a hyper-net class
        epoch_id: id of the file to load
        meta_lr:
        decay_lr:

    Returns:
        hyper_net
        base_net:
        meta-optimizer:
        schdlr:
    """
    hyper_net, base_net, meta_opt, schdlr = initialize_model(hyper_net_cls=hyper_net_cls, meta_lr=1e-3, decay_lr=decay_lr)

    if epoch_id > 0:
        checkpoint_filename = 'Epoch_{0:d}.pt'.format(epoch_id)
        checkpoint_fullpath = os.path.join(logdir, checkpoint_filename)
        if torch.cuda.is_available():
            saved_checkpoint = torch.load(
                f=checkpoint_fullpath,
                map_location=lambda storage,
                loc: storage.cuda(gpu_id)
            )
        else:
            saved_checkpoint = torch.load(
                f=checkpoint_fullpath,
                map_location=lambda storage,
                loc: storage
            )

        hyper_net.load_state_dict(state_dict=saved_checkpoint['hyper_net_state_dict'])
        meta_opt.load_state_dict(state_dict=saved_checkpoint['op_state_dict'])

        if meta_lr is not None:
            for param_g in meta_opt.param_groups:
                if param_g['lr'] != meta_lr:
                    param_g['lr'] = meta_lr

        schdlr.load_state_dict(state_dict=saved_checkpoint['lr_schdlr_state_dict'])
        if decay_lr is not None:
            if decay_lr != schdlr.gamma:
                schdlr.gamma = decay_lr

    return hyper_net, base_net, meta_opt, schdlr

# --------------------------------------------------
# Base procedures for training and testing
# --------------------------------------------------
def train(
    hyper_net_cls,
    get_f_base_net_fn: typing.Callable,
    adapt_to_episode: typing.Callable,
    loss_on_query_fn: typing.Callable
) -> None:
    """Base method used for training

    Args:

    """
    # initialize/load model
    hyper_net, base_net, meta_opt, schdlr = load_model(
        hyper_net_cls=hyper_net_cls,
        epoch_id=config['resume_epoch'],
        meta_lr=config['meta_lr'],
        decay_lr=config['decay_lr']
    )

    # zero grad
    meta_opt.zero_grad()

    # get list of episode names, each episode name consists of classes
    episodes = get_episodes(episode_file_path=config['episode_file'])

    # initialize a tensorboard summary writer for logging
    tb_writer = SummaryWriter(
        log_dir=logdir,
        purge_step=config['resume_epoch'] * config['num_episodes_per_epoch'] // minibatch_print if config['resume_epoch'] > 0 else None
    )

    try:
        for epoch_id in range(config['resume_epoch'], config['resume_epoch'] + config['num_epochs'], 1):
            episode_count = 0
            loss_monitor = 0
            # kl_div_monitor = 0
            
            while (episode_count < config['num_episodes_per_epoch']):
                # get episode from the given csv file, or just return None
                episode_name = random.sample(population=episodes, k=1)[0]

                X = eps_generator.generate_episode(episode_name=episode_name)
                
                # split into train and validation
                xt, yt, xv, yv = train_val_split(X=X, k_shot=config['k_shot'], shuffle=True)

                # move data to gpu
                x_t = torch.from_numpy(xt).float().to(device)
                y_t = torch.tensor(yt, dtype=torch.long, device=device)
                x_v = torch.from_numpy(xv).float().to(device)
                y_v = torch.tensor(yv, dtype=torch.long, device=device)

                # -------------------------
                # functional base network
                # -------------------------
                f_base_net = get_f_base_net_fn(base_net=base_net)

                # -------------------------
                # adapt on the support data
                # -------------------------
                f_hyper_net = adapt_to_episode(x=x_t, y=y_t, hyper_net=hyper_net, f_base_net=f_base_net)

                # -------------------------
                # loss on query data
                # -------------------------
                loss_meta = loss_on_query_fn(x=x_v, y=y_v, f_hyper_net=f_hyper_net, f_base_net=f_base_net, hyper_net=hyper_net)

                if torch.isnan(loss_meta):
                    raise ValueError('Validation loss is NaN.')

                loss_meta = loss_meta / config['minibatch']
                loss_meta.backward()

                # monitoring validation loss
                loss_monitor += loss_meta.item()
                # kl_div_monitor += kl_loss.item()

                episode_count += 1
                # update the meta-model
                if (episode_count % config['minibatch'] == 0):
                    # torch.nn.utils.clip_grad_norm_(parameters=hyper_net.parameters(), max_norm=10)
                    meta_opt.step()
                    meta_opt.zero_grad()

                # monitor losses
                if (episode_count % minibatch_print == 0):
                    loss_monitor /= minibatch_print
                    # kl_div_monitor /= minibatch_print

                    # print('{}, {}'.format(loss_monitor, kl_div_monitor))
                    # print(loss_monitor)

                    global_step = (epoch_id * config['num_episodes_per_epoch'] + episode_count) // minibatch_print
                    tb_writer.add_scalar(
                        tag='Loss',
                        scalar_value=loss_monitor,
                        global_step=global_step
                    )

                    loss_monitor = 0
                    # kl_div_monitor = 0

            # decay learning rate
            schdlr.step()

            # save model
            checkpoint = {
                'hyper_net_state_dict': hyper_net.state_dict(),
                'op_state_dict': meta_opt.state_dict(),
                'lr_schdlr_state_dict': schdlr.state_dict()
            }
            checkpoint_filename = 'Epoch_{0:d}.pt'.format(epoch_id + 1)
            torch.save(checkpoint, os.path.join(logdir, checkpoint_filename))
            checkpoint = 0
            print('SAVING parameters into {0:s}\n'.format(checkpoint_filename))
    finally:
        print('\nClose tensorboard summary writer')
        tb_writer.close()

    return None

def evaluate(
    hyper_net_cls,
    get_f_base_net_fn: typing.Callable,
    adapt_to_episode: typing.Callable,
    get_accuracy_fn: typing.Callable
) -> None:
    """Evaluation
    """
    acc = []

    # initialize/load model
    hyper_net, base_net, _, _ = load_model(
        hyper_net_cls=hyper_net_cls,
        epoch_id=config['resume_epoch'],
        meta_lr=config['meta_lr'],
        decay_lr=config['decay_lr']
    )

    hyper_net.eval()
    base_net.eval()

    # get list of episode names, each episode name consists of classes
    episodes = get_episodes(episode_file_path=config['episode_file'], num_episodes=config['num_episodes'])

    try:
        acc_file = open(file=os.path.join(logdir, 'accuracy.txt'), mode='w')
        for i, episode_name in enumerate(episodes):
            X = eps_generator.generate_episode(episode_name=episode_name)

            # split into train and validation
            xt, yt, xv, yv = train_val_split(X=X, k_shot=config['k_shot'], shuffle=True)

            # move data to gpu
            x_t = torch.from_numpy(xt).float().to(device)
            y_t = torch.tensor(yt, dtype=torch.long, device=device)
            x_v = torch.from_numpy(xv).float().to(device)
            y_v = torch.tensor(yv, dtype=torch.long, device=device)

            # -------------------------
            # functional base network
            # -------------------------
            f_base_net = get_f_base_net_fn(base_net=base_net)

            # -------------------------
            # adapt on the support data
            # -------------------------
            f_hyper_net = adapt_to_episode(x=x_t, y=y_t, hyper_net=hyper_net, f_base_net=f_base_net)

            # -------------------------
            # accuracy
            # -------------------------
            acc_temp = get_accuracy_fn(x=x_v, y=y_v, f_hyper_net=f_hyper_net, f_base_net=f_base_net)

            acc.append(acc_temp)
            acc_file.write('{}\n'.format(acc_temp))

            sys.stdout.write('\033[F')
            print(i)
    finally:
        acc_file.close()
    
    acc_mean = np.mean(acc)
    acc_std = np.std(acc)
    print('Accuracy = {} +/- {}'.format(acc_mean, 1.96 * acc_std / np.sqrt(len(episodes))))
    
    return None

# --------------------------------------------------
# Task adaptation - MAML-based
# --------------------------------------------------
def get_f_base_net_fn_maml(base_net: torch.nn.Module) -> higher.patch._MonkeyPatchBase:
    """Convert from conventional net to 'functional' net
    """
    f_base_net = higher.patch.make_functional(base_net)
    # higher.patch.buffer_sync(net, fnet)
    f_base_net.track_higher_grads = False
    f_base_net._fast_params = [[]]

    return f_base_net

def adapt_to_episode_innerloop(
    x: torch.Tensor,
    y: torch.Tensor,
    hyper_net: torch.nn.Module,
    f_base_net: higher.patch._MonkeyPatchBase,
    kl_div_fn: typing.Callable
) -> higher.patch._MonkeyPatchBase:
    """Also known as inner loop

    Args:
      x, y: training data and label
      hyper_net: the model of interest
      f_base_net: the functional of the base model
      kl_divergence_loss: function calculating
        the KL divergence between variational posterior and prior
    
    Return: a MonkeyPatch module
    """
    f_hyper_net = higher.patch.monkeypatch(
        module=hyper_net,
        copy_initial_weights=False,
        track_higher_grads=config['train_flag']
    )

    hyper_net_params = [p for p in hyper_net.parameters()]

    for _ in range(config['num_inner_updates']):
        grad_accum = [0] * len(hyper_net_params) # accumulate gradients of Monte Carlo sampling

        q_params = f_hyper_net.fast_params # parameters of the hypernet

        # KL divergence
        kl_div = kl_div_fn(p=hyper_net_params, q=q_params)

        for _ in range(config['num_models']):
            base_net_params = f_hyper_net.forward()
            y_logits = f_base_net.forward(x, params=base_net_params)
            cls_loss = torch.nn.functional.cross_entropy(input=y_logits, target=y)

            loss = cls_loss + kl_div * kl_weight

            if config['first_order']:
                all_grads = torch.autograd.grad(
                    outputs=loss,
                    inputs=q_params,
                    retain_graph=config['train_flag']
                )
            else:
                all_grads = torch.autograd.grad(
                    outputs=loss,
                    inputs=q_params,
                    create_graph=config['train_flag']
                )

            for i in range(len(all_grads)):
                grad_accum[i] = grad_accum[i] + all_grads[i] / config['num_models']

        new_q_params = []
        for param, grad in zip(q_params, grad_accum):
            new_q_params.append(higher.optim._add(tensor=param, a1=-config['inner_lr'], a2=grad))

        f_hyper_net.update_params(new_q_params)

    return f_hyper_net

# --------------------------------------------------
# Task adaptation - Prototype-base
# --------------------------------------------------
def get_f_base_net_fn_protonet(base_net: torch.nn.Module) -> torch.nn.Module:
    return base_net

def adapt_to_episode_prototype(
    x: torch.Tensor,
    y: torch.Tensor,
    hyper_net: torch.nn.Module,
    f_base_net: torch.nn.Module
) -> torch.Tensor:
    """
    """
    z = f_base_net.forward(x)
    z_prototypes = get_cls_prototypes(x=z, y=y)

    return z_prototypes

# --------------------------------------------------
# PROTONET-related
# --------------------------------------------------
def evaluate_fn_protonet(
    x: torch.Tensor,
    f_hyper_net: torch.Tensor,
    f_base_net: torch.nn.Module
) -> torch.Tensor:
    """A base function to evaluate on the validation subset

    Args:
        x: data in tensor format
        f_hyper_net: the prototypes
        f_base_net: the base net

    Return: logits of the prediction
    """
    z = f_base_net.forward(x)
    distance_matrix = euclidean_distance(matrixN=z, matrixM=f_hyper_net)

    logits = - distance_matrix
    
    return logits

def loss_on_query_fn_protonet(
    x: torch.Tensor,
    y: torch.Tensor,
    f_hyper_net: torch.Tensor,
    f_base_net: torch.nn.Module,
    **kwargs
) -> torch.Tensor:
    """
    Args:
        x, y: data in tensor format
        f_hyper_net: the prototypes
        f_base_net: the base net

    Return: the loss of the given data
    """
    logits = evaluate_fn_protonet(x=x, f_hyper_net=f_hyper_net, f_base_net=f_base_net)
    cls_loss = torch.nn.functional.cross_entropy(input=logits, target=y)

    return cls_loss

def get_accuracy_fn_protonet(
    x: torch.Tensor,
    y: torch.Tensor,
    f_hyper_net: torch.Tensor,
    f_base_net: torch.nn.Module
) -> float:
    """Evaluate on the validation subset

    Args:
        x, y: data in tensor
        f_hyper_net: the prototypes
        f_base_net: the base net

    Return: accuracy of the given data
    """
    logits = evaluate_fn_protonet(x=x, f_hyper_net=f_hyper_net, f_base_net=f_base_net)
    accuracy = (logits.argmax(dim=1) == y).float().mean().item()

    return accuracy
# --------------------------------------------------
# MAML-related
# --------------------------------------------------
def evaluate_fn_maml(
    x: torch.Tensor,
    f_hyper_net: higher.patch._MonkeyPatchBase,
    f_base_net: higher.patch._MonkeyPatchBase
) -> torch.Tensor:
    """A base function to evaluate on the validation subset

    Args:
        x: the data in the validation subset
        f_hyper_net: the adapted meta-parameter
        f_base_net: the functional form of the base net

    Return: the logits of the prediction
    """
    base_net_params = f_hyper_net.forward()
    logits = f_base_net.forward(x, params=base_net_params)

    return logits

def adapt_to_episode_innerloop_maml(
    x: torch.Tensor,
    y: torch.Tensor,
    hyper_net: torch.nn.Module,
    f_base_net: higher.patch._MonkeyPatchBase
) -> higher.patch._MonkeyPatchBase:
    """"""
    return adapt_to_episode_innerloop(
        x=x,
        y=y,
        hyper_net=hyper_net,
        f_base_net=f_base_net,
        kl_div_fn=lambda p, q: 0
    )

def loss_on_query_fn_maml(
    x: torch.Tensor,
    y: torch.Tensor,
    f_hyper_net: higher.patch._MonkeyPatchBase,
    f_base_net: higher.patch._MonkeyPatchBase,
    **kwargs
) -> torch.Tensor:
    """Evaluate and then calculate the loss on the given data

    Args:
        x, y: given data
        f_hyper_net: adapted meta-parameters
        f_base_net: functional form of the base net

    Return: loss on the given data
    """
    logits = evaluate_fn_maml(x=x, f_hyper_net=f_hyper_net, f_base_net=f_base_net)
    loss_meta = torch.nn.functional.cross_entropy(input=logits, target=y)

    return loss_meta

def get_accuracy_fn_maml(
    x: torch.Tensor,
    y: torch.Tensor,
    f_hyper_net: higher.patch._MonkeyPatchBase,
    f_base_net: higher.patch._MonkeyPatchBase
) -> float:
    """Calculate accuracy on the given data

    Args:
        x, y: given data
        f_hyper_net: adapted meta-parameters
        f_base_net: functional form of the base net

    Return: accuracy
    """
    logits = evaluate_fn_maml(x=x, f_hyper_net=f_hyper_net, f_base_net=f_base_net)
    accuracy = (logits.argmax(dim=1) == y).float().mean().item()

    return accuracy

# --------------------------------------------------
# VAMPIRE - related
# --------------------------------------------------
def evaluate_fn_vampire(
    x: torch.Tensor,
    f_hyper_net: higher.patch._MonkeyPatchBase,
    f_base_net: higher.patch._MonkeyPatchBase
) -> typing.List[torch.Tensor]:
    """
    """
    logits = [None] * config['num_models']
    for model_id in range(config['num_models']):
        base_net_params = f_hyper_net.forward()
        logits_temp = f_base_net.forward(x, params=base_net_params)

        logits[model_id] = logits_temp
    
    return logits

def adapt_to_episode_innerloop_vampire(
    x: torch.Tensor,
    y: torch.Tensor,
    hyper_net: torch.nn.Module,
    f_base_net: higher.patch._MonkeyPatchBase
) -> higher.patch._MonkeyPatchBase:
    """"""
    return adapt_to_episode_innerloop(
        x=x,
        y=y,
        hyper_net=hyper_net,
        f_base_net=f_base_net,
        kl_div_fn=kl_divergence_gaussians
    )

def loss_on_query_fn_vampire(
    x: torch.Tensor,
    y: torch.Tensor,
    f_hyper_net: higher.patch._MonkeyPatchBase,
    f_base_net: higher.patch._MonkeyPatchBase,
    **kwargs
) -> torch.Tensor:
    """"""
    loss_meta = 0
    
    logits = evaluate_fn_vampire(x=x, f_hyper_net=f_hyper_net, f_base_net=f_base_net)
    for logit in logits:
        loss_meta_temp = torch.nn.functional.cross_entropy(input=logit, target=y)
        loss_meta = loss_meta + loss_meta_temp
    
    loss_meta = loss_meta / len(logits)

    return loss_meta

def get_accuracy_fn_vampire(
    x: torch.Tensor,
    y: torch.Tensor,
    f_hyper_net: higher.patch._MonkeyPatchBase,
    f_base_net: higher.patch._MonkeyPatchBase
) -> float:
    """
    """
    logits_avg = 0
    
    logits = evaluate_fn_vampire(x=x, f_hyper_net=f_hyper_net, f_base_net=f_base_net)
    for logit in logits:
        logits_sm = torch.nn.functional.softmax(input=logit, dim=1)
        logits_avg = logits_avg + logits_sm
    
    logits_avg = logits_avg / len(logits)

    accuracy = (logits_avg.argmax(dim=1) == y).float().mean().item()

    return accuracy

# --------------------------------------------------
# ABML - related
# --------------------------------------------------
def loss_on_query_fn_abml(
    x: torch.Tensor,
    y: torch.Tensor,
    f_hyper_net: higher.patch._MonkeyPatchBase,
    f_base_net: higher.patch._MonkeyPatchBase,
    **kwargs
) -> torch.Tensor:
    """"""
    loss_cls = loss_on_query_fn_vampire(x=x, y=y, f_hyper_net=f_hyper_net, f_base_net=f_base_net)

    hyper_net_params = [p for p in kwargs['hyper_net'].parameters()]
    q_params = f_hyper_net.fast_params
    kl_loss = kl_divergence_gaussians(p=q_params, q=hyper_net_params)

    loss_meta = loss_cls + kl_loss * kl_weight

    # add loss due to prior here

    return loss_meta

# --------------------------------------------------
# MAIN
# --------------------------------------------------
if __name__ == "__main__":
    # verify if setting is correct
    if config['mode'] in ['maml', 'vampire', 'abml']:
        if config['min_way'] != config['max_way']:
            raise ValueError("The selected algorithm is expected a fixed number of classes. Please specify --max-way=n --min-way=n.")

    meta_algorithm = {
        'maml': {
            'hyper_net_cls': IdentityNet,
            'get_f_base_net_fn': get_f_base_net_fn_maml,
            'adapt_to_episode': adapt_to_episode_innerloop_maml,
            'loss_on_query_fn': loss_on_query_fn_maml,
            'get_accuracy_fn': get_accuracy_fn_maml
        },
        'vampire': {
            'hyper_net_cls': NormalVariationalNet,
            'get_f_base_net_fn': get_f_base_net_fn_maml,
            'adapt_to_episode': adapt_to_episode_innerloop_vampire,
            'loss_on_query_fn': loss_on_query_fn_vampire,
            'get_accuracy_fn': get_accuracy_fn_vampire
        },
        'abml': {
            'hyper_net_cls': NormalVariationalNet,
            'get_f_base_net_fn': get_f_base_net_fn_maml,
            'adapt_to_episode': adapt_to_episode_innerloop_vampire,
            'loss_on_query_fn': loss_on_query_fn_abml
        },
        'protonet': {
            'hyper_net_cls': lambda base_net: base_net,
            'get_f_base_net_fn': get_f_base_net_fn_protonet,
            'adapt_to_episode': adapt_to_episode_prototype,
            'loss_on_query_fn': loss_on_query_fn_protonet,
            'get_accuracy_fn': get_accuracy_fn_protonet
        }
    }
    if config['train_flag']:
        train(
            hyper_net_cls=meta_algorithm[config['mode']]['hyper_net_cls'],
            get_f_base_net_fn=meta_algorithm[config['mode']]['get_f_base_net_fn'],
            adapt_to_episode=meta_algorithm[config['mode']]['adapt_to_episode'],
            loss_on_query_fn=meta_algorithm[config['mode']]['loss_on_query_fn']
        )
    else:
        evaluate(
            hyper_net_cls=meta_algorithm[config['mode']]['hyper_net_cls'],
            get_f_base_net_fn=meta_algorithm[config['mode']]['get_f_base_net_fn'],
            adapt_to_episode=meta_algorithm[config['mode']]['adapt_to_episode'],
            get_accuracy_fn=meta_algorithm[config['mode']]['get_accuracy_fn']
        )
