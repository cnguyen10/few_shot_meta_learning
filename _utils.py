import torch
import random
import os
import csv
import itertools
import typing
import numpy as np

def list_dir(root: str, prefix: bool = False) -> typing.List[str]:
    """List all directories at a given root
    Args:
        root (str): Path to directory whose folders need to be listed
        prefix (bool, optional): If true, prepends the path to each result, otherwise
            only returns the name of the directories found
    """
    root = os.path.expanduser(root)
    directories = [p for p in os.listdir(root) if os.path.isdir(os.path.join(root, p))]
    if prefix is True:
        directories = [os.path.join(root, d) for d in directories]
    return directories


def list_files(root: str, suffix: str, prefix: bool = False) -> typing.List[str]:
    """List all files ending with a suffix at a given root
    Args:
        root (str): Path to directory whose folders need to be listed
        suffix (str or tuple): Suffix of the files to match, e.g. '.png' or ('.jpg', '.png').
            It uses the Python "str.endswith" method and is passed directly
        prefix (bool, optional): If true, prepends the path to each result, otherwise
            only returns the name of the files found
    """
    root = os.path.expanduser(root)
    files = [p for p in os.listdir(root) if os.path.isfile(os.path.join(root, p)) and p.endswith(suffix)]
    if prefix is True:
        files = [os.path.join(root, d) for d in files]
    return files

def train_val_split(X: typing.List[typing.List[np.ndarray]], k_shot: int, shuffle: bool = True) -> typing.Tuple[np.ndarray, typing.List[int], np.ndarray, typing.List[int]]:
    """Split data into train and validation

    Args:
        X: a list of sub-list of numpy array. 
            Each sub-list consists of data belonging to the same class
        k_shot: number of training data per class
        shuffle: shuffle data before splitting

    Returns:
    """
    # get information of image size
    nc, iH, iW = X[0][0].shape

    v_shot = len(X[0]) - k_shot
    num_classes = len(X)

    x_t = np.empty(shape=(num_classes, k_shot, nc, iH, iW))
    x_v = np.empty(shape=(num_classes, v_shot, nc, iH, iW))
    y_t = [0] * num_classes * k_shot
    y_v = [0] * num_classes * v_shot
    for cls_id in range(num_classes):
        if shuffle:
            random.shuffle(x=X[cls_id]) # in-place shuffle data within the same class
        x_t[cls_id, :, :, :, :] = np.array(X[cls_id][:k_shot])
        x_v[cls_id, :, :, :, :] = np.array(X[cls_id][k_shot:])
        y_t[k_shot * cls_id: k_shot * (cls_id + 1)] = [cls_id] * k_shot
        y_v[v_shot * cls_id: v_shot * (cls_id + 1)] = [cls_id] * v_shot

    x_t = np.concatenate(x_t, axis=0)
    x_v = np.concatenate(x_v, axis=0)

    return x_t, y_t, x_v, y_v

def get_episodes(episode_file_path: typing.Optional[str] = None, num_episodes: int = 100) -> typing.List[str]:
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
        episodes = []
        with open(file=episode_file_path, mode='r') as f_csv:
            csv_rd = csv.reader(f_csv, delimiter=',')
            episodes = list(csv_rd)
    else:
        episodes = [None] * num_episodes
    
    return episodes


def _weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        if m.weight is not None:
            torch.nn.init.kaiming_normal_(m.weight.data)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias.data)
    elif classname.find('BatchNorm') != -1:
        if m.weight is not None:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0)

def euclidean_distance(matrixN: torch.Tensor, matrixM: torch.Tensor) -> torch.Tensor:
    """Calculate Euclidean distance from N points to M points

    Args:
        matrixN: an N x D matrix for N points
        matrixM: a M x D matrix for M points

    Returns: N x M matrix
    """
    N = matrixN.size(0)
    M = matrixM.size(0)
    D = matrixN.size(1)
    assert D == matrixM.size(1)

    matrixN = matrixN.unsqueeze(1).expand(N, M, D)
    matrixM = matrixM.unsqueeze(0).expand(N, M, D)

    return torch.norm(input=matrixN - matrixM, p='fro', dim=2)

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

def kl_divergence_gaussians(p: typing.List[torch.Tensor], q: typing.List[torch.Tensor]) -> torch.Tensor:
    """Calculate KL divergence between 2 diagonal Gaussian

    Args: each paramter is list with 1st half as mean, and the 2nd half is log_std

    Returns: KL divergence
    """
    assert len(p) == len(q)

    n = len(p) // 2

    kl_div = 0
    for i in range(n):
        p_mean = p[i]
        p_log_std = p[n + i]

        q_mean = q[i]
        q_log_std = q[n + i]

        s1_vec = torch.exp(input=2 * q_log_std)
        mahalanobis = torch.sum(input=torch.square(input=p_mean - q_mean) / s1_vec)

        tr_s1inv_s0 = torch.sum(input=torch.exp(input=2 * (p_log_std - q_log_std)))

        log_det = 2 * torch.sum(input=q_log_std - p_log_std)

        kl_div_temp = mahalanobis + tr_s1inv_s0 + log_det - torch.numel(p_mean)
        kl_div_temp = kl_div_temp / 2

        kl_div = kl_div + kl_div_temp

    return kl_div

def intialize_parameters(state_dict: dict) -> typing.List[torch.Tensor]:
    """"""
    p = list(state_dict.values())
    for m in p:
        if m.ndim > 1:
            torch.nn.init.kaiming_normal_(tensor=m, nonlinearity='relu')
        else:
            torch.nn.init.zeros_(tensor=m)
    
    return p