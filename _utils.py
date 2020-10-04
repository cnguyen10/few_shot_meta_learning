import torch
import random
import os
import typing as _typing
import numpy as np

def list_dir(root: str, prefix: bool = False) -> _typing.List[str]:
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


def list_files(root: str, suffix: str, prefix: bool = False) -> _typing.List[str]:
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

def train_val_split(X: _typing.List[_typing.List[np.ndarray]], k_shot: int, shuffle: bool = True) -> _typing.Tuple[np.ndarray, _typing.List[int], np.ndarray, _typing.List[int]]:
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


def _weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv\d') != -1:
        # torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        torch.nn.init.kaiming_normal_(m.weight.data)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias.data)
    elif classname.find('BatchNorm') != -1:
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