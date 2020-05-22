import torch
import numpy as np

import os
import sys

import pickle
from PIL import Image

def load_dataset(dataset_name, subset, dataset_root='../datasets/'):
    '''
    Inputs:
        - dataset_path: path to the folder of the datasets, eg. ../datasets/miniImageNet
        - subset: train/val/test
    Outputs:
        - all_classes: a dictionary with
            + key: name of a class
            + values of a key: names of datapoints within that class
        - all_data: is also a dictionary with
            + key: name of a datapoint
            + value: embedding data of that datapoint
    '''
    all_classes = {}
    all_data = {}

    if dataset_name in ['omniglot', 'miniImageNet']:
        NORM_FACTOR = {'omniglot': 1, 'miniImageNet': 255}

        path2subset = os.path.join(dataset_root, dataset_name, subset)
        all_class_names = [folder_name for folder_name in os.listdir(path2subset) \
            if os.path.isdir(os.path.join(path2subset, folder_name))]
        
        for a_class in all_class_names:
            path_temp = os.path.join(path2subset, a_class)
            
            all_classes[a_class] = [filename for filename in os.listdir(path_temp)]
            
            for filename in all_classes[a_class]:
                img = Image.open(fp=os.path.join(path_temp, filename), mode='r')
                img_arr = np.asarray(a=img, dtype=np.float32) / NORM_FACTOR[dataset_name]
                
                if img_arr.ndim == 2:
                    img_arr = np.expand_dims(img_arr, axis=-1)
                assert img_arr.ndim == 3

                all_data[filename] = np.transpose(a=img_arr, axes=(2, 0, 1))
    elif dataset_name in ['miniImageNet_640', 'tiered_ImageNet_640']:
        with open(file='{0:s}/{1:s}/{2:s}.pkl'.format(dataset_root, dataset_name, subset), mode='rb') as f:
            all_classes, all_data = pickle.load(f)
    return all_classes, all_data

def initialize_dataloader(all_classes, num_classes_per_task):
    '''
    - all_classes = list of all classes
    - num_classes_per_task = number of ways to classify
    '''
    my_sampler = torch.utils.data.sampler.RandomSampler(
        data_source=all_classes,
        replacement=False
    )
    data_loader = torch.utils.data.DataLoader(
        dataset=all_classes,
        batch_size=num_classes_per_task,
        sampler=my_sampler,
        drop_last=True
    )

    return data_loader

def get_task_data(
    all_classes,
    all_data,
    class_labels,
    num_samples_per_class,
    num_training_samples_per_class=None
):
    x_t = []
    y_t = []
    x_v = []
    y_v = []

    for i, class_label in enumerate(class_labels):
        x_temp = []
        datapoint_labels = np.random.choice(
            a=all_classes[class_label],
            size=num_samples_per_class,
            replace=False
        ) # ndarray of data labels, such as: array(['n01558993_8291.JPEG', 'n01558993_14281.JPEG'], dtype='<U20')
        # datapoint_labels = all_classes[class_label][300:]
        for datapoint_label in datapoint_labels:
            x_temp.append(all_data[datapoint_label])
        x_t.append(np.array(x_temp[:num_training_samples_per_class]))
        y_t.extend([i] * num_training_samples_per_class)
        x_v.append(np.array(x_temp[num_training_samples_per_class:]))
        y_v.extend([i] * len(x_temp[num_training_samples_per_class:]))
    return np.concatenate(x_t, axis=0), y_t, np.concatenate(x_v, axis=0), y_v

def get_train_val_task_data(all_classes, all_data, class_labels, num_samples_per_class, num_training_samples_per_class, device):
    x_t, y_t, x_v, y_v = get_task_data(
        all_classes=all_classes,
        all_data=all_data,
        class_labels=class_labels,
        num_samples_per_class=num_samples_per_class,
        num_training_samples_per_class=num_training_samples_per_class
    )
    
    x_t = torch.tensor(data=x_t, dtype=torch.float, device=device)
    y_t = torch.tensor(data=y_t, device=device)
    x_v = torch.tensor(data=x_v, dtype=torch.float, device=device)
    y_v = torch.tensor(data=y_v, device=device)

    return x_t, y_t, x_v, y_v

def get_task_sine_line_data(data_generator, p_sine, num_training_samples, noise_flag=True):
    if (np.random.binomial(n=1, p=p_sine) == 0):
        # generate sinusoidal data
        x, y, _, _ = data_generator.generate_sinusoidal_data(noise_flag=noise_flag)
    else:
        # generate line data
        x, y, _, _ = data_generator.generate_line_data(noise_flag=noise_flag)
    
    x_t = x[:num_training_samples]
    y_t = y[:num_training_samples]
    
    x_v = x[num_training_samples:]
    y_v = y[num_training_samples:]

    return x_t, y_t, x_v, y_v

def get_num_weights(my_net):
    num_weights = 0
    weight_shape = my_net.get_weight_shape()
    for key in weight_shape.keys():
        num_weights += np.prod(weight_shape[key], dtype=np.int32)
    return num_weights
