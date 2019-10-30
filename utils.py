import torch
import torchvision
import numpy as np

import os
import collections
import pickle
import imageio

def load_dataset(dataset_name, subset):
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
    all_classes = collections.OrderedDict()
    all_data = collections.OrderedDict()

    # if dataset_name != 'tieredImageNet':
    #     dataset_root = '../datasets'
    # else:
    #     dataset_root = '/home/n10/Downloads/embeddings/tieredImageNet/center'
    dataset_root = '/media/n10/Data/datasets'

    if dataset_name in ['miniImageNet_embedding', 'tieredImageNet_embedding']:
        subset_path = os.path.join(dataset_root, dataset_name, subset + '.pkl')
        with open(file=subset_path, mode='rb') as pkl_file:
            raw_data = pickle.load(pkl_file, encoding='latin1')

        for i, k in enumerate(raw_data["keys"]):
            _, class_label, image_file = k.split("-")
            image_file_class_label = image_file.split("_")[0]

            assert class_label == image_file_class_label

            all_data[image_file] = torch.tensor(raw_data["embeddings"][i])
            if class_label not in all_classes:
                all_classes[class_label] = []
            all_classes[class_label].append(image_file)

        all_classes = collections.OrderedDict([
            (k, np.array(v)) for k, v in list(all_classes.items())
        ])
        return all_classes, all_data
    else:
        np2tensor = torchvision.transforms.ToTensor()

        subset_path = os.path.join(dataset_root, dataset_name, subset)

        all_class_names = [folder_name for folder_name in os.listdir(subset_path) if os.path.isdir(os.path.join(subset_path, folder_name))]
        for a_class_name in all_class_names:
            a_class_path = os.path.join(subset_path, a_class_name)
            all_classes[a_class_name] = [datapoint_name for datapoint_name in os.listdir(a_class_path)]
        
            for a_datapoint in all_classes[a_class_name]:
                datapoint_embedding = imageio.imread(os.path.join(a_class_path, a_datapoint))
                all_data[a_datapoint] = np2tensor(datapoint_embedding)
        
        return all_classes, all_data

def get_task_image_data(
    all_classes,
    all_data,
    class_labels,
    num_total_samples_per_class,
    num_training_samples_per_class,
    device,
    rnd_seed=None
):
    if rnd_seed is not None:
        np.random.seed(rnd_seed)

    x_t = []
    y_t = []
    x_v = []
    y_v = []

    for i, class_label in enumerate(class_labels):
        x0 = []
        imgs_from_class = np.random.choice(a=all_classes[class_label], size=num_total_samples_per_class, replace=False)
        for img_from_class in imgs_from_class:
            x0.append(all_data[img_from_class])
        
        x_t.extend(x0[:num_training_samples_per_class])
        x_v.extend(x0[num_training_samples_per_class:])

        y_t.extend([i]*num_training_samples_per_class)
        y_v.extend([i]*(num_total_samples_per_class - num_training_samples_per_class))
    x_t = torch.stack(x_t).to(device)
    x_v = torch.stack(x_v).to(device)
    y_t = torch.tensor(y_t).to(device)
    y_v = torch.tensor(y_v).to(device)
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

def get_weights_target_net(w_generated, row_id, w_target_shape):
    w = {}
    temp = 0
    for key in w_target_shape.keys():
        w_temp = w_generated[row_id, temp:(temp + np.prod(w_target_shape[key]))]
        if 'b' in key:
            w[key] = w_temp
        else:
            w[key] = w_temp.view(w_target_shape[key])
        temp += np.prod(w_target_shape[key])
    return w