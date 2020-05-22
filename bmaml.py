'''
python3 bmaml.py --datasource=sine_line --n_way=1 --k_shot=5 --inner_lr=1e-3 --meta_lr=1e-3 --minibatch=10 --num_particles=10 --num_epochs=10 --resume_epoch=0

python3 bmaml.py --datasource=sine_line --n_way=1 --k_shot=5 --inner_lr=1e-3 --meta_lr=1e-3 --minibatch=10 --num_particles=10 --num_epochs=10 --resume_epoch=20 --test --num_val_tasks=1000

python3 vampire.py --datasource=omniglot --n_way=5 --k_shot=1 --inner_lr=1e-2 --num_inner_updates=5 --meta_lr=1e-3 --lr_decay=0.95 --Lt=4 --Lv=4 --kl_reweight=0.1 --minibatch=10 --num_epochs=50

python3 vampire.py --datasource=omniglot --n_way=5 --k_shot=1 --inner_lr=1e-2 --num_inner_updates=5 --Lt=32 --Lv=32 --kl_reweight=0.1 --resume_epoch=11 --test --no_uncertainty --num_val_tasks=100000

python3 bmaml.py --datasource=miniImageNet --n_way=5 --k_shot=1 --inner_lr=1e-2 --meta_lr=1e-3 --minibatch=2 --num_particles=8 --num_epochs=20 --resume_epoch=0

python3 bmaml.py --datasource=miniImageNet --n_way=5 --k_shot=1 --inner_lr=1e-2 --num_particles=8 --resume_epoch=9 --test --uncertainty --num_val_tasks=15504

python3 bmaml.py --datasource=miniImageNet_640 --n_way=5 --k_shot=1 --inner_lr=1e-2 --meta_lr=1e-3 --lr_decay=0.97 --num_particles=32 --minibatch=20 --num_epochs=25 --resume_epoch=25
'''

import torch

import numpy as np
import random
import itertools

from utils import load_dataset, initialize_dataloader, get_train_val_task_data

import os
import sys
import csv

import argparse

# -------------------------------------------------------------------------------------------------
# Setup input parser
# -------------------------------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='Setup variables for BMAML.')
parser.add_argument('--datasource', type=str, default='miniImageNet', help='sine_line, omniglot or miniImageNet')

parser.add_argument('--n_way', type=int, default=5, help='Number of classes per task')
parser.add_argument('--k_shot', type=int, default=1, help='Number of training samples per class')
parser.add_argument('--num_val_shots', type=int, default=15, help='Number of validation samples per class')

parser.add_argument('--inner_lr', type=float, default=1e-2, help='Learning rate for task adaptation')
parser.add_argument('--num_inner_updates', type=int, default=5, help='Number of gradient updates for task adaptation')
parser.add_argument('--meta_lr', type=float, default=1e-3, help='Learning rate of meta-parameters')
parser.add_argument('--lr_decay', type=float, default=1, help='Decay factor of meta-learning rate (<=1), 1 = no decay')
parser.add_argument('--minibatch', type=int, default=25, help='Number of tasks per minibatch to update meta-parameters')

parser.add_argument('--num_epochs', type=int, default=100, help='How many 10,000 tasks are used to train?')
parser.add_argument('--resume_epoch', type=int, default=0, help='Epoch id to resume learning or perform testing')

parser.add_argument('--train', dest='train_flag', action='store_true')
parser.add_argument('--test', dest='train_flag', action='store_false')
parser.set_defaults(train_flag=True)

parser.add_argument('--num_particles', type=int, default=1, help='Number of particles used for SVGD')
 
parser.add_argument('--num_val_tasks', type=int, default=100, help='Number of validation tasks')
parser.add_argument('--uncertainty', dest='uncertainty_flag', action='store_true')
parser.add_argument('--no_uncertainty', dest='uncertainty_flag', action='store_false')
parser.set_defaults(uncertainty_flag=True)

args = parser.parse_args()

# -------------------------------------------------------------------------------------------------
# Setup CPU or GPU
# -------------------------------------------------------------------------------------------------
gpu_id = 0
device = torch.device('cuda:{0:d}'.format(gpu_id) if torch.cuda.is_available() else "cpu")

# -------------------------------------------------------------------------------------------------
# Parse dataset and related variables
# -------------------------------------------------------------------------------------------------
datasource = args.datasource
print('Dataset = {0:s}'.format(datasource))

train_flag = args.train_flag
print('Learning mode = {0}'.format(train_flag))

num_classes_per_task = args.n_way
print('Number of ways = {0:d}'.format(num_classes_per_task))

num_training_samples_per_class = args.k_shot
print('Number of shots = {0:d}'.format(num_training_samples_per_class))

num_val_samples_per_class = args.num_val_shots
print('Number of validation samples per class = {0:d}'.format(num_val_samples_per_class))

num_samples_per_class = num_training_samples_per_class + num_val_samples_per_class

# -------------------------------------------------------------------------------------------------
#   Setup based model/network
# -------------------------------------------------------------------------------------------------
if datasource == 'sine_line':
    from DataGeneratorT import DataGenerator
    from FCNet import FCNet
    from utils import get_task_sine_line_data

    # loss function as mean-squared error
    loss_fn = torch.nn.MSELoss()

    # Bernoulli probability for sine and line
    # 0.5 = uniform
    p_sine = 0.5

    noise_flag = True

    # based network
    net = FCNet(
        dim_input=1,
        dim_output=1,
        # num_hidden_units=(40, 40),
        num_hidden_units=(100, 100, 100),
        device=device
    )
else:
    train_set = 'train'
    val_set = 'val'
    test_set = 'test'

    loss_fn = torch.nn.CrossEntropyLoss()
    sm = torch.nn.Softmax(dim=-1)

    if datasource in ['omniglot', 'miniImageNet']:
        from ConvNet import ConvNet

        DIM_INPUT = {
            'omniglot': (1, 28, 28),
            'miniImageNet': (3, 84, 84)
        }

        net = ConvNet(
            dim_input=DIM_INPUT[datasource],
            dim_output=num_classes_per_task,
            num_filters=(32, 32, 32, 32),
            filter_size=(3, 3),
            device=device
        )

    elif datasource in ['miniImageNet_640', 'tieredImageNet_640']:
        import pickle
        from FC640 import FC640
        net = FC640(
            dim_output=num_classes_per_task,
            num_hidden_units=(128, 32),
            device=device
        )
    else:
        sys.exit('Unknown dataset!')

weight_shape = net.get_weight_shape()

# -------------------------------------------------------------------------------------------------
# Parse training parameters
# -------------------------------------------------------------------------------------------------
inner_lr = args.inner_lr
print('Inner learning rate = {0}'.format(inner_lr))

num_inner_updates = args.num_inner_updates
print('Number of inner updates = {0:d}'.format(num_inner_updates))

meta_lr = args.meta_lr
print('Meta learning rate = {0}'.format(meta_lr))

num_tasks_per_minibatch = args.minibatch
print('Minibatch = {0:d}'.format(num_tasks_per_minibatch))

num_meta_updates_print = int(100 / num_tasks_per_minibatch)
print('Mini batch size = {0:d}'.format(num_tasks_per_minibatch))

num_epochs_save = 1

num_epochs = args.num_epochs

expected_total_tasks_per_epoch = 500
num_tasks_per_epoch = int(expected_total_tasks_per_epoch / num_tasks_per_minibatch)*num_tasks_per_minibatch

expected_tasks_save_loss = 500
num_tasks_save_loss = int(expected_tasks_save_loss / num_tasks_per_minibatch)*num_tasks_per_minibatch

num_val_tasks = args.num_val_tasks
uncertainty_flag = args.uncertainty_flag

num_particles = args.num_particles
assert num_particles > 2
print('Number of particles = {0:d}'.format(num_particles))

# -------------------------------------------------------------------------------------------------
# Setup destination folder
# -------------------------------------------------------------------------------------------------
dst_folder_root = './BMAML'
dst_folder = '{0:s}/{1:s}_{2:d}way_{3:d}shot'.format(
    dst_folder_root,
    datasource,
    num_classes_per_task,
    num_training_samples_per_class
)
if not os.path.exists(dst_folder):
    os.makedirs(dst_folder)
    print('No folder for storage found')
    print('Make folder to store meta-parameters at')
else:
    print('Found existing folder. Meta-parameters will be stored at')
print(dst_folder)

# -------------------------------------------------------------------------------------------------
# Initialize/Load meta-parameters
# -------------------------------------------------------------------------------------------------
resume_epoch = args.resume_epoch
if resume_epoch == 0:
    # initialise meta-parameters
    theta = []
    for _ in range(num_particles):
        theta_flatten = []
        for key in weight_shape.keys():
            if isinstance(weight_shape[key], tuple):
                theta_temp = torch.empty(weight_shape[key], device=device)
                torch.nn.init.xavier_normal_(tensor=theta_temp)
            else:
                theta_temp = torch.zeros(weight_shape[key], device=device)
            theta_flatten.append(torch.flatten(theta_temp, start_dim=0, end_dim=-1))
            
        theta.append(torch.cat(theta_flatten))
    theta = torch.stack(theta)
    theta.requires_grad_()
else:
    print('Restore previous theta...')
    print('Resume epoch {0:d}'.format(resume_epoch))
    checkpoint_filename = 'Epoch_{0:d}.pt'.format(resume_epoch)
    checkpoint_file = os.path.join(dst_folder, checkpoint_filename)
    print('Start to load weights from')
    print('{0:s}'.format(checkpoint_file))
    if torch.cuda.is_available():
        saved_checkpoint = torch.load(
            checkpoint_file,
            map_location=lambda storage,
            loc: storage.cuda(gpu_id)
        )
    else:
        saved_checkpoint = torch.load(
            checkpoint_file,
            map_location=lambda storage,
            loc: storage
        )

    theta = saved_checkpoint['theta']

op_theta = torch.optim.Adam(
    params=[theta],
    lr=meta_lr
)

if resume_epoch > 0:
    op_theta.load_state_dict(saved_checkpoint['op_theta'])
    # op_theta.param_groups[0]['lr'] = meta_lr
    del saved_checkpoint

# decay the learning rate
scheduler = torch.optim.lr_scheduler.ExponentialLR(
    optimizer=op_theta,
    gamma=args.lr_decay
)

print(op_theta)
print()

# -------------------------------------------------------------------------------------------------
# MAIN program
# -------------------------------------------------------------------------------------------------
def main():
    if train_flag:
        meta_train()
    else: # validation
        assert resume_epoch > 0

        if datasource == 'sine_line':
            validate_regression(uncertainty_flag=uncertainty_flag, num_val_tasks=num_val_tasks)
        else:
            all_class_test, all_data_test = load_dataset(
                dataset_name=datasource,
                subset=test_set
            )
        
            validate_classification(
                all_classes=all_class_test,
                all_data=all_data_test,
                num_val_tasks=num_val_tasks,
                rand_flag=False,
                uncertainty=uncertainty_flag,
                csv_flag=True
            )

def meta_train():
    if datasource == 'sine_line':
        data_generator = DataGenerator(num_samples=num_samples_per_class)
        # create dummy sampler
        all_class_train = [0] * 10
    else:
        all_class_train, all_data_train = load_dataset(
            dataset_name=datasource,
            subset=train_set
        )
        all_class_val, all_data_val = load_dataset(
            dataset_name=datasource,
            subset=val_set
        )
        all_class_train.update(all_class_val)
        all_data_train.update(all_data_val)
        
    # initialize data loader
    train_loader = initialize_dataloader(
        all_classes=[class_label for class_label in all_class_train],
        num_classes_per_task=num_classes_per_task
    )
    
    for epoch in range(resume_epoch, resume_epoch + num_epochs):
        # variables used to store information of each epoch for monitoring purpose
        meta_loss_saved = [] # meta loss to save
        val_accuracies = []
        train_accuracies = []

        meta_loss = 0 # accumulate the loss of many ensambling networks to descent gradient for meta update
        num_meta_updates_count = 0

        meta_loss_avg_print = 0 # compute loss average to print

        meta_loss_avg_save = [] # meta loss to save

        task_count = 0 # a counter to decide when a minibatch of task is completed to perform meta update

        while (task_count < num_tasks_per_epoch):
            for class_labels in train_loader:
                if datasource == 'sine_line':
                    x_t, y_t, x_v, y_v = get_task_sine_line_data(
                        data_generator=data_generator,
                        p_sine=p_sine,
                        num_training_samples=num_training_samples_per_class,
                        noise_flag=True
                    )
                    x_t = torch.tensor(x_t, dtype=torch.float, device=device)
                    y_t = torch.tensor(y_t, dtype=torch.float, device=device)
                    x_v = torch.tensor(x_v, dtype=torch.float, device=device)
                    y_v = torch.tensor(y_v, dtype=torch.float, device=device)
                else:
                    x_t, y_t, x_v, y_v = get_train_val_task_data(
                        all_classes=all_class_train,
                        all_data=all_data_train,
                        class_labels=class_labels,
                        num_samples_per_class=num_samples_per_class,
                        num_training_samples_per_class=num_training_samples_per_class,
                        device=device
                    )
                
                q = adapt_to_task(x=x_t, y=y_t, theta0=theta)
                y_pred = predict(x=x_v, q=q)

                loss_NLL = 0
                for i in range(num_particles):
                    loss_NLL_temp = loss_fn(input=y_pred[i], target=y_v)
                    loss_NLL = loss_NLL + loss_NLL_temp
                loss_NLL = loss_NLL / num_particles

                if torch.isnan(loss_NLL).item():
                    sys.exit('NaN error')

                # accumulate meta loss
                meta_loss = meta_loss + loss_NLL

                task_count = task_count + 1

                if task_count % num_tasks_per_minibatch == 0:
                    meta_loss = meta_loss/num_tasks_per_minibatch

                    # accumulate into different variables for printing purpose
                    meta_loss_avg_print += meta_loss.item()

                    op_theta.zero_grad()
                    meta_loss.backward()
                    op_theta.step()

                    # Printing losses
                    num_meta_updates_count += 1
                    if (num_meta_updates_count % num_meta_updates_print == 0):
                        meta_loss_avg_save.append(meta_loss_avg_print/num_meta_updates_count)
                        print('{0:d}, {1:2.4f}'.format(
                            task_count,
                            meta_loss_avg_save[-1]
                        ))

                        num_meta_updates_count = 0
                        meta_loss_avg_print = 0
                    
                    if (task_count % num_tasks_save_loss == 0):
                        meta_loss_saved.append(np.mean(meta_loss_avg_save))

                        meta_loss_avg_save = []

                        if datasource != 'sine_line':
                            val_accs = validate_classification(
                                all_classes=all_class_val,
                                all_data=all_data_val,
                                num_val_tasks=100,
                                rand_flag=True,
                                uncertainty=False,
                                csv_flag=False
                            )
                            val_acc = np.mean(val_accs)
                            val_ci95 = 1.96*np.std(val_accs)/np.sqrt(num_val_tasks)
                            print('Validation accuracy = {0:2.4f} +/- {1:2.4f}'.format(val_acc, val_ci95))
                            val_accuracies.append(val_acc)

                            train_accs = validate_classification(
                                all_classes=all_class_train,
                                all_data=all_data_train,
                                num_val_tasks=100,
                                rand_flag=True,
                                uncertainty=False,
                                csv_flag=False
                            )
                            train_acc = np.mean(train_accs)
                            train_ci95 = 1.96*np.std(train_accs)/np.sqrt(num_val_tasks)
                            print('Train accuracy = {0:2.4f} +/- {1:2.4f}\n'.format(train_acc, train_ci95))
                            train_accuracies.append(train_acc)
                    
                    # reset meta loss
                    meta_loss = 0

                if (task_count >= num_tasks_per_epoch):
                    break
        if ((epoch + 1)% num_epochs_save == 0):
            checkpoint = {
                'theta': theta,
                'meta_loss': meta_loss_saved,
                'val_accuracy': val_accuracies,
                'train_accuracy': train_accuracies,
                'op_theta': op_theta.state_dict()
            }
            print('SAVING WEIGHTS...')
            checkpoint_filename = 'Epoch_{0:d}.pt'.format(epoch + 1)
            print(checkpoint_filename)
            torch.save(checkpoint, os.path.join(dst_folder, checkpoint_filename))
        scheduler.step()
        print()

def adapt_to_task(x, y, theta0):
    # 1st gradient update
    distance_NLL = []
    for particle_id in range(num_particles):
        w = get_weights_target_net(w_generated=theta0, row_id=particle_id, w_target_shape=weight_shape)
        y_pred = net.forward(x=x, w=w)
        loss_NLL = loss_fn(input=y_pred, target=y)
        
        loss_NLL_grads = torch.autograd.grad(
            outputs=loss_NLL,
            inputs=w.values(),
            create_graph=True
        )
        loss_NLL_gradients_dict = dict(zip(w.keys(), loss_NLL_grads))
        loss_NLL_gradients = dict2tensor(dict_obj=loss_NLL_gradients_dict)
        distance_NLL.append(loss_NLL_gradients)
    distance_NLL = torch.stack(distance_NLL)
    kernel_matrix, grad_kernel, _ = get_kernel(particle_tensor=theta0)

    q = theta0 - inner_lr * (torch.matmul(kernel_matrix, distance_NLL) - grad_kernel)

    # 2nd updates
    for _ in range(num_inner_updates - 1):
        distance_NLL = []
        for particle_id in range(num_particles):
            w = get_weights_target_net(w_generated=q, row_id=particle_id, w_target_shape=weight_shape)
            y_pred = net.forward(x=x, w=w)
            loss_NLL = loss_fn(y_pred, y)
            
            loss_NLL_grads = torch.autograd.grad(
                outputs=loss_NLL,
                inputs=w.values(),
                create_graph=True
            )
            loss_NLL_gradients_dict = dict(zip(w.keys(), loss_NLL_grads))
            loss_NLL_gradients = dict2tensor(dict_obj=loss_NLL_gradients_dict)
            distance_NLL.append(loss_NLL_gradients)
        distance_NLL = torch.stack(distance_NLL)
        kernel_matrix, grad_kernel, _ = get_kernel(particle_tensor=q)

        q = q - inner_lr * (torch.matmul(kernel_matrix, distance_NLL) - grad_kernel)
    return q

def predict(x, q):
    y_pred = torch.empty(
        size=(num_particles, num_val_samples_per_class * num_classes_per_task, num_classes_per_task),
        device=device
    )

    for i in range(num_particles):
        w = get_weights_target_net(
            w_generated=q,
            row_id=i,
            w_target_shape=weight_shape
        )
        y_pred[i, :, :] = net.forward(x=x, w=w)
    return y_pred

# -------------------------------------------------------------------------------------------------
# TEST
# -------------------------------------------------------------------------------------------------
def validate_regression(uncertainty_flag, num_val_tasks=1):
    assert datasource == 'sine_line'

    if uncertainty_flag:
        from scipy.special import erf
        quantiles = np.arange(start=0., stop=1.1, step=0.1)
        filename = 'VAMPIRE_calibration_{0:s}_{1:d}shot_{2:d}.csv'.format(
            datasource,
            num_training_samples_per_class,
            resume_epoch
        )
        outfile = open(file=os.path.join('csv', filename), mode='w')
        wr = csv.writer(outfile, quoting=csv.QUOTE_NONE)
    else: # visualization
        from matplotlib import pyplot as plt
        num_stds_plot = 2

    data_generator = DataGenerator(num_samples=num_training_samples_per_class)
    std = data_generator.noise_std

    x0 = torch.linspace(start=-5, end=5, steps=100, device=device).view(-1, 1)

    for _ in range(num_val_tasks):
        # throw a coin to see 0 - 'sine' or 1 - 'line'
        binary_flag = np.random.binomial(n=1, p=p_sine)
        if (binary_flag == 0):
            # generate sinusoidal data
            x_t, y_t, amp, phase = data_generator.generate_sinusoidal_data(noise_flag=True)
            y0 = amp * np.sin(x0 + phase)
        else:
            # generate line data
            x_t, y_t, slope, intercept = data_generator.generate_line_data(noise_flag=True)
            y0 = slope * x0 + intercept
        
        x_t = torch.tensor(x_t, dtype=torch.float, device=device)
        y_t = torch.tensor(y_t, dtype=torch.float, device=device)
        y0 = y0.numpy().reshape(shape=(1, -1))

        q = adapt_to_task(x=x_t, y=y_t, theta0=theta)
        y_pred = predict(x=x0, q=q)
        y_pred = torch.squeeze(y_pred, dim=-1).detach().cpu().numpy() # convert to numpy array Lv x len(x0)

        if uncertainty_flag:
            # each column in y_pred represents a distribution for that x0-value at that column
            # hence, we calculate the quantile along axis 0
            y_preds_quantile = np.quantile(a=y_pred, q=quantiles, axis=0, keepdims=False)

            # ground truth cdf
            cal_temp = (1 + erf((y_preds_quantile - y0)/(np.sqrt(2) * std)))/2
            cal_temp_avg = np.mean(a=cal_temp, axis=1) # average for a task
            wr.writerow(cal_temp_avg)
        else:
            y_mean = np.mean(a=y_pred, axis=0)
            y_std = np.std(a=y_pred, axis=0)
            y_top = y_mean + num_stds_plot * y_std
            y_bottom = y_mean - num_stds_plot * y_std

            plt.figure(figsize=(4, 4))

            plt.scatter(x_t.numpy(), y_t.numpy(), marker='^', label='Training data')
            plt.fill_between(
                x=torch.squeeze(x0).cpu().numpy(),
                y1=y_bottom.cpu().detach().numpy(),
                y2=y_top.cpu().detach().numpy(),
                alpha=0.25,
                zorder=0,
                label='Prediction'
            )
            plt.plot(x0, y0, linewidth=1, linestyle='--', label='Ground-truth')
            plt.xlabel('x')
            plt.ylabel('y')
            plt.legend()
            plt.tight_layout()
            plt.show()
    if uncertainty_flag:
        outfile.close()
        print('Reliability data is stored at {0:s}'.format(os.path.join('csv', filename)))

def validate_classification(
    all_classes,
    all_data,
    num_val_tasks,
    rand_flag=False,
    uncertainty=False,
    csv_flag=False
):
    if csv_flag:
        filename = 'VAMPIRE_{0:s}_{1:d}way_{2:d}shot_{3:s}_{4:d}.csv'.format(
            datasource,
            num_classes_per_task,
            num_training_samples_per_class,
            'uncertainty' if uncertainty else 'accuracy',
            resume_epoch
        )
        outfile = open(file=os.path.join('csv', filename), mode='w')
        wr = csv.writer(outfile, quoting=csv.QUOTE_NONE)
    else:
        accuracies = []
    
    total_val_samples_per_task = num_val_samples_per_class * num_classes_per_task
    all_class_names = [class_name for class_name in sorted(all_classes.keys())]
    all_task_names = itertools.combinations(all_class_names, r=num_classes_per_task)

    task_count = 0
    for class_labels in all_task_names:
        if rand_flag:
            skip_task = np.random.binomial(n=1, p=0.5) # sample from an uniform Bernoulli distribution
            if skip_task == 1:
                continue
        
        x_t, y_t, x_v, y_v = get_train_val_task_data(
            all_classes=all_classes,
            all_data=all_data,
            class_labels=class_labels,
            num_samples_per_class=num_samples_per_class,
            num_training_samples_per_class=num_training_samples_per_class,
            device=device
        )

        q = adapt_to_task(x=x_t, y=y_t, theta0=theta)
        raw_scores = predict(x=x_v, q=q)
        sm_scores = sm(input=raw_scores)
        sm_scores_avg = torch.mean(sm_scores, dim=0)
        
        prob, y_pred = torch.max(input=sm_scores_avg, dim=1)
        correct = [1 if y_pred[i] == y_v[i] else 0 for i in range(total_val_samples_per_task)]

        accuracy = np.mean(a=correct, axis=0)
        
        if csv_flag:
            if not uncertainty:
                outline = [class_label for class_label in class_labels]
                outline.append(accuracy)
                wr.writerow(outline)
            else:
                for correct_, prob_ in zip(correct, prob):
                    outline = [correct_, prob_]
                    wr.writerow(outline)
        else:
            accuracies.append(accuracy)

        task_count = task_count + 1
        if not train_flag:
            sys.stdout.write('\033[F')
            print(task_count)
        if task_count >= num_val_tasks:
            break
    if csv_flag:
        outfile.close()
        return None
    else:
        return accuracies

def get_kernel(particle_tensor):
    '''
    Compute the RBF kernel for the input particles
    Input: particles = tensor of shape (N, M)
    Output: kernel_matrix = tensor of shape (N, N)
    '''
    pairwise_d_matrix = get_pairwise_distance_matrix(particle_tensor)

    median_dist = torch.median(pairwise_d_matrix)  # tf.reduce_mean(euclidean_dists) ** 2
    h = median_dist / np.log(num_particles)

    kernel_matrix = torch.exp(-pairwise_d_matrix / h)
    kernel_sum = torch.sum(input=kernel_matrix, dim=1, keepdim=True)
    grad_kernel = -torch.matmul(kernel_matrix, particle_tensor)
    grad_kernel += particle_tensor * kernel_sum
    grad_kernel /= h
    return kernel_matrix, grad_kernel, h

def get_pairwise_distance_matrix(particle_tensor):
    '''
    Input: tensors of particles
    Output: matrix of pairwise distances
    '''
    num_particles = particle_tensor.shape[0]
    euclidean_dists = torch.nn.functional.pdist(input=particle_tensor, p=2) # shape of (N)

    # initialize matrix of pairwise distances as a N x N matrix
    pairwise_d_matrix = torch.zeros((num_particles, num_particles), device=device)

    # assign upper-triangle part
    triu_indices = torch.triu_indices(row=num_particles, col=num_particles, offset=1)
    pairwise_d_matrix[triu_indices[0], triu_indices[1]] = euclidean_dists

    # assign lower-triangle part
    pairwise_d_matrix = torch.transpose(pairwise_d_matrix, dim0=0, dim1=1)
    pairwise_d_matrix[triu_indices[0], triu_indices[1]] = euclidean_dists

    return pairwise_d_matrix

def dict2tensor(dict_obj):
    d2tensor = []
    for key in dict_obj.keys():
        tensor_temp = torch.flatten(dict_obj[key], start_dim=0, end_dim=-1)
        d2tensor.append(tensor_temp)
    d2tensor = torch.cat(d2tensor)
    return d2tensor

def get_weights_target_net(w_generated, row_id, w_target_shape):
    w = {}
    if type(w_generated) is torch.Tensor:
        temp = 0
        for key in w_target_shape.keys():
            w_temp = w_generated[row_id, temp:(temp + np.prod(w_target_shape[key]))]
            if 'b' in key:
                w[key] = w_temp
            else:
                w[key] = w_temp.view(w_target_shape[key])
            temp += np.prod(w_target_shape[key])
    elif type(w_generated) is dict:
        for key in w_generated.keys():
            w[key] = w_generated[key][row_id]

    return w

if __name__ == "__main__":
    main()
