'''
python3 bmaml.py --datasource=sine_line --n_way=1 --k_shot=5 --inner_lr=1e-3 --meta_lr=1e-3 --minibatch_size=10 --num_particles=10 --num_epochs=10 --resume_epoch=0

python3 bmaml.py --datasource=miniImageNet --n_way=5 --k_shot=1 --inner_lr=1e-2 --meta_lr=1e-3 --minibatch_size=2 --num_particles=8 --num_epochs=20 --resume_epoch=0

python3 bmaml.py --datasource=miniImageNet --n_way=5 --k_shot=1 --inner_lr=1e-2 --meta_lr=1e-3 --minibatch_size=2 --num_particles=8 --num_epochs=20 --resume_epoch=9 --test --uncertainty --num_val_tasks=15504
'''

import torch
import torchvision

import numpy as np
import random
import itertools

from utils import get_num_weights, get_weights_target_net

import os
import sys

import argparse

# region input parameters
parser = argparse.ArgumentParser(description='Setup variables for BMAML.')
parser.add_argument('--datasource', type=str, default='miniImageNet', help='sine_line, omniglot or miniImageNet')
parser.add_argument('--k_shot', type=int, default=1, help='Number of training samples per class')
parser.add_argument('--n_way', type=int, default=5, help='Number of classes per task')
parser.add_argument('--resume_epoch', type=int, default=0, help='Epoch id to resume learning or perform testing')

parser.add_argument('--train', dest='train_flag', action='store_true')
parser.add_argument('--test', dest='train_flag', action='store_false')
parser.set_defaults(train_flag=True)

parser.add_argument('--inner_lr', type=float, default=0.1, help='Learning rate for task-specific parameters - 0.1 for Omniglot, 0.01 for miniImageNet')
parser.add_argument('--num_inner_updates', type=int, default=5, help='Number of gradient updates for task-specific parameters')
parser.add_argument('--meta_lr', type=float, default=1e-3, help='Learning rate of meta-parameters')
parser.add_argument('--minibatch_size', type=int, default=10, help='Number of tasks per minibatch to update meta-parameters')
parser.add_argument('--num_epochs', type=int, default=1000, help='How many 10,000 tasks are used to train?')

parser.add_argument('--num_particles', type=int, default=1, help='Number of particles used for SVGD')

parser.add_argument('--lr_decay', type=float, default=1, help='Decay factor of meta-learning rate')
parser.add_argument('--meta_l2_regularization', type=float, default=0, help='L2 regularization coeff. of meta-parameters')
 
parser.add_argument('--num_val_tasks', type=int, default=100, help='Number of validation tasks')
parser.add_argument('--uncertainty', dest='uncertainty_flag', action='store_true')
parser.add_argument('--no_uncertainty', dest='uncertainty_flag', action='store_false')
parser.set_defaults(uncertainty_flag=True)

parser.add_argument('--p_dropout_base', type=float, default=0., help='Dropout rate for the base network')

parser.add_argument('--datasubset', type=str, default='sine', help='sine or line')

args = parser.parse_args()
# endregion

gpu_id = 0
device = torch.device('cuda:{0:d}'.format(gpu_id) if torch.cuda.is_available() else "cpu")

train_flag = args.train_flag

num_training_samples_per_class = args.k_shot # define the number of k-shot
print('{0:d}-shot'.format(num_training_samples_per_class))

# if train_flag:
num_total_samples_per_class = num_training_samples_per_class + 15 # total number of samples per class
# else:
#     num_total_samples_per_class = num_training_samples_per_class + 20

num_classes_per_task = args.n_way # n-way
print('{0:d}-way'.format(num_classes_per_task))

total_validation_samples = (num_total_samples_per_class - num_training_samples_per_class)*num_classes_per_task

datasource = args.datasource

num_tasks_per_minibatch = args.minibatch_size
num_meta_updates_print = int(1000/num_tasks_per_minibatch)
print('Dataset = {0:s}'.format(datasource))
print('Mini batch size = {0:d}'.format(num_tasks_per_minibatch))

inner_lr = args.inner_lr
print('Inner learning rate = {0}'.format(inner_lr))

expected_total_tasks_per_epoch = 10000
num_tasks_per_epoch = int(expected_total_tasks_per_epoch/num_tasks_per_minibatch)*num_tasks_per_minibatch

expected_tasks_save_loss = 2000
num_tasks_save_loss = int(expected_tasks_save_loss/num_tasks_per_minibatch)*num_tasks_per_minibatch

num_epochs_save = 1
num_epoch_decay_lr = 1

num_epochs = args.num_epochs # myriad = 10k tasks

num_inner_updates = args.num_inner_updates
print('Number of inner updates = {0:d}'.format(num_inner_updates))

# learning rate
meta_lr = args.meta_lr
print('Meta learning rate = {0}'.format(meta_lr))
 
lr_decay_factor = args.lr_decay
 
meta_l2_regularization = args.meta_l2_regularization
print('Meta L2 regularization = {0:1.3f}'.format(meta_l2_regularization))

num_particles = args.num_particles
print('Number of particles = {0:d}'.format(num_particles))

train_set = 'train'
val_set = 'val'
test_set = 'test'

# region Network architecture depends on the dataset
if datasource == 'sine_line':
    from utils import get_task_sine_line_data
    from DataGeneratorT import DataGenerator
    from FCNet import FCNet

    net = FCNet(
        dim_input=1,
        dim_output=1,
        num_hidden_units=(100, 100, 100),
        device=device
    )
    p_sine = 0.5 # bernoulli probability to pick sine or (1-p_sine) for line
    loss_fn = torch.nn.MSELoss()
else:
    from utils import load_dataset, get_task_image_data

    loss_fn = torch.nn.CrossEntropyLoss()
    sm_loss = torch.nn.Softmax(dim=2)

    # load data onto RAM
    if train_flag:
        all_class_train, embedding_train = load_dataset(datasource, train_set)
        all_class_val, embedding_val = load_dataset(datasource, val_set)
    else:
        all_class_test, embedding_test = load_dataset(datasource, test_set)

    if (datasource == 'omniglot'):
        from ConvNet import ConvNet

        net = ConvNet(
            dim_input=(1, 28, 28),
            dim_output=num_classes_per_task,
            num_filters=(64, 64, 64, 64),
            filter_size=(3, 3),
            device=device
        )
    elif (datasource == 'miniImageNet') or (datasource == 'tieredImageNet'):
        from ConvNet import ConvNet

        net = ConvNet(
            dim_input=(3, 84, 84),
            dim_output=num_classes_per_task,
            num_filters=(32, 32, 32, 32),
            filter_size=(3, 3),
            device=device
        )
    elif (datasource == 'miniImageNet_embedding') or (datasource == 'tieredImageNet_embedding'):
        from FC640 import FC640

        net = FC640(
            dim_output=num_classes_per_task,
            num_hidden_units=(256, 64),
            device=device
        )
    else:
        sys.exit('Unknown dataset')
# endregion

w_shape = net.get_weight_shape()
# print(w_shape)

num_weights = get_num_weights(net)
print('Number of parameters of base model = {0:d}'.format(num_weights))

num_val_tasks = args.num_val_tasks

p_dropout_base = args.p_dropout_base

dst_folder_root = '.'
dst_folder = '{0:s}/BMAML_few_shot/BMAML_{1:s}_{2:d}way_{3:d}shot'.format(
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

resume_epoch = args.resume_epoch
if resume_epoch == 0:
    # initialise meta-parameters
    theta = []
    for _ in range(num_particles):
        theta_flatten = []
        for key in w_shape.keys():
            if isinstance(w_shape[key], tuple):
                theta_temp = torch.empty(w_shape[key], device=device)
                torch.nn.init.xavier_normal_(tensor=theta_temp)
            else:
                theta_temp = torch.zeros(w_shape[key], device=device)
            theta_flatten.append(torch.flatten(theta_temp, start_dim=0, end_dim=-1))
            
        theta.append(torch.cat(theta_flatten))
    theta = torch.stack(theta)
    theta.requires_grad_()
else:
    print('Restore previous theta...')
    print('Resume epoch {0:d}'.format(resume_epoch))
    checkpoint_filename = ('{0:s}_{1:d}way_{2:d}shot_{3:d}.pt')\
                    .format(datasource,
                            num_classes_per_task,
                            num_training_samples_per_class,
                            resume_epoch)
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

    op_theta.param_groups[0]['lr'] = meta_lr

    del saved_checkpoint

print()

uncertainty_flag = args.uncertainty_flag

def main():
    if train_flag:
        meta_train()
    elif resume_epoch > 0:
        if datasource == 'sine_line':
            cal_data = meta_validation(datasubset=args.datasubset, num_val_tasks=num_val_tasks)

            if num_val_tasks > 0:
                cal_data = np.array(cal_data)
                np.savetxt(fname='bmaml_{0:s}_calibration.csv'.format(datasource), X=cal_data, delimiter=',')
        else:
            if not uncertainty_flag:
                accs, all_task_names = meta_validation(
                    datasubset=test_set,
                    num_val_tasks=num_val_tasks,
                    return_uncertainty=uncertainty_flag
                )
                with open(file='bmaml_{0:s}_{1:d}_{2:d}_accuracies.csv'.format(datasource, num_classes_per_task, num_training_samples_per_class), mode='w') as result_file:
                    for acc, classes_in_task in zip(accs, all_task_names):
                        row_str = ''
                        for class_in_task in classes_in_task:
                            row_str = '{0}{1},'.format(row_str, class_in_task)
                        result_file.write('{0}{1}\n'.format(row_str, acc))
            else:
                corrects, probs = meta_validation(
                    datasubset=test_set,
                    num_val_tasks=num_val_tasks,
                    return_uncertainty=uncertainty_flag
                )
                with open(file='bmaml_{0:s}_correct_prob.csv'.format(datasource), mode='w') as result_file:
                    for correct, prob in zip(corrects, probs):
                        result_file.write('{0}, {1}\n'.format(correct, prob))
                        # print(correct, prob)
    else:
        sys.exit('Unknown action')

def meta_train(train_subset=train_set):
    #region PREPARING DATALOADER
    if datasource == 'sine_line':
        data_generator = DataGenerator(
            num_samples=num_total_samples_per_class,
            device=device
        )
        # create dummy sampler
        all_class = [0]*100
        sampler = torch.utils.data.sampler.RandomSampler(data_source=all_class)
        train_loader = torch.utils.data.DataLoader(
            dataset=all_class,
            batch_size=num_classes_per_task,
            sampler=sampler,
            drop_last=True
        )
    else:
        all_class = all_class_train
        embedding = embedding_train
        sampler = torch.utils.data.sampler.RandomSampler(
            data_source=list(all_class.keys()),
            replacement=False
        )
        train_loader = torch.utils.data.DataLoader(
            dataset=list(all_class.keys()),
            batch_size=num_classes_per_task,
            sampler=sampler,
            drop_last=True
        )
    #endregion
    print('Start to train...')
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
                else:
                    x_t, y_t, x_v, y_v = get_task_image_data(
                        all_class,
                        embedding,
                        class_labels,
                        num_total_samples_per_class,
                        num_training_samples_per_class,
                        device
                    )
                
                loss_NLL = get_task_prediction(x_t, y_t, x_v, y_v)

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

                        # print('Saving loss...')
                        # val_accs, _ = meta_validation(
                        #     datasubset=val_set,
                        #     num_val_tasks=num_val_tasks,
                        #     return_uncertainty=False)
                        # val_acc = np.mean(val_accs)
                        # val_ci95 = 1.96*np.std(val_accs)/np.sqrt(num_val_tasks)
                        # print('Validation accuracy = {0:2.4f} +/- {1:2.4f}'.format(val_acc, val_ci95))
                        # val_accuracies.append(val_acc)

                        # train_accs, _ = meta_validation(
                        #     datasubset=train_set,
                        #     num_val_tasks=num_val_tasks,
                        #     return_uncertainty=False)
                        # train_acc = np.mean(train_accs)
                        # train_ci95 = 1.96*np.std(train_accs)/np.sqrt(num_val_tasks)
                        # print('Train accuracy = {0:2.4f} +/- {1:2.4f}\n'.format(train_acc, train_ci95))
                        # train_accuracies.append(train_acc)
                    
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
            checkpoint_filename = ('{0:s}_{1:d}way_{2:d}shot_{3:d}.pt')\
                        .format(datasource,
                                num_classes_per_task,
                                num_training_samples_per_class,
                                epoch + 1)
            print(checkpoint_filename)
            torch.save(checkpoint, os.path.join(dst_folder, checkpoint_filename))
        print()

def get_task_prediction(x_t, y_t, x_v, y_v=None):
    '''
    If y_v is not None:
        this is training
        return NLL loss
    Else:
        this is testing
        return the predicted labels y_pred_v of x_v
    '''

    # region 1st update:
    d_NLL = []
    for particle_id in range(num_particles):
        w = get_weights_target_net(w_generated=theta, row_id=particle_id, w_target_shape=w_shape)
        y_pred_t = net.forward(x=x_t, w=w, p_dropout=p_dropout_base)
        loss_NLL = loss_fn(y_pred_t, y_t)
        
        NLL_grads = torch.autograd.grad(
            outputs=loss_NLL,
            inputs=w.values(),
            create_graph=True
        )
        NLL_gradients = dict(zip(w.keys(), NLL_grads))
        NLL_gradients_tensor = dict2tensor(dict_obj=NLL_gradients)
        d_NLL.append(NLL_gradients_tensor)
    d_NLL = torch.stack(d_NLL)
    kernel_matrix, grad_kernel, _ = get_kernel(particle_tensor=theta)

    q = theta - inner_lr*(torch.matmul(kernel_matrix, d_NLL) - grad_kernel)
    # endregion

    # region 2nd updates
    for _ in range(num_inner_updates):
        d_NLL = []
        for particle_id in range(num_particles):
            w = get_weights_target_net(w_generated=q, row_id=particle_id, w_target_shape=w_shape)
            y_pred_t = net.forward(x=x_t, w=w, p_dropout=p_dropout_base)
            loss_NLL = loss_fn(y_pred_t, y_t)
            
            NLL_grads = torch.autograd.grad(
                outputs=loss_NLL,
                inputs=w.values(),
                create_graph=True
            )
            NLL_gradients = dict(zip(w.keys(), NLL_grads))
            NLL_gradients_tensor = dict2tensor(dict_obj=NLL_gradients)
            d_NLL.append(NLL_gradients_tensor)
        d_NLL = torch.stack(d_NLL)
        kernel_matrix, grad_kernel, _ = get_kernel(particle_tensor=theta)

        q = q - inner_lr*(torch.matmul(kernel_matrix, d_NLL) - grad_kernel)
    # endregion

    # region validation
    if y_v is None:
        y_pred_v = []
    else:
        loss_NLL = 0
    
    for particle_id in range(num_particles):
        w = get_weights_target_net(w_generated=q, row_id=particle_id, w_target_shape=w_shape)
        y_pred_ = net.forward(x=x_v, w=w, p_dropout=0)
        if y_v is None:
            y_pred_v.append(y_pred_)
        else:
            loss_NLL += loss_fn(y_pred_, y_v)
    
    if y_v is None:
        return y_pred_v
    else:
        loss_NLL /= num_particles
        return loss_NLL
    # endregion

def meta_validation(datasubset, num_val_tasks, return_uncertainty=False):
    if datasource == 'sine_line':
        x0 = torch.linspace(start=-5, end=5, steps=100, device=device).view(-1, 1) # vector

        if num_val_tasks == 0:
            from matplotlib import pyplot as plt
            import matplotlib
            matplotlib.rcParams['xtick.labelsize'] = 16
            matplotlib.rcParams['ytick.labelsize'] = 16
            matplotlib.rcParams['axes.labelsize'] = 18

            num_stds = 2
            data_generator = DataGenerator(
                num_samples=num_training_samples_per_class,
                device=device
            )
            if datasubset == 'sine':
                x_t, y_t, amp, phase = data_generator.generate_sinusoidal_data(noise_flag=True)
                y0 = amp*torch.sin(x0 + phase)
            else:
                x_t, y_t, slope, intercept = data_generator.generate_line_data(noise_flag=True)
                y0 = slope*x0 + intercept

            y_preds = get_task_prediction(x_t=x_t, y_t=y_t, x_v=x0)

            '''LOAD MAML DATA'''
            maml_folder = '{0:s}/MAML_mixed_sine_line'.format(dst_folder_root)
            maml_filename = 'MAML_mixed_{0:d}shot_{1:s}.pt'.format(num_training_samples_per_class, '{0:d}')

            i = 1
            maml_checkpoint_filename = os.path.join(maml_folder, maml_filename.format(i))
            while(os.path.exists(maml_checkpoint_filename)):
                i = i + 1
                maml_checkpoint_filename = os.path.join(maml_folder, maml_filename.format(i))
            print(maml_checkpoint_filename)
            maml_checkpoint = torch.load(
                os.path.join(maml_folder, maml_filename.format(i - 1)),
                map_location=lambda storage,
                loc: storage.cuda(gpu_id)
            )
            theta_maml = maml_checkpoint['theta']
            y_pred_maml = get_task_prediction_maml(x_t=x_t, y_t=y_t, x_v=x0, meta_params=theta_maml)

            '''PLOT'''
            _, ax = plt.subplots(figsize=(5, 5))
            y_top = torch.squeeze(torch.mean(y_preds, dim=0) + num_stds*torch.std(y_preds, dim=0))
            y_bottom = torch.squeeze(torch.mean(y_preds, dim=0) - num_stds*torch.std(y_preds, dim=0))

            ax.fill_between(
                x=torch.squeeze(x0).cpu().numpy(),
                y1=y_bottom.cpu().detach().numpy(),
                y2=y_top.cpu().detach().numpy(),
                alpha=0.25,
                color='C3',
                zorder=0,
                label='VAMPIRE'
            )
            ax.plot(x0.cpu().numpy(), y0.cpu().numpy(), color='C7', linestyle='-', linewidth=3, zorder=1, label='Ground truth')
            ax.plot(x0.cpu().numpy(), y_pred_maml.cpu().detach().numpy(), color='C2', linestyle='--', linewidth=3, zorder=2, label='MAML')
            ax.scatter(x=x_t.cpu().numpy(), y=y_t.cpu().numpy(), color='C0', marker='^', s=300, zorder=3, label='Data')
            plt.xticks([-5, -2.5, 0, 2.5, 5])
            plt.savefig(fname='img/mixed_sine_temp.svg', format='svg')
            return 0
        else:
            from scipy.special import erf

            quantiles = np.arange(start=0., stop=1.1, step=0.1)
            cal_data = []

            data_generator = DataGenerator(num_samples=num_training_samples_per_class, device=device)
            for _ in range(num_val_tasks):
                binary_flag = np.random.binomial(n=1, p=p_sine)
                if (binary_flag == 0):
                    # generate sinusoidal data
                    x_t, y_t, amp, phase = data_generator.generate_sinusoidal_data(noise_flag=True)
                    y0 = amp*torch.sin(x0 + phase)
                else:
                    # generate line data
                    x_t, y_t, slope, intercept = data_generator.generate_line_data(noise_flag=True)
                    y0 = slope*x0 + intercept
                y0 = y0.view(1, -1).cpu().numpy() # row vector
                
                y_preds = torch.stack(get_task_prediction(x_t=x_t, y_t=y_t, x_v=x0)) # K x len(x0)

                y_preds_np = torch.squeeze(y_preds, dim=-1).detach().cpu().numpy()
                
                y_preds_quantile = np.quantile(a=y_preds_np, q=quantiles, axis=0, keepdims=False)

                # ground truth cdf
                std = data_generator.noise_std
                cal_temp = (1 + erf((y_preds_quantile - y0)/(np.sqrt(2)*std)))/2
                cal_temp_avg = np.mean(a=cal_temp, axis=1) # average for a task
                cal_data.append(cal_temp_avg)
            return cal_data
    else:
        accuracies = []
        corrects = []
        probability_pred = []

        total_validation_samples = (num_total_samples_per_class - num_training_samples_per_class)*num_classes_per_task
        
        if datasubset == 'train':
            all_class_data = all_class_train
            embedding_data = embedding_train
        elif datasubset == 'val':
            all_class_data = all_class_val
            embedding_data = embedding_val
        elif datasubset == 'test':
            all_class_data = all_class_test
            embedding_data = embedding_test
        else:
            sys.exit('Unknown datasubset for validation')
        
        all_class_names = list(all_class_data.keys())
        all_task_names = itertools.combinations(all_class_names, r=num_classes_per_task)

        if train_flag:
            all_task_names = list(all_task_names)
            random.shuffle(all_task_names)

        task_count = 0
        for class_labels in all_task_names:
            x_t, y_t, x_v, y_v = get_task_image_data(
                all_class_data,
                embedding_data,
                class_labels,
                num_total_samples_per_class,
                num_training_samples_per_class,
                device)
            
            y_pred_v = get_task_prediction(x_t, y_t, x_v, y_v=None)
            y_pred_v = torch.stack(y_pred_v)
            y_pred_v = sm_loss(y_pred_v)
            y_pred = torch.mean(input=y_pred_v, dim=0, keepdim=False)

            prob_pred, labels_pred = torch.max(input=y_pred, dim=1)
            correct = (labels_pred == y_v)
            corrects.extend(correct.detach().cpu().numpy())

            accuracy = torch.sum(correct, dim=0).item()/total_validation_samples
            accuracies.append(accuracy)

            probability_pred.extend(prob_pred.detach().cpu().numpy())

            task_count += 1
            if not train_flag:
                print(task_count)
            if (task_count >= num_val_tasks):
                break
        if not return_uncertainty:
            return accuracies, all_task_names
        else:
            return corrects, probability_pred


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

if __name__ == "__main__":
    main()
