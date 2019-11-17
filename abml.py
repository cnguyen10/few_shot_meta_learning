'''
python3 abml.py --datasource=sine_line --k_shot=5 --n_way=1 --Lt=128 --Lv=128 --minibatch_size=10 --inner_lr=1e-3 --meta_lr=1e-3 --resume_epoch=0

python3 abml.py --datasource=miniImageNet --k_shot=1 --n_way=5 --Lt=10 --Lv=10 --minibatch_size=2 --inner_lr=1e-2 --meta_lr=1e-3 --kl_reweight=0.1 --resume_epoch=0 --num_epochs=10
'''

import torch
import torchvision

import numpy as np

import random
import os
import sys

import collections

from utils import get_num_weights, get_weights_target_net, load_dataset, get_task_image_data

import argparse
import itertools

parser = argparse.ArgumentParser(description='Setup variables for Amortized-Bayesian Meta-learning.')
parser.add_argument(
    '--datasource',
    type=str,
    default='miniImageNet',
    help='\'omniglot\' or \'miniImageNet\' or \'sine_line\''
)
parser.add_argument('--k_shot', type=int, default=1, help='Number of training samples per class')
parser.add_argument('--n_way', type=int, default=5, help='Number of classes per task')
parser.add_argument('--resume_epoch', type=int, default=0, help='Epoch id to resume learning or perform testing')

parser.add_argument('--train', dest='train_flag', action='store_true')
parser.add_argument('--test', dest='train_flag', action='store_false')
parser.set_defaults(train_flag=True)

parser.add_argument('--inner_lr', type=float, default=0.1, help='Learning rate for task-specific parameters - 0.1 for Omniglot, 0.01 for miniImageNet')
parser.add_argument('--num_inner_updates', type=int, default=5, help='Number of gradient updates for task-specific parameters')
parser.add_argument('--meta_lr', type=float, default=1e-3, help='Learning rate of meta-parameters')
parser.add_argument('--Lt', type=int, default=1, help='Number of ensemble networks to train task specific parameters')
parser.add_argument('--Lv', type=int, default=1, help='Number of ensemble networks to validate meta-parameters')
parser.add_argument('--minibatch_size', type=int, default=10, help='Number of tasks per minibatch to update meta-parameters')
parser.add_argument('--num_epochs', type=int, default=1000, help='How many 10,000 tasks are used to train?')

parser.add_argument('--lr_decay', type=float, default=1, help='Decay factor of meta-learning rate')

parser.add_argument('--kl_reweight', type=float, default=1, help='Reweight factor of the KL divergence between variational posterior q and prior p')

parser.add_argument('--p_dropout_base', type=float, default=0, help='Dropout rate of base model')

parser.add_argument('--num_val_tasks', type=int, default=100, help='Number of validation tasks')
parser.add_argument('--uncertainty', dest='uncertainty_flag', action='store_true')
parser.add_argument('--no_uncertainty', dest='uncertainty_flag', action='store_false')
parser.set_defaults(uncertainty_flag=True)
parser.add_argument('--rnd_seed', type=int, default=None, help='Random seed when picking images')

parser.add_argument('--datasubset', type=str, default='sine', help='sine or line')

args = parser.parse_args()

gpu_id = 0
device = torch.device('cuda:{0:d}'.format(gpu_id) if torch.cuda.is_available() else "cpu")
# device = torch.device('cpu')

datasource = args.datasource
train_flag = args.train_flag

num_training_samples_per_class = args.k_shot # define the number of k-shot
print('{0:d}-shot'.format(num_training_samples_per_class))

if (datasource == 'sine_line') and (not train_flag):
    num_total_samples_per_class = 1024
elif train_flag:
    num_total_samples_per_class = num_training_samples_per_class + 15 # total number of samples per class
else:
    num_total_samples_per_class = num_training_samples_per_class + 20
print('Number of total samples per class = {0:d}'.format(num_total_samples_per_class))

num_classes_per_task = args.n_way # n-way
print('{0:d}-way'.format(num_classes_per_task))

total_validation_samples = (num_total_samples_per_class - num_training_samples_per_class)*num_classes_per_task

num_tasks_per_minibatch = args.minibatch_size
num_meta_updates_print = int(1000/num_tasks_per_minibatch)
print('Dataset = {0:s}'.format(datasource))
print('Number of tasks per meta update = {0:d}'.format(num_tasks_per_minibatch))

inner_lr = args.inner_lr
print('Inner learning rate = {0}'.format(inner_lr))

expected_total_tasks_per_epoch = 10000
num_tasks_per_epoch = int(expected_total_tasks_per_epoch/num_tasks_per_minibatch)*num_tasks_per_minibatch

expected_tasks_save_loss = 2000
num_tasks_save_loss = int(expected_tasks_save_loss/num_tasks_per_minibatch)*num_tasks_per_minibatch

# num_meta_updates_print = 1

num_epochs_save = 1
num_epoch_decay_lr = 1

num_epochs = args.num_epochs

num_inner_updates = args.num_inner_updates
print('Number of inner updates = {0:d}'.format(num_inner_updates))

print('Meta generator:')
# learning rate
meta_lr = args.meta_lr
print('\t- Meta learning rate = {0}'.format(meta_lr))

lr_decay_factor = args.lr_decay

# number of ensembling networks
L = args.Lt
K = args.Lv
num_val_tasks = args.num_val_tasks
L_test = args.Lt
K_test = args.Lv
print('L = {0:d}, K = {1:d}'.format(L, K))


if datasource == 'sine_line':
    from FCNet import FCNet
    from DataGeneratorT import DataGenerator
    from utils import get_task_sine_line_data

    net = FCNet(
        num_hidden_units=(100, 100, 100),
        device=device
    )
    p_sine = 0.5
elif datasource == 'miniImageNet_embedding':
    from FC640 import FC640

    net = FC640(
        dim_output=num_classes_per_task,
        num_hidden_units=(128,),
        device=device
    )
elif datasource == 'miniImageNet':
    from ConvNet import ConvNet

    net = ConvNet(
        dim_input=(3, 84, 84),
        dim_output=num_classes_per_task,
        num_filters=[32]*4
    )

w_target_shape = net.get_weight_shape()
# print(w_target_shape)
num_weights = get_num_weights(my_net=net)
print('Number of weights of base model = \t {0:d}'.format(num_weights))

dst_folder_root = '.'

dst_folder = '{0:s}/Amortised_ML_few_shot_meta/Amortised_ML_{1:s}_{2:d}way_{3:d}shot'.format(
    dst_folder_root,
    datasource,
    num_classes_per_task,
    num_training_samples_per_class
)
if not os.path.exists(dst_folder):
    os.makedirs(dst_folder)
    print('Create folder to store weights')
    print(dst_folder)

train_set = 'train'
val_set = 'val'
test_set = 'test'

KL_reweight = args.kl_reweight
# KL_reweight = 0.01#float(2/num_weights)
print('KL reweight = {0:1.2e}'.format(KL_reweight))

p_base_dropout = args.p_dropout_base

print('Dropout:')
print('\t - Base:\t {0}'.format(p_base_dropout))

# loss functions
if datasource == 'sine_line':
    loss_fn = torch.nn.MSELoss()
    L1_loss = torch.nn.L1Loss()
else:
    loss_fn = torch.nn.CrossEntropyLoss()
sm_loss = torch.nn.Softmax(dim=2)

resume_epoch = args.resume_epoch
if resume_epoch == 0:
    # initialise meta-parameters
    theta = {}
    theta['mean'] = {}
    theta['logSigma'] = {}
    for key in w_target_shape.keys():
        if 'b' in key:
            theta['mean'][key] = torch.zeros(w_target_shape[key], device=device, requires_grad=True)
        else:
            theta['mean'][key] = torch.empty(w_target_shape[key], device=device)
            torch.nn.init.xavier_normal_(tensor=theta['mean'][key], gain=1.)
            theta['mean'][key].requires_grad_()
        theta['logSigma'][key] = torch.rand(w_target_shape[key], device=device) - 4
        theta['logSigma'][key].requires_grad_()
else:
    print('Restore previous theta...')
    print('Resume epoch {0:d}'.format(resume_epoch))
    checkpoint_filename = '{0:s}_{1:d}way_{2:d}shot_{3:d}.pt'.format(
        datasource,
        num_classes_per_task,
        num_training_samples_per_class,
        resume_epoch
    )
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
    [
        {
            'params': theta['mean'].values(),
            'weight_decay': 0
        },
        {
            'params': theta['logSigma'].values()
        }
    ],
    lr=meta_lr
)

if resume_epoch > 0:
    op_theta.load_state_dict(saved_checkpoint['op_theta'])
    op_theta.param_groups[0]['lr'] = meta_lr
    op_theta.param_groups[1]['lr'] = meta_lr

    del saved_checkpoint

print(op_theta)

uncertainty_flag = args.uncertainty_flag

if datasource != 'sine_line':
    # pre-load dataset
    print('Load dataset...')
    if train_flag:
        all_class_train, embedding_train = load_dataset(datasource, train_set)
        all_class_val, embedding_val = load_dataset(datasource, val_set)
    else:
        all_class_test, embedding_test = load_dataset(datasource, test_set)
    print('Done')
print()
# -------------------------------------------------------------------------------------------------
def main():
    if train_flag:
        meta_train()
    elif resume_epoch > 0:
        if datasource == 'sine_line':
            cal_data = meta_validation(datasubset=args.datasubset, num_val_tasks=num_val_tasks)

            if num_val_tasks > 0:
                cal_data = np.array(cal_data)
                np.savetxt(fname='vampire_{0:s}_calibration.csv'.format(datasource), X=cal_data, delimiter=',')
        else:
            if not uncertainty_flag:
                accs, all_task_names = meta_validation(
                    datasubset=test_set,
                    num_val_tasks=num_val_tasks,
                    return_uncertainty=uncertainty_flag
                )
                with open(file='aml_{0:s}_{1:d}_{2:d}_accuracies.csv'.format(datasource, num_classes_per_task, num_training_samples_per_class), mode='w') as result_file:
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
                with open(file='aml_{0:s}_correct_prob.csv'.format(datasource), mode='w') as result_file:
                    for correct, prob in zip(corrects, probs):
                        result_file.write('{0}, {1}\n'.format(correct, prob))
                        # print(correct, prob)
    else:
        sys.exit('Unknown action')

def meta_train():
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
        # all_class.update(all_class_val)
        embedding = embedding_train
        # embedding.update(embedding_val)

        sampler = torch.utils.data.sampler.RandomSampler(data_source=list(all_class.keys()), replacement=False)

        train_loader = torch.utils.data.DataLoader(
            dataset=list(all_class.keys()),
            batch_size=num_classes_per_task,
            sampler=sampler,
            drop_last=True
        )
    
    for epoch in range(resume_epoch, resume_epoch + num_epochs):
        # variables used to store information of each epoch for monitoring purpose
        meta_loss_saved = [] # meta loss to save
        kl_loss_saved = []
        val_accuracies = []
        train_accuracies = []

        task_count = 0 # a counter to decide when a minibatch of task is completed to perform meta update
        meta_loss = 0 # accumulate the loss of many ensambling networks to descent gradient for meta update
        num_meta_updates_count = 0

        meta_loss_avg_print = 0 # compute loss average to print

        kl_loss = 0
        kl_loss_avg_print = 0

        meta_loss_avg_save = [] # meta loss to save
        kl_loss_avg_save = []

        task_count = 0
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
                loss_NLL, KL_loss = get_task_prediction(
                    x_t=x_t,
                    y_t=y_t,
                    x_v=x_v,
                    y_v=y_v,
                    p_dropout=p_base_dropout
                )

                meta_loss = meta_loss + loss_NLL

                kl_loss = kl_loss + KL_loss
                task_count = task_count + 1

                if torch.isnan(meta_loss).item():
                    sys.exit('nan')

                if (task_count % num_tasks_per_minibatch == 0):
                    # average over the number of tasks per minibatch
                    meta_loss = meta_loss/num_tasks_per_minibatch
                    kl_loss = kl_loss/num_tasks_per_minibatch

                    # accumulate for printing purpose
                    meta_loss_avg_print += meta_loss.item()
                    kl_loss_avg_print += kl_loss.item()

                    op_theta.zero_grad()
                    meta_loss.backward(retain_graph=True)
                    # torch.nn.utils.clip_grad_norm_(parameters=theta.values(), max_norm=1)
                    op_theta.step()
                    
                    # Printing losses
                    num_meta_updates_count += 1
                    if (num_meta_updates_count % num_meta_updates_print == 0):
                        meta_loss_avg_save.append(meta_loss_avg_print/num_meta_updates_count)
                        kl_loss_avg_save.append(kl_loss_avg_print/num_meta_updates_count)
                        print('{0:d}, {1:2.4f}, {2:1.4f}'.format(
                            task_count,
                            meta_loss_avg_save[-1],
                            kl_loss_avg_save[-1]
                        ))
                        num_meta_updates_count = 0

                        meta_loss_avg_print = 0
                        kl_loss_avg_print = 0
                        
                    if (task_count % num_tasks_save_loss == 0):
                        meta_loss_saved.append(np.mean(meta_loss_avg_save))
                        kl_loss_saved.append(np.mean(kl_loss_avg_save))

                        meta_loss_avg_save = []
                        kl_loss_avg_save = []
                        # if datasource != 'sine_line':
                        #     val_accs = meta_validation(
                        #         datasubset=val_set,
                        #         num_val_tasks=num_val_tasks)
                        #     val_acc = np.mean(val_accs)
                        #     val_ci95 = 1.96*np.std(val_accs)/np.sqrt(num_val_tasks)
                        #     print('Validation accuracy = {0:2.4f} +/- {1:2.4f}'.format(val_acc, val_ci95))
                        #     val_accuracies.append(val_acc)

                        #     train_accs = meta_validation(
                        #         datasubset=train_set,
                        #         num_val_tasks=num_val_tasks)
                        #     train_acc = np.mean(train_accs)
                        #     train_ci95 = 1.96*np.std(train_accs)/np.sqrt(num_val_tasks)
                        #     print('Train accuracy = {0:2.4f} +/- {1:2.4f}\n'.format(train_acc, train_ci95))
                        #     train_accuracies.append(train_acc)

                    # reset meta loss for the next minibatch of tasks
                    meta_loss = 0
                    kl_loss = 0

                if (task_count >= num_tasks_per_epoch):
                    break
        if ((epoch + 1)% num_epochs_save == 0):
            checkpoint = {
                'theta': theta,
                'meta_loss': meta_loss_saved,
                'kl_loss': kl_loss_saved,
                'val_accuracy': val_accuracies,
                'train_accuracy': train_accuracies,
                'op_theta': op_theta.state_dict(),
            }
            print('SAVING WEIGHTS...')
            checkpoint_filename = ('{0:s}_{1:d}way_{2:d}shot_{3:d}.pt')\
                        .format(datasource,
                                num_classes_per_task,
                                num_training_samples_per_class,
                                epoch + 1)
            print(checkpoint_filename)
            torch.save(checkpoint, os.path.join(dst_folder, checkpoint_filename))
            # scheduler.step()
        print()

def get_task_prediction(x_t, y_t, x_v, y_v=None, p_dropout=0):
    '''1st update with initialisation from theta'''
    q = initialise_dict_of_dict(theta['mean'].keys())

    loss_vfe = 0
    for _ in range(L):
        w = generate_weights(meta_params=theta)
        y_pred_t = net.forward(x=x_t, w=w, p_dropout=p_dropout)
        # loss_t = loss_fn(y_pred_t, y_t)
        loss_vfe = loss_vfe + loss_fn(y_pred_t, y_t)
    loss_vfe = loss_vfe/L
    grads_mean = torch.autograd.grad(
        outputs=loss_vfe,
        inputs=theta['mean'].values(),
        create_graph=True
    )
    grads_logSigma = torch.autograd.grad(
        outputs=loss_vfe,
        inputs=theta['logSigma'].values(),
        create_graph=True
    )
    gradients_mean = dict(zip(theta['mean'].keys(), grads_mean))
    gradients_logSigma = dict(zip(theta['logSigma'].keys(), grads_logSigma))

    for key in w.keys():
        q['mean'][key] = theta['mean'][key] - inner_lr*gradients_mean[key]
        q['logSigma'][key] = theta['logSigma'][key] - inner_lr*gradients_logSigma[key]

    '''2nd update'''
    for _ in range(num_inner_updates - 1):
        loss_vfe = 0
        for _ in range(L):
            w = generate_weights(meta_params=q)
            y_pred_t = net.forward(x=x_t, w=w, p_dropout=p_dropout)
            loss_vfe = loss_vfe + loss_fn(y_pred_t, y_t)
        loss_vfe = loss_vfe/L
        grads_mean = torch.autograd.grad(
            outputs=loss_vfe,
            inputs=q['mean'].values(),
            retain_graph=True
        )
        grads_logSigma = torch.autograd.grad(
            outputs=loss_vfe,
            inputs=q['logSigma'].values(),
            retain_graph=True
        )
        gradients_mean = dict(zip(q['mean'].keys(), grads_mean))
        gradients_logSigma = dict(zip(q['logSigma'].keys(), grads_logSigma))

        for key in w.keys():
            q['mean'][key] = q['mean'][key] - inner_lr*(gradients_mean[key] \
                - KL_reweight*torch.exp(-2*theta['logSigma'][key])*(theta['mean'][key] - q['mean'][key]))
            q['logSigma'][key] = q['logSigma'][key] - inner_lr*(gradients_logSigma[key] \
                + KL_reweight*(torch.exp(2*(q['logSigma'][key] - theta['logSigma'][key])) - 1))

    '''Task prediction'''
    if y_v is None:
        y_pred_v = [] #torch.zeros((x_v.shape[0], num_classes_per_task), device=device)
    else:
        loss_NLL = 0
    for _ in range(K):
        w = generate_weights(meta_params=q)
        if train_flag:
            y_pred_ = net.forward(x=torch.cat((x_t, x_v)), w=w, p_dropout=0)
        else:
            y_pred_ = net.forward(x=x_v, w=w, p_dropout=0)

        if y_v is not None:
            loss_NLL = loss_NLL + loss_fn(y_pred_, torch.cat((y_t, y_v)))
        else:
            y_pred_v.append(y_pred_)
    
    if y_v is None:
        return torch.stack(y_pred_v)
    else:
        loss_NLL = loss_NLL/K

    KL_loss = 0
    for key in q['mean'].keys():
        KL_loss += torch.sum(torch.exp(2*(q['logSigma'][key] - theta['logSigma'][key])) \
            + (theta['mean'][key] - q['mean'][key])**2/torch.exp(2*theta['logSigma'][key]))\
                + torch.sum(2*(theta['logSigma'][key] - q['logSigma'][key]))
    KL_loss = (KL_loss - num_weights)/2
    return loss_NLL, KL_loss

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
                
                y_preds = get_task_prediction(x_t=x_t, y_t=y_t, x_v=x0) # K x len(x0)

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
        all_task_names = list(itertools.combinations(all_class_names, r=num_classes_per_task))

        if train_flag:
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
            
            y_pred_v = sm_loss(get_task_prediction(x_t, y_t, x_v, y_v=None))
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
            if task_count >= num_val_tasks:
                break
        if not return_uncertainty:
            return accuracies, all_task_names
        else:
            return corrects, probability_pred

def get_task_prediction_maml(x_t, y_t, x_v, meta_params):
    q = {}

    y_pred_t = net.forward(x=x_t, w=meta_params)
    loss_vfe = loss_fn(y_pred_t, y_t)

    grads = torch.autograd.grad(
        outputs=loss_vfe,
        inputs=meta_params.values(),
        create_graph=True
    )
    gradients = dict(zip(meta_params.keys(), grads))

    for key in meta_params.keys():
        q[key] = meta_params[key] - inner_lr*gradients[key]

    '''2nd update'''
    for _ in range(num_inner_updates - 1):
        loss_vfe = 0
        y_pred_t = net.forward(x=x_t, w=q, p_dropout=p_base_dropout)
        loss_vfe = loss_fn(y_pred_t, y_t)
        grads = torch.autograd.grad(
            outputs=loss_vfe,
            inputs=q.values(),
            retain_graph=True
        )
        gradients = dict(zip(q.keys(), grads))

        for key in q.keys():
            q[key] = q[key] - inner_lr*gradients[key]
    
    '''Task prediction'''
    y_pred_v = net.forward(x=x_v, w=q, p_dropout=0)
    
    return y_pred_v

def generate_weights(meta_params):
    w = {}
    for key in meta_params['mean'].keys():
        eps_sampled = torch.randn(meta_params['mean'][key].shape, device=device)
        w[key] = meta_params['mean'][key] + eps_sampled*torch.exp(meta_params['logSigma'][key])

    return w

def initialise_dict_of_dict(key_list):
    q = dict.fromkeys(['mean', 'logSigma'])
    for para in q.keys():
        q[para] = {}
        for key in key_list:
            q[para][key] = 0
    return q


if __name__ == "__main__":
    main()
