'''
python3 simba.py --datasource=sine_line --n_way=1 --k_shot=5 --Lt=32 --Lv=32 --delta=0.01 --inner_lr=1e-3 --meta_lr=1e-4 --discriminator_inner_lr=1e-4 --discriminator_meta_lr=1e-5 --minibatch_size=50 --discriminator_minibatch_size=1024 --encoder_lr=1e-4 --resume_epoch=0

python3 simba.py --datasource=sine_line --n_way=1 --k_shot=5 --Lt=64 --Lv=64 --inner_lr=1e-3 --discriminator_inner_lr=1e-4 --discriminator_minibatch_size=1024 --resume_epoch=22 --test --num_val_tasks=0 --sine_or_line=sine


python3 simba.py --datasource=miniImageNet --n_way=5 --k_shot=1 --Lt=8 --Lv=8 --delta=1e-1 --inner_lr=1e-2 --meta_lr=1e-4 --L2_regularization=1e-5 --discriminator_inner_lr=1e-4 --discriminator_meta_lr=1e-5 --minibatch_size=2 --discriminator_minibatch_size=1024 --encoder_lr=1e-4 --clip_grad_value=10 --resume_epoch=14

python3 simba.py --datasource=miniImageNet --n_way=5 --k_shot=1 --Lt=8 --Lv=8 --delta=1e-1 --inner_lr=1e-2 --meta_lr=1e-4 --L2_regularization=1e-5 --discriminator_inner_lr=1e-4 --discriminator_meta_lr=1e-5 --minibatch_size=2 --discriminator_minibatch_size=1024 --encoder_lr=1e-4 --clip_grad_value=10 --resume_epoch=14 --test --no_uncertainty --num_val_tasks=15504


python3 simba.py --datasource=miniImageNet_embedding --n_way=5 --k_shot=1 --Lt=32 --Lv=32 --delta=1e-1 --inner_lr=1e-2 --meta_lr=1e-4 --L2_regularization=1e-5 --discriminator_inner_lr=1e-4 --discriminator_meta_lr=1e-5 --minibatch_size=2 --discriminator_minibatch_size=1024 --encoder_lr=1e-4 --clip_grad_value=10 --p_dropout_base=0.3 --resume_epoch=0 --num_epochs=5

python3 simba.py --datasource=tieredImageNet_embedding --n_way=5 --k_shot=1 --Lt=32 --Lv=32 --delta=1e-1 --inner_lr=1e-2 --meta_lr=1e-4 --L2_regularization=1e-5 --discriminator_inner_lr=1e-4 --discriminator_meta_lr=1e-5 --minibatch_size=2 --discriminator_minibatch_size=1024 --encoder_lr=1e-4 --clip_grad_value=10 --p_base_dropout=0.3 --resume_epoch=0
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

parser = argparse.ArgumentParser(description='Setup variables for VAMPIRE.')
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
parser.add_argument('--L2_regularization', type=float, default=0, help='L2 regularization for theta')
parser.add_argument('--Lt', type=int, default=1, help='Number of ensemble networks to train task specific parameters')
parser.add_argument('--Lv', type=int, default=1, help='Number of ensemble networks to validate meta-parameters')
parser.add_argument('--minibatch_size', type=int, default=10, help='Number of tasks per minibatch to update meta-parameters')
parser.add_argument('--num_epochs', type=int, default=1000, help='How many 10,000 tasks are used to train?')

parser.add_argument('--lr_decay', type=float, default=1, help='Decay factor of meta-learning rate')

parser.add_argument('--discriminator_inner_lr', type=float, default=1e-3, help='Discriminator inner learning rate')
parser.add_argument('--discriminator_meta_lr', type=float, default=1e-5, help='Learning of meta-discriminator')
parser.add_argument('--discriminator_minibatch_size', type=int, default=512, help='Number of weights sampled to train the discriminator')

parser.add_argument('--encoder_lr', type=float, default=1e-4, help='Learning rate of task-encoder')

parser.add_argument('--kl_reweight', type=float, default=1, help='Reweight factor of the KL divergence between variational posterior q and prior p')

parser.add_argument('--delta', type=float, default=0.1, help='Delta in PAC-Bayes bound')

parser.add_argument('--clip_grad_value', type=float, default=1., help='Clip gradient value for task adaptation')

parser.add_argument('--p_dropout_base', type=float, default=0, help='Dropout rate of base model')
parser.add_argument('--p_dropout_generator', type=float, default=0, help='Dropout rate of generator model')
parser.add_argument('--p_dropout_discriminator', type=float, default=0, help='Dropout rate of discriminator model')
parser.add_argument('--p_dropout_encoder', type=float, default=0, help='Dropout rate of encoder model')

parser.add_argument('--num_val_tasks', type=int, default=100, help='Number of validation tasks')
parser.add_argument('--uncertainty', dest='uncertainty_flag', action='store_true')
parser.add_argument('--no_uncertainty', dest='uncertainty_flag', action='store_false')
parser.set_defaults(uncertainty_flag=True)
parser.add_argument('--rnd_seed', type=int, default=None, help='Random seed when picking images')

parser.add_argument('--sine_or_line', type=str, default='sine', help='Specify sine or line to plot')

args = parser.parse_args()

gpu_id = 0
device = torch.device('cuda:{0:d}'.format(gpu_id) if torch.cuda.is_available() else torch.device('cpu'))
# device = torch.device('cpu')

datasource = args.datasource
train_flag = args.train_flag

num_training_samples_per_class = args.k_shot # define the number of k-shot
print('{0:d}-shot'.format(num_training_samples_per_class))

if (datasource == 'sine_line') and (not train_flag):
    num_total_samples_per_class = 1024
else:
    num_total_samples_per_class = num_training_samples_per_class + 15 # total number of samples per class
print('Number of total samples per class = {0:d}'.format(num_total_samples_per_class))

num_classes_per_task = args.n_way # n-way
print('{0:d}-way'.format(num_classes_per_task))

total_validation_samples = (num_total_samples_per_class - num_training_samples_per_class)*num_classes_per_task

num_tasks_per_minibatch = args.minibatch_size
num_meta_updates_print = int(100/num_tasks_per_minibatch)
print('Dataset = {0:s}'.format(datasource))
print('Number of tasks per meta update = {0:d}'.format(num_tasks_per_minibatch))

inner_lr = args.inner_lr
print('Inner learning rate = {0}'.format(inner_lr))

expected_total_tasks_per_epoch = 10000
num_tasks_per_epoch = int(expected_total_tasks_per_epoch/num_tasks_per_minibatch)*num_tasks_per_minibatch

expected_tasks_save_loss = expected_total_tasks_per_epoch
num_tasks_save_loss = int(expected_tasks_save_loss/num_tasks_per_minibatch)*num_tasks_per_minibatch

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

encoder_lr = args.encoder_lr
print('Encoder learning rate = {0:1.2e}'.format(encoder_lr))

if datasource == 'sine_line':
    from FCNet import FCNet
    from DataGeneratorT import DataGenerator
    from WGenerator2 import WeightDiscriminator, WeightGenerator
    from utils import get_task_sine_line_data

    net = FCNet(
        dim_input=1,
        dim_output=1,
        num_hidden_units=(40, 40),
        device=device
    )
    p_sine = 0.5

    z_dim = 40
    encoder = FCNet(
        dim_input=1,
        dim_output=z_dim,
        num_hidden_units=(40, 40),
        device=device
    )

    generator_hidden_units = (128, 512)
    tanh_scale = 1
    label_eps = 0

    L2_regularization_discriminator = 0

elif datasource == 'miniImageNet':
    from ConvNet import ConvNet
    from WGenerator2 import WeightDiscriminator, WeightGenerator

    net = ConvNet(
        dim_input=(3, 84, 84),
        dim_output=num_classes_per_task,
        num_filters=[32]*4,
        bn_flag=True
    )

    encoder = ConvNet(
        dim_input=(3, 84, 84),
        dim_output=None,
        num_filters=[32]*5,
        bn_flag=False,
        device=device
    )

    z_dim = 128

    generator_hidden_units = (256, 512)
    tanh_scale = 20
    label_eps = 0.1

    L2_regularization_discriminator = 0

elif datasource in ['miniImageNet_embedding', 'tieredImageNet_embedding']:
    from FC640 import FC640
    from WGenerator2 import WeightDiscriminator, WeightGenerator
    net = FC640(
        dim_output=num_classes_per_task,
        num_hidden_units=(128, 32),
        device=device
    )

    z_dim = 128

    encoder = FC640(
        dim_output=z_dim,
        num_hidden_units=(256, 128),
        device=device
    )

    generator_hidden_units = (256, 512)
    tanh_scale = 20
    L2_regularization_discriminator = 0
    label_eps = 0.1

# set the clipping value
clip_grad_value = args.clip_grad_value

# base net/target net
w_target_shape = net.get_weight_shape()
num_weights = get_num_weights(my_net=net)
print('Number of weights of base model = \t {0:d}'.format(num_weights))

dst_folder_root = '.'

dst_folder = '{0:s}/SImBa_few_shot_meta/SImBa_{1:s}_{2:d}way_{3:d}shot'.format(
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

kl_reweight = args.kl_reweight
print('KL reweight = {0:1.2e}'.format(kl_reweight))

delta = args.delta
print('Confident parameter delta = {0:1.2e}'.format(delta))

L2_regularization = args.L2_regularization

p_dropout_base = args.p_dropout_base
p_dropout_generator = args.p_dropout_generator
p_dropout_discriminator = args.p_dropout_discriminator
p_dropout_encoder = args.p_dropout_encoder

print('Dropout:')
print('\t - Base:\t {0}'.format(p_dropout_base))
print('\t - Generator:\t {0}'.format(p_dropout_generator))
print('\t - Discriminator:\t {0}'.format(p_dropout_discriminator))
print('\t - Encoder:\t {0}'.format(p_dropout_encoder))

# discriminator
print('\nDiscriminator:')
discriminator_inner_lr = args.discriminator_inner_lr
discriminator_meta_lr = args.discriminator_meta_lr
# discriminator_meta_l2_regularization = args.discriminator_meta_l2_regularization
discriminator_minibatch_size = args.discriminator_minibatch_size

print('\t- inner learning rate = {0}'.format(discriminator_inner_lr))
print('\t- meta learning rate = {0}'.format(discriminator_meta_lr))
print('\t- minibatch of weights = {0:d}'.format(discriminator_minibatch_size))

real_label = 1 - label_eps
fake_label = 0 + label_eps

# loss functions
if datasource == 'sine_line':
    loss_fn = torch.nn.MSELoss()
else:
    loss_fn = torch.nn.CrossEntropyLoss()
BCE_with_logit_loss = torch.nn.BCEWithLogitsLoss()
sm_loss = torch.nn.Softmax(dim=2)

wGenerator = WeightGenerator(
    z_dim=z_dim,
    dim_output=num_weights,
    num_hidden_units=generator_hidden_units,
    tanh_scale=tanh_scale,
    device=device
)
wDiscriminator = WeightDiscriminator(
    z_dim=z_dim,
    dim_input=num_weights,
    num_hidden_units=generator_hidden_units,
    device=device
)

resume_epoch = args.resume_epoch
if resume_epoch == 0:
    # initialise the generator
    theta = wGenerator.initialise_generator() # theta = weights of the generator

    # initialise the discriminator
    w_discriminator = wDiscriminator.initialise_discriminator() # weights of the discriminator

    # initialize weights for the task-encoder
    w_encoder = encoder.initialise_weights()
    w_encoder_2 = encoder.initialise_weights()
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
    w_discriminator = saved_checkpoint['w_discriminator']
    w_encoder = saved_checkpoint['w_encoder']
    w_encoder_2 = saved_checkpoint['w_encoder_2']

if train_flag:
    op_theta = torch.optim.Adam([
        {
            'params': theta.values(),
            'lr': meta_lr,
            'weight_decay': 0
        },
        {
            'params': w_encoder.values(),
            'lr': encoder_lr,
            'weight_decay': 0
        },
        {
            'params': w_encoder_2.values(),
            'lr': encoder_lr
        }
    ])

    op_discriminator = torch.optim.Adam(
        params=w_discriminator.values(),
        lr=discriminator_meta_lr,
        weight_decay=L2_regularization_discriminator
    )

    if resume_epoch > 0:
        op_theta.load_state_dict(saved_checkpoint['op_theta'])
        op_theta.param_groups[0]['lr'] = meta_lr
        op_theta.param_groups[1]['lr'] = encoder_lr
        op_theta.param_groups[2]['lr'] = encoder_lr

        op_discriminator.load_state_dict(saved_checkpoint['op_discriminator'])
        op_discriminator.param_groups[0]['lr'] = discriminator_meta_lr
        op_discriminator.param_groups[0]['weight_decay'] = L2_regularization_discriminator

        del saved_checkpoint

    print(op_theta)
    print(op_discriminator)

# scheduler = torch.optim.lr_scheduler.ExponentialLR(
#     optimizer=op_theta,
#     gamma=args.lr_decay)

uncertainty_flag = args.uncertainty_flag

if datasource != 'sine_line':
    # pre-load dataset
    print('Load dataset...')
    if train_flag:
        all_class_train, embedding_train = load_dataset(datasource, train_set)
        # all_class_val, embedding_val = load_dataset(datasource, val_set)
    else:
        all_class_test, embedding_test = load_dataset(datasource, test_set)
    print('Done')
print()

r0_const = np.log(num_tasks_per_minibatch*(num_tasks_per_minibatch + 1)/delta)
ri_const = np.log((num_tasks_per_minibatch + 1)*total_validation_samples/delta)

# d_loss_const = 2*np.log(2)

# -------------------------------------------------------------------------------------------------
def main():
    if train_flag:
        meta_train()
    elif resume_epoch > 0:
        if datasource == 'sine_line':
            cal_avg = meta_validation(datasubset=args.sine_or_line, num_val_tasks=num_val_tasks)

            if num_val_tasks > 0:
                print(cal_avg)
        else:
            if not uncertainty_flag:
                accs, all_task_names = meta_validation(
                    datasubset=test_set,
                    num_val_tasks=num_val_tasks,
                    return_uncertainty=uncertainty_flag
                )
                with open(file='simba_{0:s}_{1:d}_{2:d}_accuracies.csv'.format(datasource, num_classes_per_task, num_training_samples_per_class), mode='w') as result_file:
                    for acc, classes_in_task in zip(accs, all_task_names):
                        row_str = ''
                        for class_in_task in classes_in_task:
                            row_str = '{0}{1},'.format(row_str, class_in_task)
                        result_file.write('{0}{1}\n'.format(row_str, acc))

                print('Accuracy = {0:2.4f} +/- {1:2.4f}'.format(np.mean(accs), 1.96*np.std(accs)/np.sqrt(len(accs))))
            else:
                corrects, probs = meta_validation(
                    datasubset=test_set,
                    num_val_tasks=num_val_tasks,
                    return_uncertainty=uncertainty_flag
                )
                with open(file='simba_{0:s}_correct_prob.csv'.format(datasource), mode='w') as result_file:
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
        loss_NLL_saved = []
        kl_loss_saved = []
        d_loss_saved = []
        val_accuracies = []
        train_accuracies = []

        task_count = 0 # a counter to decide when a minibatch of task is completed to perform meta update
        meta_loss = 0 # accumulate the loss of many ensambling networks to descent gradient for meta update
        num_meta_updates_count = 0

        loss_NLL_v = 0
        loss_NLL_avg_print = 0

        kl_loss = 0
        kl_loss_avg_print = 0

        d_loss = 0
        d_loss_avg_print = 0

        loss_NLL_avg_save = []
        kl_loss_avg_save = []
        d_loss_avg_save = []

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
                loss_NLL, KL_loss, discriminator_loss = get_task_prediction(
                    x_t=x_t,
                    y_t=y_t,
                    x_v=x_v,
                    y_v=y_v,
                    p_dropout=p_dropout_base,
                    p_dropout_g=p_dropout_generator,
                    p_dropout_d=p_dropout_discriminator,
                    p_dropout_e=p_dropout_encoder
                )

                if torch.isnan(loss_NLL).item():
                    sys.exit('nan')

                loss_NLL_v += loss_NLL.item()

                if (loss_NLL.item() > 1):
                    loss_NLL.data = torch.tensor([1.], device=device)

                # if (discriminator_loss.item() > d_loss_const):
                #     discriminator_loss.data = torch.tensor([d_loss_const], device=device)

                kl_loss = kl_loss + KL_loss

                if (KL_loss.item() < 0):
                    KL_loss.data = torch.tensor([0.], device=device)
                
                Ri = torch.sqrt((KL_loss + ri_const)/(2*(total_validation_samples - 1)))
                meta_loss = meta_loss + loss_NLL + Ri

                d_loss = d_loss + discriminator_loss

                task_count = task_count + 1

                if (task_count % num_tasks_per_minibatch == 0):
                    # average over the number of tasks per minibatch
                    meta_loss = meta_loss/num_tasks_per_minibatch
                    loss_NLL_v /= num_tasks_per_minibatch
                    kl_loss = kl_loss/num_tasks_per_minibatch
                    d_loss = d_loss/num_tasks_per_minibatch

                    # accumulate for printing purpose
                    loss_NLL_avg_print += loss_NLL_v
                    kl_loss_avg_print += kl_loss.item()
                    d_loss_avg_print += d_loss.item()

                    # adding R0
                    R0 = 0
                    for key in theta.keys():
                        R0 += theta[key].norm(2)
                    R0 = torch.sqrt((L2_regularization*R0 + r0_const)/(2*(num_tasks_per_minibatch - 1)))
                    meta_loss += R0

                    # optimize theta
                    op_theta.zero_grad()
                    meta_loss.backward(retain_graph=True)
                    torch.nn.utils.clip_grad_norm_(parameters=theta.values(), max_norm=clip_grad_value)
                    torch.nn.utils.clip_grad_norm_(parameters=w_encoder.values(), max_norm=clip_grad_value)
                    op_theta.step()

                    # optimize the discriminator
                    op_discriminator.zero_grad()
                    d_loss.backward()
                    torch.nn.utils.clip_grad_norm_(parameters=w_discriminator.values(), max_norm=clip_grad_value)
                    op_discriminator.step()
                    
                    # Printing losses
                    num_meta_updates_count += 1
                    if (num_meta_updates_count % num_meta_updates_print == 0):
                        loss_NLL_avg_save.append(loss_NLL_avg_print/num_meta_updates_count)
                        kl_loss_avg_save.append(kl_loss_avg_print/num_meta_updates_count)
                        d_loss_avg_save.append(d_loss_avg_print/num_meta_updates_count)
                        print('{0:d}, {1:2.4f}, {2:1.4f}, {3:2.4e}'.format(
                            task_count,
                            loss_NLL_avg_save[-1],
                            kl_loss_avg_save[-1],
                            d_loss_avg_save[-1]
                        ))
                        num_meta_updates_count = 0

                        loss_NLL_avg_print = 0
                        kl_loss_avg_print = 0
                        d_loss_avg_print = 0
                        
                    if (task_count % num_tasks_save_loss == 0):
                        loss_NLL_saved.append(np.mean(loss_NLL_avg_save))
                        kl_loss_saved.append(np.mean(kl_loss_avg_save))
                        d_loss_saved.append(np.mean(d_loss_avg_save))

                        loss_NLL_avg_save = []
                        kl_loss_avg_save = []
                        d_loss_avg_save = []
                        
                        # if datasource != 'sine_line':
                        #     val_accs = meta_validation(
                        #         datasubset=val_set,
                        #         num_val_tasks=num_val_tasks)
                        #     val_acc = np.mean(val_accs)
                        #     val_ci95 = 1.96*np.std(val_accs)/np.sqrt(num_val_tasks)
                        #     print('Validation accuracy = {0:2.4f} +/- {1:2.4f}'.format(val_acc, val_ci95))
                        #     val_accuracies.append(val_acc)

                            # train_accs = meta_validation(
                            #     datasubset=train_set,
                            #     num_val_tasks=num_val_tasks)
                            # train_acc = np.mean(train_accs)
                            # train_ci95 = 1.96*np.std(train_accs)/np.sqrt(num_val_tasks)
                            # print('Train accuracy = {0:2.4f} +/- {1:2.4f}\n'.format(train_acc, train_ci95))
                            # train_accuracies.append(train_acc)

                    # reset meta loss for the next minibatch of tasks
                    meta_loss = 0
                    kl_loss = 0
                    d_loss = 0
                    loss_NLL_v = 0

                if (task_count >= num_tasks_per_epoch):
                    break
        if ((epoch + 1)% num_epochs_save == 0):
            checkpoint = {
                'w_discriminator': w_discriminator,
                'theta': theta,
                'w_encoder': w_encoder,
                'w_encoder_2': w_encoder_2,
                'meta_loss': loss_NLL_saved,
                'kl_loss': kl_loss_saved,
                'd_loss': d_loss_saved,
                'val_accuracy': val_accuracies,
                'train_accuracy': train_accuracies,
                'op_theta': op_theta.state_dict(),
                'op_discriminator': op_discriminator.state_dict()
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

def get_task_prediction(x_t, y_t, x_v, y_v=None, p_dropout=0, p_dropout_g=0, p_dropout_d=0, p_dropout_e=0):
    q = {}
    
    # region p(z|x)
    # encode unlabelled data into parameters of Dirichlet
    # x_z = encoder.forward(x=torch.cat(tensors=(x_t, x_v)), w=w_encoder, p_dropout=p_dropout_e)
    # x_z_2 = encoder.forward(x=torch.cat(tensors=(x_t, x_v)), w=w_encoder_2, p_dropout=p_dropout_e)
    x_z = encoder.forward(x=x_t, w=w_encoder, p_dropout=p_dropout_e)
    x_z_2 = encoder.forward(x=x_t, w=w_encoder_2, p_dropout=p_dropout_e)
    x_z = torch.nn.functional.softplus(x_z)
    x_z_2 = torch.nn.functional.softplus(x_z_2)

    # # permutation invariance
    alpha_z = torch.mean(x_z, dim=0)
    alpha_z_2 = torch.mean(x_z_2, dim=0)

    # # define auto-encoded distribution
    p_z = torch.distributions.beta.Beta(concentration0=alpha_z, concentration1=alpha_z_2)

    # endregion

    '''1st update with initialisation from theta'''
    # sample z from Dirichlet distribution
    z = p_z.rsample(sample_shape=(L,))
    # generate a list of weights of the base/targeted network
    w_ps = wGenerator.forward(z, theta, p_dropout=p_dropout_g)
    
    loss_NLL = 0
    for l in range(L):
        # get each weight set of the base/targeted network
        w_target_net = get_weights_target_net(w_ps, l, w_target_shape)
        # forward to predict (no softmax)
        y_pred_t = net.forward(x=x_t, w=w_target_net, p_dropout=p_dropout)
        # calculate NLL
        loss_NLL_temp = loss_fn(y_pred_t, y_t)
        # accumulate NLL
        loss_NLL = loss_NLL + loss_NLL_temp
    # average (take expectation) of NLL
    loss_NLL = loss_NLL/L
    
    loss_VFE = loss_NLL # KL divergence is zero at the beginning
    loss_VFE_grads = torch.autograd.grad(
        outputs=loss_VFE,
        inputs=theta.values(),
        create_graph=train_flag
    )
    loss_VFE_gradients = dict(zip(theta.keys(), loss_VFE_grads))
    for key in theta.keys():
        q[key] = theta[key] - inner_lr*loss_VFE_gradients[key]

    '''2nd update'''
    for i in range(num_inner_updates - 1):
        # train discriminator
        if i == 0:
            create_graph_flag = True
            w_d = train_discriminator(
                wg_p=theta,
                wg_q=q,
                w_d0=w_discriminator,
                p_z=p_z,
                p_dropout_d=p_dropout_d,
                p_dropout_g=p_dropout_g,
                create_graph_flag=create_graph_flag
            )
        else:
            create_graph_flag = False
            w_d = train_discriminator(
                wg_p=theta,
                wg_q=q,
                w_d0=w_d,
                p_z=p_z,
                p_dropout_d=p_dropout_d,
                p_dropout_g=p_dropout_g,
                create_graph_flag=create_graph_flag
            )

        z = p_z.rsample(sample_shape=(L,))
        w_qs = wGenerator.forward(z, q, p_dropout=p_dropout_g)

        loss_NLL = 0
        for l in range(L):
            w_target_net = get_weights_target_net(w_qs, l, w_target_shape)
            y_pred_t = net.forward(x=x_t, w=w_target_net, p_dropout=p_dropout)
            loss_NLL_temp = loss_fn(y_pred_t, y_t)
            loss_NLL = loss_NLL + loss_NLL_temp
        loss_NLL = loss_NLL/L
        
        # z = p_z.rsample(sample_shape=(L,))
        # w_qs = wGenerator.forward(z, q, p_dropout=p_dropout_g)

        # z = p_z.rsample(sample_shape=(L,))
        # w_ps = wGenerator.forward(z, theta, p_dropout=p_dropout_g)

        KL_loss = -wDiscriminator.forward(
            w_input=w_qs,
            w_discriminator=w_d,
            p_dropout=p_dropout_d
        )
        KL_loss = torch.mean(KL_loss)
        # if KL_loss.item() < 0:
        #     # KL_loss = torch.tensor([0.], device=device)
        
        loss_VFE = loss_NLL + kl_reweight*KL_loss
        
        loss_VFE_grads = torch.autograd.grad(
            outputs=loss_VFE,
            inputs=q.values(),
            retain_graph=train_flag
        )
        loss_VFE_gradients = dict(zip(q.keys(), loss_VFE_grads))

        for key in q.keys():
            q[key] = q[key] - inner_lr*loss_VFE_gradients[key]
        
    #     print('{0:1.3f}, {1:1.4f}'.format(loss_NLL.item(), KL_loss.item()))
    #     print()
    # sys.exit()
    
    '''Prediction'''
    if y_v is None:
        y_pred_v = [] #torch.zeros((x_v.shape[0], num_classes_per_task), device=device)
    else:
        loss_NLL = 0
    
    z = p_z.rsample(sample_shape=(K,))
    w_qs = wGenerator.forward(z=z, w_generator=q, p_dropout=0)

    for k in range(K):
        w_target_net = get_weights_target_net(w_qs, k, w_target_shape)
        y_pred_ = net.forward(x=x_v, w=w_target_net)
        # y_pred_t = net.forward(x=x_t, w=w_target_net)
        if y_v is not None:
            loss_NLL = loss_NLL + loss_fn(y_pred_, y_v) #+ loss_fn(y_pred_t, y_t)
        else:
            y_pred_v.append(y_pred_)
    
    if y_v is None:
        return torch.stack(y_pred_v)
    else:
        loss_NLL = loss_NLL/K

    # KL loss for monitoring
    z = p_z.rsample(sample_shape=(K,))
    w_qs = wGenerator.forward(z=z, w_generator=q, p_dropout=0)
    KL_loss = -wDiscriminator.forward(w_input=w_qs, w_discriminator=w_d)
    KL_loss = torch.mean(KL_loss)

    z = p_z.rsample(sample_shape=(K,))
    w_ps = wGenerator.forward(z=z, w_generator=theta, p_dropout=0)

    z = p_z.rsample(sample_shape=(K,))
    w_qs = wGenerator.forward(z=z, w_generator=q, p_dropout=0)
    # discriminator loss to train the meta discriminator
    d_loss = compute_discriminator_loss(w_p=w_ps, w_q=w_qs, w_d=w_d)

    return loss_NLL, KL_loss, d_loss

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
            maml_folder = '{0:s}/MAML_sine_line_{1:d}way_{2:d}shot'.format(dst_folder_root, num_classes_per_task, num_training_samples_per_class)
            maml_filename = 'sine_line_{0:d}way_{1:d}shot_{2:s}.pt'.format(num_classes_per_task, num_training_samples_per_class, '{0:d}')

            # i = 1
            # maml_checkpoint_filename = os.path.join(maml_folder, maml_filename.format(i))
            # while(os.path.exists(maml_checkpoint_filename)):
            #     i = i + 1
            #     maml_checkpoint_filename = os.path.join(maml_folder, maml_filename.format(i))
            # print(maml_checkpoint_filename)
            i = 3
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
            plt.xlabel('x')
            plt.ylabel('y')
            plt.tight_layout()
            plt.savefig(fname='img/mixed_sine_temp.svg', format='svg')
            return 0
        else:
            from scipy.special import erf

            quantiles = np.arange(start=0., stop=1.1, step=0.1)
            cal_avg = np.zeros(shape=quantiles.shape)

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
                # cal_temp = 1/(np.sqrt(2*np.pi)*std)*np.exp(-(y_preds_quantile - y0)**2/(2*std**2))
                cal_temp = (1 + erf((y_preds_quantile - y0)/(np.sqrt(2)*std)))/2
                cal_temp_avg = np.mean(a=cal_temp, axis=1)
                cal_avg = cal_avg + cal_temp_avg
            cal_avg = cal_avg / num_val_tasks
            return cal_avg
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
        while (task_count < num_val_tasks):
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

def train_discriminator(wg_p, wg_q, w_d0, create_graph_flag, p_z, p_dropout_d=0, p_dropout_g=0):
    w_d1 = {}

    z = p_z.rsample(sample_shape=(discriminator_minibatch_size,))
    w_p = wGenerator.forward(z=z, w_generator=wg_p, p_dropout=p_dropout_g)

    z = p_z.rsample(sample_shape=(discriminator_minibatch_size,))
    w_q = wGenerator.forward(z=z, w_generator=wg_q, p_dropout=p_dropout_g)
    
    d_loss = compute_discriminator_loss(w_p, w_q, w_d0, p_dropout=p_dropout_d)
    
    # if d_loss.item() > 10:
    #     d_loss.data = torch.tensor([10.], device=device)

    if create_graph_flag:
        d_loss_grads = torch.autograd.grad(
            outputs=d_loss,
            inputs=w_d0.values(),
            create_graph=train_flag
        )
    else:
        d_loss_grads = torch.autograd.grad(
            outputs=d_loss,
            inputs=w_d0.values(),
            retain_graph=train_flag
        )
    
    d_loss_gradients = dict(zip(w_d0.keys(), d_loss_grads))
    for key in w_d0.keys():
        w_d1[key] = w_d0[key] - discriminator_inner_lr*d_loss_gradients[key]

    return w_d1


def compute_discriminator_loss(w_p, w_q, w_d, p_dropout=0):
    d_p = wDiscriminator.forward(w_input=w_p, w_discriminator=w_d, p_dropout=p_dropout)
    # real_label = random.uniform(a=0.9, b=1.)
    label_real = torch.ones(d_p.shape, device=device)*real_label
    d_p_loss = BCE_with_logit_loss(d_p, label_real)

    d_q = wDiscriminator.forward(w_input=w_q, w_discriminator=w_d, p_dropout=p_dropout)
    # fake_label = random.uniform(a=0., b=0.3)
    # fake_label = 0.
    label_fake = torch.ones(d_q.shape, device=device)*fake_label
    d_q_loss = BCE_with_logit_loss(d_q, label_fake)

    return d_p_loss + d_q_loss

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
        y_pred_t = net.forward(x=x_t, w=q, p_dropout=p_dropout_base)
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

if __name__ == "__main__":
    main()
