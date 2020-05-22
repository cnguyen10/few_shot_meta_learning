'''
python3 abml.py --datasource=sine_line --n_way=1 --k_shot=5 --inner_lr=1e-3 --num_inner_updates=5 --meta_lr=1e-3 --Lt=32 --Lv=32 --kl_reweight=1 --minibatch_size=10 --resume_epoch=0
python3 vampire.py --datasource=sine_line --n_way=1 --k_shot=5 --inner_lr=1e-3 --num_inner_updates=5 --Lt=32 --Lv=32 --kl_reweight=1 --resume_epoch=50 --test --no_uncertainty --num_val_tasks=1

python3 abml.py --datasource=miniImageNet --n_way=5 --k_shot=1 --inner_lr=1e-2 --num_inner_updates=5 --meta_lr=1e-3 --minibatch_size=2  --Lt=10 --Lv=10 --kl_reweight=0.1 --num_epochs=50 --resume_epoch=0

python3 abml.py --datasource=miniImageNet --n_way=5 --k_shot=1 --inner_lr=1e-2 --num_inner_updates=5  --Lt=10 --Lv=10 --kl_reweight=0.1 --resume_epoch=50 --test --no_uncertainty --num_val_tasks=600

python3 abml.py --datasource=tieredImageNet_640 --n_way=5 --k_shot=1 --inner_lr=1e-2 --num_inner_updates=5 --meta_lr=1e-3 --minibatch_size=10  --Lt=32 --Lv=32 --kl_reweight=0.1 --num_epochs=50 --resume_epoch=0
'''

import torch

import numpy as np
import random
import itertools

import os
import sys

import csv

import argparse

from utils import load_dataset, initialize_dataloader, get_train_val_task_data

# -------------------------------------------------------------------------------------------------
# Setup input parser
# -------------------------------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='Setup variables for ABML.')

parser.add_argument('--datasource', type=str, default='sine_line', help='Datasource: sine_line, omniglot, miniImageNet, miniImageNet_640')
parser.add_argument('--n_way', type=int, default=5, help='Number of classes per task')
parser.add_argument('--k_shot', type=int, default=1, help='Number of training samples per class or k-shot')
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

parser.add_argument('--num_val_tasks', type=int, default=100, help='Number of validation tasks')

parser.add_argument('--kl_reweight', type=float, default=1, help='Reweight factor of the KL divergence between variational posterior q and prior p')
parser.add_argument('--Lt', type=int, default=1, help='Number of ensemble networks to train task specific parameters')
parser.add_argument('--Lv', type=int, default=1, help='Number of ensemble networks to validate meta-parameters')

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

num_meta_updates_print = int(1000 / num_tasks_per_minibatch)
print('Mini batch size = {0:d}'.format(num_tasks_per_minibatch))

num_epochs_save = 1

num_epochs = args.num_epochs

expected_total_tasks_per_epoch = 10000
num_tasks_per_epoch = int(expected_total_tasks_per_epoch / num_tasks_per_minibatch)*num_tasks_per_minibatch

expected_tasks_save_loss = 10000
num_tasks_save_loss = int(expected_tasks_save_loss / num_tasks_per_minibatch)*num_tasks_per_minibatch

num_val_tasks = args.num_val_tasks
uncertainty_flag = args.uncertainty_flag

# Number of emsembling models/networks
Lt = args.Lt
Lv = args.Lv
print('Lt = {0:d}, Lv = {1:d}'.format(Lt, Lv))

KL_reweight = args.kl_reweight
print('KL reweight = {0}'.format(KL_reweight))

# -------------------------------------------------------------------------------------------------
# Setup destination folder
# -------------------------------------------------------------------------------------------------
dst_folder_root = './ABML'
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
    theta = {}
    theta['mean'] = {}
    theta['logSigma'] = {}
    for key in weight_shape.keys():
        if 'b' in key:
            theta['mean'][key] = torch.zeros(weight_shape[key], device=device, requires_grad=True)
        else:
            theta['mean'][key] = torch.empty(weight_shape[key], device=device)
            torch.nn.init.xavier_normal_(tensor=theta['mean'][key], gain=np.sqrt(2))
            theta['mean'][key].requires_grad_()
        theta['logSigma'][key] = torch.rand(weight_shape[key], device=device) - 4
        theta['logSigma'][key].requires_grad_()
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
    [
        {
            'params': theta['mean'].values(),
            'weight_decay': 0
        },
        {
            'params': theta['logSigma'].values(),
            'weight_decay': 0
        }
    ],
    lr=meta_lr
)

if resume_epoch > 0:
    op_theta.load_state_dict(saved_checkpoint['op_theta'])
    # op_theta.param_groups[0]['lr'] = meta_lr
    # op_theta.param_groups[1]['lr'] = meta_lr
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

def meta_train(train_subset=train_set):
    if datasource == 'sine_line':
        data_generator = DataGenerator(num_samples=num_samples_per_class)
        # create dummy sampler
        all_class_train = [0] * 100
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
                y_pred = predict(x=torch.cat((x_t, x_v)), q=q, num_models=Lv)
                loss_NLL = 0
                for lv in range(Lv):
                    loss_NLL_temp = loss_fn(input=y_pred[lv], target=torch.cat((y_t, y_v)))
                    loss_NLL = loss_NLL + loss_NLL_temp
                loss_NLL = loss_NLL / Lv

                if torch.isnan(loss_NLL).item():
                    sys.exit('NaN error')

                # accumulate meta loss
                meta_loss = meta_loss + loss_NLL

                task_count = task_count + 1
                if task_count % num_tasks_per_minibatch == 0:
                    meta_loss = meta_loss / num_tasks_per_minibatch

                    # accumulate into different variables for printing purpose
                    meta_loss_avg_print = meta_loss_avg_print + meta_loss.item()

                    op_theta.zero_grad()
                    meta_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        parameters=theta['mean'].values(),
                        max_norm=10
                    )
                    torch.nn.utils.clip_grad_norm_(
                        parameters=theta['logSigma'].values(),
                        max_norm=10
                    )
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

                        # if datasource != 'sine_line':
                        #     val_accs = validate_classification(
                        #         all_classes=all_class_val,
                        #         all_data=all_data_val,
                        #         num_val_tasks=100,
                        #         rand_flag=True,
                        #         uncertainty=False,
                        #         csv_flag=False
                        #     )
                        #     val_acc = np.mean(val_accs)
                        #     val_ci95 = 1.96*np.std(val_accs)/np.sqrt(num_val_tasks)
                        #     print('Validation accuracy = {0:2.4f} +/- {1:2.4f}'.format(val_acc, val_ci95))
                        #     val_accuracies.append(val_acc)

                        #     train_accs = validate_classification(
                        #         all_classes=all_class_train,
                        #         all_data=all_data_train,
                        #         num_val_tasks=100,
                        #         rand_flag=True,
                        #         uncertainty=False,
                        #         csv_flag=False
                        #     )
                        #     train_acc = np.mean(train_accs)
                        #     train_ci95 = 1.96*np.std(train_accs)/np.sqrt(num_val_tasks)
                        #     print('Train accuracy = {0:2.4f} +/- {1:2.4f}\n'.format(train_acc, train_ci95))
                        #     train_accuracies.append(train_acc)
                    
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
        print()
        scheduler.step()

def adapt_to_task(x, y, theta0):
    q = initialize_q(key_list=weight_shape.keys())

    # 1st gradient update - KL divergence is zero
    loss_NLL = 0
    for _ in range(Lt):
        w = sample_nn_weight(meta_params=theta0)
        y_pred = net.forward(x=x, w=w)
        loss_NLL_temp = loss_fn(input=y_pred, target=y)
        loss_NLL = loss_NLL + loss_NLL_temp
    loss_NLL = loss_NLL / Lt

    loss_NLL_grads_mean = torch.autograd.grad(
        outputs=loss_NLL,
        inputs=theta0['mean'].values(),
        create_graph=True
    )
    loss_NLL_grads_logSigma = torch.autograd.grad(
        outputs=loss_NLL,
        inputs=theta0['logSigma'].values(),
        create_graph=True
    )
    gradients_mean = dict(zip(theta0['mean'].keys(), loss_NLL_grads_mean))
    gradients_logSigma = dict(zip(theta0['logSigma'].keys(), loss_NLL_grads_logSigma))
    for key in weight_shape.keys():
        q['mean'][key] = theta0['mean'][key] - inner_lr * gradients_mean[key]
        q['logSigma'][key] = theta0['logSigma'][key] - inner_lr * gradients_logSigma[key]
    
    # 2nd updates
    for _ in range(num_inner_updates - 1):
        loss_NLL = 0
        for _ in range(Lt):
            w = sample_nn_weight(meta_params=q)
            y_pred = net.forward(x=x, w=w)
            loss_NLL_temp = loss_fn(input=y_pred, target=y)
            loss_NLL = loss_NLL + loss_NLL_temp
        loss_NLL = loss_NLL / Lt

        loss_NLL_grads_mean = torch.autograd.grad(
            outputs=loss_NLL,
            inputs=q['mean'].values(),
            retain_graph=True
        )
        loss_NLL_grads_logSigma = torch.autograd.grad(
            outputs=loss_NLL,
            inputs=q['logSigma'].values(),
            retain_graph=True
        )
        loss_NLL_gradients_mean = dict(zip(q['mean'].keys(), loss_NLL_grads_mean))
        loss_NLL_gradients_logSigma = dict(zip(q['logSigma'].keys(), loss_NLL_grads_logSigma))

        for key in w.keys():
            loss_KL_grad_mean = -torch.exp(-2 * theta['logSigma'][key]) * (theta['mean'][key] - q['mean'][key])
            loss_KL_grad_logSigma = torch.exp(2 * (q['logSigma'][key] - theta['logSigma'][key])) - 1
            
            q['mean'][key] = q['mean'][key] - inner_lr * (loss_NLL_gradients_mean[key] + KL_reweight * loss_KL_grad_mean)
            q['logSigma'][key] = q['logSigma'][key] - inner_lr * (loss_NLL_gradients_logSigma[key] \
                + KL_reweight * loss_KL_grad_logSigma)
    return q

def predict(x, q, num_models=1):
    y_pred = torch.empty(
        size=(num_models, x.shape[0], num_classes_per_task),
        device=device
    )
    for lv in range(num_models):
        w = sample_nn_weight(meta_params=q)
        y_pred[lv, :, :] = net.forward(x=x, w=w)
    return y_pred

def validate_regression(uncertainty_flag, num_val_tasks=1):
    assert datasource == 'sine_line'

    if uncertainty_flag:
        from scipy.special import erf
        quantiles = np.arange(start=0., stop=1.1, step=0.1)
        filename = 'ABML_calibration_{0:s}_{1:d}shot_{2:d}.csv'.format(
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
        y_pred = predict(x=x0, q=q, num_models=Lv)
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
        filename = 'ABML_{0:s}_{1:d}way_{2:d}shot_{3:s}_{4:d}.csv'.format(
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
        raw_scores = predict(x=x_v, q=q, num_models=Lv) # Lv x num_samples x num_classes
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

def sample_nn_weight(meta_params):
    w = {}
    for key in meta_params['mean'].keys():
        eps_sampled = torch.randn_like(input=meta_params['mean'][key], device=device)
        w[key] = meta_params['mean'][key] + eps_sampled * torch.exp(meta_params['logSigma'][key])

    return w

def initialize_q(key_list):
    q = dict.fromkeys(['mean', 'logSigma'])
    for para in q.keys():
        q[para] = {}
        for key in key_list:
            q[para][key] = 0
    return q

if __name__ == "__main__":
    main()
