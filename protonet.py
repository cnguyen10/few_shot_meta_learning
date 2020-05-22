'''
python3 protonet.py --datasource=omniglot --n_way=5 --k_shot=1 --minibatch=20 --train --num_epochs=100 --lr_decay=0.99

python3 protonet.py --datasource=omniglot --n_way=5 --k_shot=1 --num_val_shots=19 --resume_epoch=100 --test --no_uncertainty --num_val_tasks=100000

python3 protonet.py --datasource=miniImageNet --n_way=5 --k_shot=1 --minibatch=10 --train --num_epochs=100 --lr_decay=0.99
python3 protonet.py --datasource=miniImageNet --n_way=5 --k_shot=1 --resume_epoch=100 --test --no_uncertainty --num_val_tasks=600

python3 protonet.py --datasource=miniImageNet_640 --n_way=5 --k_shot=1 --minibatch=100 --train --num_epochs=100 --lr_decay=0.99
'''
import torch

import numpy as np
import random
import itertools

import os
import sys

import argparse

from utils import load_dataset, get_train_val_task_data, initialize_dataloader

# -------------------------------------------------------------------------------------------------
# Setup input parser
# -------------------------------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='Setup variables for the Prototypical Network.')

parser.add_argument('--datasource', type=str, default='sine_line', help='Datasource: sine_line, omniglot, miniImageNet, miniImageNet_640')
parser.add_argument('--n_way', type=int, default=5, help='Number of classes per task')
parser.add_argument('--k_shot', type=int, default=1, help='Number of training samples per class or k-shot')
parser.add_argument('--num_val_shots', type=int, default=15, help='Number of validation samples per class')

parser.add_argument('--meta_lr', type=float, default=1e-3, help='Learning rate of meta-parameters')

parser.add_argument('--minibatch_size', type=int, default=25, help='Number of tasks per minibatch to update meta-parameters')
parser.add_argument('--num_epochs', type=int, default=100, help='How many 10,000 tasks are used to continue to train?')
parser.add_argument('--resume_epoch', type=int, default=0, help='Epoch id to resume learning or perform testing')

parser.add_argument('--lr_decay', type=float, default=1, help='Decay factor of meta-learning rate')

parser.add_argument('--train', dest='train_flag', action='store_true')
parser.add_argument('--test', dest='train_flag', action='store_false')
parser.set_defaults(train_flag=True)

parser.add_argument('--uncertainty', dest='uncertainty_flag', action='store_true')
parser.add_argument('--no_uncertainty', dest='uncertainty_flag', action='store_false')
parser.set_defaults(uncertainty_flag=True)

parser.add_argument('--num_val_tasks', type=int, default=100, help='Number of validation tasks')
parser.add_argument('--val_subset', type=str, default='test', help='Which subset will be tested')

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

train_set = 'train'
val_set = 'val'
test_set = 'test'
val_subset = args.val_subset
# -------------------------------------------------------------------------------------------------
#   Setup based model/network
# -------------------------------------------------------------------------------------------------
loss_fn = torch.nn.CrossEntropyLoss()
sm = torch.nn.Softmax(dim=1)

if datasource in ['omniglot', 'miniImageNet']:
    from ConvNet import ConvNet
    DIM_INPUT = {
        'omniglot': (1, 28, 28),
        'miniImageNet': (3, 84, 84)
    }

    net = ConvNet(
        dim_input=DIM_INPUT[datasource],
        dim_output=None,
        num_filters=(32, 32, 32, 32),
        filter_size=(3, 3),
        device=device
    )

elif datasource in ['miniImageNet_640', 'tieredImageNet_640']:
    import pickle
    from FC640 import FC640
    net = FC640(
        num_hidden_units=(128, 32),
        dim_output=None,
        device=device
    )
else:
    sys.exit('Unknown dataset!')
weight_shape = net.get_weight_shape()
# -------------------------------------------------------------------------------------------------
# Parse training parameters
# -------------------------------------------------------------------------------------------------
meta_lr = args.meta_lr
print('Meta learning rate = {0}'.format(meta_lr))

num_tasks_per_minibatch = args.minibatch_size
print('Number of tasks per minibatch = {0:d}'.format(num_tasks_per_minibatch))

num_meta_updates_print = int(1000/num_tasks_per_minibatch)

num_epochs_save = 1

expected_total_tasks_per_epoch = 10000
num_tasks_per_epoch = int(expected_total_tasks_per_epoch/num_tasks_per_minibatch)*num_tasks_per_minibatch

expected_tasks_save_loss = 10000
num_tasks_save_loss = int(expected_tasks_save_loss/num_tasks_per_minibatch)*num_tasks_per_minibatch

num_epochs = args.num_epochs

num_val_tasks = args.num_val_tasks

uncertainty_flag = args.uncertainty_flag

# -------------------------------------------------------------------------------------------------
# Setup destination folder
# -------------------------------------------------------------------------------------------------
dst_root_folder = './ProtoNet'
dst_folder = '{0:s}/{1:s}_{2:d}way_{3:d}shot'.format(
    dst_root_folder,
    datasource,
    num_classes_per_task,
    num_training_samples_per_class
)
if not os.path.exists(dst_folder):
    os.makedirs(dst_folder)

# -------------------------------------------------------------------------------------------------
# Intialize/Load meta-parameters
# -------------------------------------------------------------------------------------------------
resume_epoch = args.resume_epoch
if resume_epoch == 0:
    loss_meta_saved = [] # to monitor the meta-loss
    loss_kl_saved = []

    # initialise meta-parameters
    theta = {}
    for key in weight_shape.keys():
        if 'b' not in key:
            theta[key] = torch.empty(
                weight_shape[key],
                device=device
            )
            torch.nn.init.xavier_normal_(theta[key])
            theta[key].requires_grad_()
        else:
            theta[key] = torch.randn(weight_shape[key], device=device, requires_grad=True)

else: # load meta-parameter
    checkpoint_filename = ('Epoch_{0:d}.pt').format(resume_epoch)
    checkpoint_file = os.path.join(dst_folder, checkpoint_filename)
    print('Start to load weights from')
    print('{0:s}'.format(checkpoint_file))
    if torch.cuda.is_available():
        saved_checkpoint = torch.load(checkpoint_file, map_location=lambda storage, loc: storage.cuda(gpu_id))
    else:
        saved_checkpoint = torch.load(checkpoint_file, map_location=lambda storage, loc: storage)

    theta = saved_checkpoint['theta']

# intialize an optimizer for the meta-parameter
op_theta = torch.optim.Adam(params=theta.values(), lr=meta_lr)

# load the optimizer
if resume_epoch > 0:
    op_theta.load_state_dict(saved_checkpoint['op_theta'])
    # op_theta.param_groups[0]['lr'] = meta_lr
    del saved_checkpoint
print(op_theta)

# decay the learning rate
scheduler = torch.optim.lr_scheduler.ExponentialLR(
    optimizer=op_theta,
    gamma=args.lr_decay)

p_rand = None
if train_flag == 'test':
    uncertainty_flag = args.uncertainty_flag
    print('Uncertainty flag = {0}'.format(uncertainty_flag))

    if datasource in ['omniglot', 'tieredImageNet']:
        p_rand = 1e-7

print()

# -------------------------------------------------------------------------------------------------
# MAIN program
# -------------------------------------------------------------------------------------------------
def main():
    if train_flag:
        meta_train()
    else:
        all_class_test, all_data_test = load_dataset(
            dataset_name=datasource,
            subset=test_set
        )
        
        meta_validation(
            all_classes=all_class_test,
            all_data=all_data_test,
            num_val_tasks=num_val_tasks,
            p_rand=p_rand,
            uncertainty=uncertainty_flag,
            csv_flag=True
        )

# -------------------------------------------------------------------------------------------------
# TRAIN
# -------------------------------------------------------------------------------------------------
def meta_train():
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

        task_count = 0 # a counter to decide when a minibatch of task is completed to perform meta update
        meta_loss = 0 # accumulate the loss of many ensambling networks to descent gradient for meta update
        num_meta_updates_count = 0

        meta_loss_avg_print = 0 # compute loss average to print

        meta_loss_avg_save = [] # meta loss to save

        while (task_count < num_tasks_per_epoch):
            for class_labels in train_loader:
                x_t, y_t, x_v, y_v = get_train_val_task_data(
                    all_classes=all_class_train,
                    all_data=all_data_train,
                    class_labels=class_labels,
                    num_samples_per_class=num_samples_per_class,
                    num_training_samples_per_class=num_training_samples_per_class,
                    device=device
                )
                
                prototypes = get_class_prototypes(x=x_t, y=y_t, w=theta, device=device)

                z_v = net.forward(x=x_v, w=theta)
                distance_matrix = euclidean_distance(matrixN=z_v, matrixM=prototypes)
                
                loss_NLL = loss_fn(input=-distance_matrix, target=y_v)

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
                    torch.nn.utils.clip_grad_norm_(parameters=theta.values(), max_norm=10)
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

                        print('Saving loss...')
                        val_accs = meta_validation(
                            all_classes=all_class_val,
                            all_data=all_data_val,
                            num_val_tasks=100,
                            p_rand=0.5,
                            uncertainty=False,
                            csv_flag=False
                        )
                        val_acc = np.mean(val_accs)
                        val_ci95 = 1.96 * np.std(val_accs) / np.sqrt(num_val_tasks)
                        print('Validation accuracy = {0:2.4f} +/- {1:2.4f}'.format(val_acc, val_ci95))
                        val_accuracies.append(val_acc)

                        train_accs = meta_validation(
                            all_classes=all_class_train,
                            all_data=all_data_train,
                            num_val_tasks=100,
                            p_rand=0.5,
                            uncertainty=False,
                            csv_flag=False
                        )
                        train_acc = np.mean(train_accs)
                        train_ci95 = 1.96 * np.std(train_accs) / np.sqrt(num_val_tasks)
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
            checkpoint_filename = 'Epoch_{0:d}'.format(epoch + 1)
            print(checkpoint_filename)
            torch.save(checkpoint, os.path.join(dst_folder, checkpoint_filename))
        scheduler.step()
        print()

def get_class_prototypes(x, y, w, device=torch.device('cpu')):
    # calculate the embeddings of the data
    z = net.forward(x=x, w=w)
    
    # initialize prototypes
    prototypes = torch.zeros((num_classes_per_task, z.shape[1]), device=device)

    for i in range(num_classes_per_task):
        z_one_class = z[y == i, :]
        prototypes[i, :] = torch.mean(input=z_one_class, dim=0)
    
    return prototypes

def euclidean_distance(matrixN, matrixM):
    ''' Calculate distance from N points to M points
    - matrixN = N x D matrix
    - matrixM = M x D matrix
    Return: N x M matrix
    '''
    N = matrixN.size(0)
    M = matrixM.size(0)
    D = matrixN.size(1)
    assert D == matrixM.size(1)

    matrixN = matrixN.unsqueeze(1).expand(N, M, D)
    matrixM = matrixM.unsqueeze(0).expand(N, M, D)

    return torch.norm(input=matrixN - matrixM, p='fro', dim=2)

def predict(x, prototypes):
    z = net.forward(x=x, w=theta)
    distance_matrix = euclidean_distance(matrixN=z, matrixM=prototypes)
    likelihood = sm(input=-distance_matrix)

    prob, y_pred = torch.max(input=likelihood, dim=1)
    return y_pred, prob.detach().cpu().numpy()

def meta_validation(all_classes, all_data, num_val_tasks, p_rand=None, uncertainty=False, csv_flag=False):
    if csv_flag:
        import csv
        filename = 'ProtoNet_{0:s}_{1:d}way_{2:d}shot_{3:s}_{4:d}.csv'.format(
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
        if p_rand is not None:
            skip_task = np.random.binomial(n=1, p=p_rand)
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
        
        prototypes = get_class_prototypes(x=x_t, y=y_t, w=theta, device=device)
        y_pred, prob = predict(x=x_v, prototypes=prototypes)
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

if __name__ == "__main__":
    main()
