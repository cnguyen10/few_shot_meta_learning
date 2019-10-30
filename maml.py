'''
python3 maml.py --datasource=sine_line --num_inner_updates=5 --inner_lr=1e-3 --meta_lr=1e-3 --minibatch_size=10 --resume_epoch=30 --test --num_val_tasks=1000

python3 maml.py --datasource=miniImageNet --n_way=5 --k_shot=1 --num_inner_updates=5 --inner_lr=1e-2 --meta_lr=1e-3 --minibatch_size=10 --resume_epoch=0
'''
import torch

import numpy as np
import random
import itertools

import os
import sys


import argparse

parser = argparse.ArgumentParser(description='Setup variables for MAML.')

parser.add_argument('--datasource', type=str, default='sine_line', help='Datasource: sine_line, miniImageNet, miniImageNet_embedding')
parser.add_argument('--n_way', type=int, default=1, help='Number of classes per task')
parser.add_argument('--k_shot', type=int, default=5, help='Number of training samples per class or k-shot')
parser.add_argument('--resume_epoch', type=int, default=0, help='Epoch id to resume learning or perform testing')

parser.add_argument('--inner_lr', type=float, default=1e-3, help='Learning rate for task-specific parameters')
parser.add_argument('--num_inner_updates', type=int, default=5, help='Number of gradient updates for task-specific parameters')
parser.add_argument('--meta_lr', type=float, default=1e-3, help='Learning rate of meta-parameters')
parser.add_argument('--minibatch_size', type=int, default=25, help='Number of tasks per minibatch to update meta-parameters')
parser.add_argument('--num_epochs', type=int, default=1000, help='How many 10,000 tasks are used to train?')

parser.add_argument('--lr_decay', type=float, default=1, help='Decay factor of meta-learning rate')
parser.add_argument('--meta_l2_regularization', type=float, default=0, help='L2 regularization coeff. of meta-parameters')

parser.add_argument('--train', dest='train_flag', action='store_true')
parser.add_argument('--test', dest='train_flag', action='store_false')

parser.add_argument('--uncertainty', dest='uncertainty_flag', action='store_true')
parser.add_argument('--no_uncertainty', dest='uncertainty_flag', action='store_false')
parser.set_defaults(uncertainty_flag=True)

parser.add_argument('--num_val_tasks', type=int, default=100, help='Number of validation tasks')

parser.set_defaults(train_flag=True)

args = parser.parse_args()

gpu_id = 0
device = torch.device('cuda:{0:d}'.format(gpu_id) if torch.cuda.is_available() else "cpu")
# device = torch.device('cpu')

datasource = args.datasource

train_flag = args.train_flag

num_training_samples_per_class = args.k_shot

if train_flag:
    num_total_samples_per_class = num_training_samples_per_class + 15 # total number of samples per class
else:
    num_total_samples_per_class = num_training_samples_per_class + 20

num_tasks_per_minibatch = args.minibatch_size

num_classes_per_task = args.n_way

num_tasks_per_minibatch = args.minibatch_size
num_meta_updates_print = int(500/num_tasks_per_minibatch)
print('Mini batch size = {0:d}'.format(num_tasks_per_minibatch))

num_epochs_save = 1

inner_lr = args.inner_lr
print('Inner learning rate = {0}'.format(inner_lr))

meta_lr = args.meta_lr
print('Meta learning rate = {0}'.format(meta_lr))

expected_total_tasks_per_epoch = 100000
num_tasks_per_epoch = int(expected_total_tasks_per_epoch/num_tasks_per_minibatch)*num_tasks_per_minibatch

expected_tasks_save_loss = 100000
num_tasks_save_loss = int(expected_tasks_save_loss/num_tasks_per_minibatch)*num_tasks_per_minibatch

num_epochs = args.num_epochs

num_inner_updates = args.num_inner_updates
print('Number of inner updates = {0:d}'.format(num_inner_updates))

dst_root_folder = '/media/n10/Data'
dst_folder = '{0:s}/MAML_{1:s}_{2:d}way_{3:d}shot'.format(
    dst_root_folder,
    datasource,
    num_classes_per_task,
    num_training_samples_per_class
)
if not os.path.exists(dst_folder):
    os.makedirs(dst_folder)

noise_flag = True

num_val_tasks = args.num_val_tasks

p_dropout = 0.

if datasource == 'sine_line':
    from DataGeneratorT import DataGenerator
    from FCNet import FCNet
    from utils import get_task_sine_line_data

    p_sine = 0.5
    net = FCNet(
        dim_input=1,
        dim_output=1,
        # num_hidden_units=(100, 100, 100),
        num_hidden_units=(40, 40),
        device=device
    )
    # loss function as mean-squared error
    loss_fn = torch.nn.MSELoss()
elif datasource == 'miniImageNet':
    from ConvNet import ConvNet
    from utils import load_dataset, get_task_image_data

    train_set = 'train'
    val_set = 'val'
    test_set = 'test'

    loss_fn = torch.nn.CrossEntropyLoss()
    sm_loss = torch.nn.Softmax(dim=1)

    net = ConvNet(
        dim_input=(3, 84, 84),
        dim_output=num_classes_per_task,
        num_filters=(32, 32, 32, 32),
        filter_size=(3, 3),
        device=device
    )

    if train_flag:
        all_class_train, embedding_train = load_dataset(datasource, train_set)
        all_class_val, embedding_val = load_dataset(datasource, val_set)
    else:
        all_class_test, embedding_test = load_dataset(datasource, test_set)
# elif datasource == 'miniImageNet_embedding':
#     pass
else:
    sys.exit('Unknown dataset!')

weight_shape = net.get_weight_shape()

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

else:
    checkpoint_filename = ('{0:s}_{1:d}way_{2:d}shot_{3:d}.pt').format(
        datasource,
        num_classes_per_task,
        num_training_samples_per_class,
        resume_epoch
    )
    checkpoint_file = os.path.join(dst_folder, checkpoint_filename)
    print('Start to load weights from')
    print('{0:s}'.format(checkpoint_file))
    if torch.cuda.is_available():
        saved_checkpoint = torch.load(checkpoint_file, map_location=lambda storage, loc: storage.cuda(gpu_id))
    else:
        saved_checkpoint = torch.load(checkpoint_file, map_location=lambda storage, loc: storage)
    # saved_checkpoint = torch.load(checkpoint_file, map_location=lambda storage, loc: storage)

    theta = saved_checkpoint['theta']

op_theta = torch.optim.Adam(params=theta.values(), lr=meta_lr)
if resume_epoch > 0:
    op_theta.load_state_dict(saved_checkpoint['op_theta'])
    op_theta.param_groups[0]['lr'] = meta_lr
print(op_theta)

# decay the learning rate
scheduler = torch.optim.lr_scheduler.ExponentialLR(
    optimizer=op_theta,
    gamma=args.lr_decay)

train_set = 'train'
val_set = 'val'
test_set = 'test'

uncertainty_flag = args.uncertainty_flag

print()

def main():
    if train_flag:
        meta_train()
    elif resume_epoch > 0:
        if datasource == 'sine_line':
            cal_data = meta_validation(datasubset=args.datasubset, num_val_tasks=num_val_tasks)

            if num_val_tasks > 0:
                cal_data = np.array(cal_data)
                np.savetxt(fname='maml_{0:s}_calibration.csv'.format(datasource), X=cal_data, delimiter=',')
        else:
            if not uncertainty_flag:
                accs, all_task_names = meta_validation(
                    datasubset=test_set,
                    num_val_tasks=num_val_tasks,
                    return_uncertainty=uncertainty_flag
                )
                with open(file='maml_{0:s}_{1:d}_{2:d}_accuracies.csv'.format(datasource, num_classes_per_task, num_training_samples_per_class), mode='w') as result_file:
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
                with open(file='maml_{0:s}_correct_prob.csv'.format(datasource), mode='w') as result_file:
                    for correct, prob in zip(corrects, probs):
                        result_file.write('{0}, {1}\n'.format(correct, prob))
                        # print(correct, prob)
    else:
        sys.exit('Unknown action')

def meta_train():
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
                        if datasource != 'sine_line':
                            val_accs, _ = meta_validation(
                                datasubset=val_set,
                                num_val_tasks=num_val_tasks,
                                return_uncertainty=False)
                            val_acc = np.mean(val_accs)
                            val_ci95 = 1.96*np.std(val_accs)/np.sqrt(num_val_tasks)
                            print('Validation accuracy = {0:2.4f} +/- {1:2.4f}'.format(val_acc, val_ci95))
                            val_accuracies.append(val_acc)

                            train_accs, _ = meta_validation(
                                datasubset=train_set,
                                num_val_tasks=num_val_tasks,
                                return_uncertainty=False)
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
            checkpoint_filename = ('{0:s}_{1:d}way_{2:d}shot_{3:d}.pt')\
                        .format(datasource,
                                num_classes_per_task,
                                num_training_samples_per_class,
                                epoch + 1)
            print(checkpoint_filename)
            torch.save(checkpoint, os.path.join(dst_folder, checkpoint_filename))
        print()

def get_task_prediction(x_t, y_t, x_v, y_v=None):
    q = {}

    y_pred_t = net.forward(x=x_t, w=theta, p_dropout=p_dropout)
    loss_NLL = loss_fn(y_pred_t, y_t)

    grads = torch.autograd.grad(
        outputs=loss_NLL,
        inputs=theta.values(),
        create_graph=True
    )
    gradients = dict(zip(theta.keys(), grads))

    for key in theta.keys():
        q[key] = theta[key] - inner_lr*gradients[key]

    '''2nd update'''
    for _ in range(num_inner_updates - 1):
        loss_NLL = 0
        y_pred_t = net.forward(x=x_t, w=q, p_dropout=p_dropout)
        loss_NLL = loss_fn(y_pred_t, y_t)
        grads = torch.autograd.grad(
            outputs=loss_NLL,
            inputs=q.values(),
            retain_graph=True
        )
        gradients = dict(zip(q.keys(), grads))

        for key in q.keys():
            q[key] = q[key] - inner_lr*gradients[key]
    
    '''Task prediction'''
    y_pred_v = net.forward(x=x_v, w=q, p_dropout=0)
    
    if y_v is None:
        return y_pred_v
    else:
        loss_NLL = loss_fn(y_pred_v, y_v)
        return loss_NLL
        

def meta_validation(datasubset, num_val_tasks, return_uncertainty=False):
    if datasource == 'sine_line':
        from scipy.special import erf
        
        x0 = torch.linspace(start=-5, end=5, steps=100, device=device).view(-1, 1)

        cal_avg = 0

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
            y0 = y0.view(1, -1).cpu().numpy()
            
            y_preds = get_task_prediction(x_t=x_t, y_t=y_t, x_v=x0)

            y_preds_np = torch.squeeze(y_preds, dim=-1).detach().cpu().numpy()

            # ground truth cdf
            std = data_generator.noise_std
            cal_temp = (1 + erf((y_preds_np - y0)/(np.sqrt(2)*std)))/2
            cal_temp_avg = np.mean(a=cal_temp, axis=1)
            cal_avg = cal_avg + cal_temp_avg
        cal_avg = cal_avg / num_val_tasks
        return cal_avg
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
            
            y_pred_v = get_task_prediction(x_t, y_t, x_v, y_v=None)
            y_pred = sm_loss(y_pred_v)

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

if __name__ == "__main__":
    main()