import os
import sys
from functools import partial

import ray
import torch
import torch.nn.functional as F
import torch.optim as optim
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from torch.utils.data import DataLoader

import core
import data
import params


def get_spec(config):
    model = config['model']

    if model == 'Baseline':
        config['hidden_units'] = tune.grid_search([128])

    return config


def get_model(config, in_features, out_features):
    model = config['model']

    if model == 'Baseline':
        return core.Baseline(in_features, out_features, config['hidden_units'])


def objective(output, target, weights, delta=1.0):
    """
    :param output: (num_samples, num_classes)
    :param target: (num_samples)
    :param weights: (num_samples)
    :param delta: default 1.0
    :return:
    """
    device = output.device
    num_samples, num_classes = output.size()

    delta = torch.tensor(data=delta, dtype=torch.float32, device=device)

    target_mask = F.one_hot(target, num_classes=num_classes).to(device=device)  # (num_samples, num_classes)
    inverse_target_mask = torch.ones_like(target_mask, device=device).sub(target_mask)

    target = output.mul(target_mask)
    target = target.matmul(torch.ones(num_classes, num_classes, device=device))  # (num_samples, num_classes)

    loss = (output - target + delta).mul(inverse_target_mask)  # (num_samples, num_classes)
    loss = loss.clamp_min(0.0)

    ranks = (loss > 0.0).sum(dim=1).clamp_min(1).sub(1)  # (num_samples)

    betas = 1.0 / torch.arange(1, num_classes, dtype=torch.float32)  # (num_classes - 1)
    betas = betas.cumsum(dim=0) / torch.arange(1, num_classes, dtype=torch.float32)
    betas = betas.to(device=device)
    betas = betas[ranks]  # (num_samples)

    obj = loss.sum(dim=1, dtype=torch.float32).mul(betas).mul(weights)  # (num_samples)
    obj = obj.mean()

    return obj


def train(model, criterion, optimizer, train_loader, train_embs, config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device=device)

    train_embs = train_embs.to(device)

    model.train()

    for batch_idx, data in enumerate(train_loader, 0):
        # Get the inputs; data is a list of [features, labels, weights]
        features, labels, weights = data
        features, labels, weights = features.to(device), labels.to(device), weights.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward + backward + optimize
        outputs = model(features, train_embs)
        loss = criterion(outputs, labels, weights, delta=config['delta'])
        loss.backward()
        optimizer.step()


def eval(model, criterion, data_loader, data_embs, config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device=device)

    data_embs = data_embs.to(device)

    model.eval()

    eval_loss = 0.0
    eval_steps = 0
    total = correct = 0
    with torch.no_grad():
        for batch_idx, data in enumerate(data_loader, 0):
            features, labels, weights = data
            features, labels, weights = features.to(device), labels.to(device), weights.to(device)

            outputs = model(features, data_embs)
            _, predicted = torch.max(outputs.data, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            loss = criterion(outputs, labels, weights, delta=config['delta'])
            eval_loss += loss.cpu().numpy()
            eval_steps += 1

    return eval_loss / eval_steps, correct / total


def train_audioset(config, checkpoint_dir=None, train_files=None, valid_files=None, test_files_list=None):
    # Load data
    in_features, out_features, train_set, train_embs, train_enc = data.load_data(*train_files)
    _, _, valid_set, valid_embs, valid_enc = data.load_data(*valid_files)

    train_loader = DataLoader(dataset=train_set, batch_size=config['batch_size'], shuffle=True)
    valid_loader = DataLoader(dataset=valid_set, batch_size=config['batch_size'], shuffle=True)

    test_loaders, test_embs_list, test_enc_list = [], [], []
    for test_files in test_files_list:
        _, _, test_set, test_embs, test_enc = data.load_data(*test_files)

        test_loader = DataLoader(dataset=test_set, batch_size=config['batch_size'], shuffle=True)

        test_loaders.append(test_loader)
        test_embs_list.append(test_embs)
        test_enc_list.append(test_enc)

    # Get model
    model = get_model(config, in_features, out_features)

    # Configure optimizer & loss
    criterion = objective
    optimizer = optim.SGD(model.parameters(), lr=config['lr'], momentum=config['momentum'], weight_decay=config['l2'])

    if checkpoint_dir:
        model_state, optimizer_state = torch.load(os.path.join(checkpoint_dir, 'checkpoint'))
        model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)

    for epoch in range(config['epochs']):  # Loop over the dataset multiple times

        train(model, criterion, optimizer, train_loader, train_embs, config)

        train_loss, train_acc = eval(model, criterion, train_loader, train_embs, config)
        valid_loss, valid_acc = eval(model, criterion, valid_loader, valid_embs, config)

        test_dict = {}
        for i, (test_loader, test_embs, test_enc) in enumerate(zip(test_loaders, test_embs_list, test_enc_list)):
            test_loss, test_acc = eval(model, criterion, test_loader, test_embs, config)

            test_dict['test{}_loss'.format(i)] = test_loss
            test_dict['test{}_acc'.format(i)] = test_acc

        # Save the model to the trial directory: local_dir/exp_name/trial_name/checkpoint_<step>
        with tune.checkpoint_dir(step=epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, 'checkpoint')
            torch.save((model.state_dict(), optimizer.state_dict()), path)

        # Send the current training result back to Tune
        tune.report(train_loss=train_loss, train_acc=train_acc,
                    valid_loss=valid_loss, valid_acc=valid_acc, **test_dict)

    print('Finished Training')


# Main
if __name__ == '__main__':
    exp_no = int(sys.argv[1])

    # Check running mode
    smoke_test = params.local_mode

    local_dir = '?'

    train_files = (params.train_partition_csv, params.embedding_npz)
    valid_files = (params.valid_partition_csv, params.embedding_npz)
    test_files_list = [(params.test_partition1_csv, params.embedding_npz),
                       (params.test_partition2_csv, params.embedding_npz),
                       (params.test_partition3_csv, params.embedding_npz)]

    model_name = 'Baseline'

    exp_name = '{}-{}'.format(model_name, exp_no)

    # Configure computing resources
    num_cpus, num_gpus, num_trials = 1, 0, 1
    ray.init(local_mode=False, num_cpus=num_cpus, num_gpus=num_gpus)

    min_num_epochs, max_num_epochs = 1000, 3000

    # Configure hyper-parameter search space  - start

    search_space = {
        'model': model_name,
        'epochs': max_num_epochs,
        'batch_size': tune.grid_search([32]),
        'lr': tune.grid_search([1e-5]),
        'momentum': tune.grid_search([0.9]),
        'l2': tune.grid_search([1e-1]),  # L2 regularization coefficient
        'delta': tune.grid_search([1.0])
    }

    search_space = get_spec(search_space)

    # Configure hyper-parameter search space  - end

    # Early stopping
    scheduler = ASHAScheduler(
        time_attr='training_iteration',
        max_t=max_num_epochs,
        grace_period=min_num_epochs,
        reduction_factor=4
    )

    # Search algorithm
    # search_alg = HyperOptSearch()

    # Restore previous search state checkpoint
    # search_alg_state = os.path.join(local_dir, exp_name)
    # if os.path.isdir(search_alg_state):
    #     print('Restore search state:', search_alg_state)
    #     search_alg.restore_from_dir(search_alg_state)

    # Repeat each trial 3 times, not recommended to use with TrialSchedulers
    # search_alg = Repeater(searcher=search_alg, repeat=3)
    # search_alg = ConcurrencyLimiter(search_alg, max_concurrent=max(num_cpus, num_gpus))

    # Progress reporter
    reporter = CLIReporter()
    reporter.add_metric_column(metric='train_loss')
    reporter.add_metric_column(metric='train_acc')
    reporter.add_metric_column(metric='valid_loss')
    reporter.add_metric_column(metric='valid_acc')

    for i in range(len(test_files_list)):
        reporter.add_metric_column(metric='test{}_loss'.format(i))
        reporter.add_metric_column(metric='test{}_acc'.format(i))

    # Ray tune - local_dir/exp_name/trial_name_x
    analysis = tune.run(
        partial(train_audioset, train_files=train_files, valid_files=valid_files, test_files_list=test_files_list),
        metric='valid_acc',
        mode='max',
        name=exp_name,
        stop={'training_iteration': 1 if smoke_test else max_num_epochs},
        config=search_space,
        resources_per_trial={'cpu': num_cpus, 'gpu': num_gpus},
        num_samples=num_trials,
        local_dir=local_dir,
        # search_alg=search_alg,
        scheduler=scheduler,
        keep_checkpoints_num=5,
        checkpoint_score_attr='valid_acc',
        progress_reporter=reporter,
        log_to_file=('stdout.log', 'stderr.log'),
        # max_failures=2,
        fail_fast=True,
        # resume='ERRORED_ONLY',
        queue_trials=False,
        reuse_actors=True,
        raise_on_failed_trial=True
    )

    # Select the best trial
    best_trial = analysis.get_best_trial(metric='valid_acc', mode='max', scope='all')
    print('Best trial: {}'.format(best_trial.trial_id))
    print('Best trial config: {}'.format(best_trial.config))
    print('Best trial validation accuracy: {}'.format(best_trial.metric_analysis['valid_acc']['max']))

    best_checkpoint_dir = analysis.get_best_checkpoint(trial=best_trial, metric='valid_acc', mode='max')
    model_state, optimizer_state = torch.load(os.path.join(best_checkpoint_dir, 'checkpoint'))
    print('{} Best checkpoint: {}'.format(exp_no, best_checkpoint_dir))
