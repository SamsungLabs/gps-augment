import os
import time
import models
import tqdm
import numpy as np
import pandas as pd
import argparse
import torchvision
from collections import defaultdict

import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.models as tv_models

from metrics import metrics_kfold
from scipy.special import logsumexp
from utils.randaugment import BetterRandAugment

ood_transforms = ['brightness', 'defocus_blur', 'fog',
                  'gaussian_blur', 'glass_blur', 'jpeg_compression',
                  'motion_blur', 'shot_noise', 'spatter', 'zoom_blur', 'contrast',
                  'elastic_transform', 'frost', 'gaussian_noise', 'impulse_noise',
                  'pixelate', 'saturate', 'snow', 'speckle_noise'
                  ]

def default_to_regular(d):
    if isinstance(d, defaultdict):
        d = {k: default_to_regular(v) for k, v in d.items()}
    return d

def get_sd(fname, args):
    sd = torch.load(fname)
    if 'model_state' in sd:
        return sd['model_state']
    if 'state_dict' in sd:
        return sd['state_dict']
    return sd

def get_targets(loader, args):
    targets = []
    print('loading')
    for _, target in loader:
        targets += [target]
    targets = np.concatenate(targets)

    return targets

class Logger:
    def __init__(self, base='./logs/'):
        self.res = []
        self.base = base
        os.makedirs(base, exist_ok=True)
        self.df = None

    def add(self, ns, metrics, args, info='', end='\n', silent=False):
        for m in metrics:
            self.res += [[args.dataset, args.model, args.method, ns, m, metrics[m], info, args.fname]]
        if not silent:
            if 'temperature' in metrics:
                print('ns %s: acc %.4f, nll %.4f, temp %.4f' % (ns, metrics['acc'], metrics['ll'], metrics['temperature']), flush=True, end=end)
            else:
                print('ns %s: acc %.4f, nll %.4f' % (ns, metrics['acc'], metrics['ll']), flush=True, end=end)

    def save(self, args, silent=True):
        self.df = pd.DataFrame(
            self.res, columns=['dataset', 'model', 'method', 'n_samples', 'metric', 'value', 'info', 'fname'])
        dir = '%s-%s-%s-%s.csv' % (args.dataset, args.model, args.method, args.fname)
        dir = os.path.join(self.base, dir)
        if not silent:
            print('Saved to:', dir, flush=True)
        self.df.to_csv(dir)

    def print(self):
        print(self.df, flush=True)

    def add_metrics_ts(self, ns, log_probs, targets, args, time_=0, n_splits=2, return_metrics=False):

        if args.dataset == 'ImageNet':
            disable = ('misclass_MI_auroc', 'sce', 'ace')
            n_runs = 2
        else:
            n_runs = 5
            disable = ('misclass_MI_auroc', 'sce', 'ace', 'misclass_entropy_auroc@5', 'misclass_confidence_auroc@5')
        log_prob = logsumexp(np.dstack(log_probs), axis=2) - np.log(ns+1)
        metrics = metrics_kfold(log_prob, targets, n_splits=n_splits, n_runs=n_runs, disable=disable)
        self.add(ns+1, metrics, args, end=' ')

        args.method = args.method + ' (ts)'
        metrics_ts = metrics_kfold(log_prob, targets, n_splits=n_splits, n_runs=n_runs, temp_scale=True, disable=disable)
        # self.add(ns+1, metrics_ts, args, silent=True)
        self.add(ns+1, metrics_ts, args)
        args.method = args.method[:-5]
        print("time: %.3f" % (time.time() - time_))
        
        if return_metrics:
            return metrics

def get_model(args):
    if args.dataset == 'ImageNet':
        model = tv_models.__dict__['resnet50']()
        model = torch.nn.DataParallel(model).cuda()
    else:
        model_cfg = getattr(models, args.model)
        model = model_cfg.base(*model_cfg.args, num_classes=args.num_classes, **model_cfg.kwargs).cuda()

    return model

def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def save_checkpoint(dir, epoch, **kwargs):
    state = {'epoch': epoch}
    state.update(kwargs)
    filepath = os.path.join(dir, 'checkpoint-%d.pt' % epoch)
    torch.save(state, filepath)

def train_epoch(loader, model, criterion, optimizer):
    loss_sum = 0.0
    correct = 0.0

    model.train()

    for i, (input, target) in enumerate(loader):
        input = input.cuda()
        target = target.cuda()
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        output = model(input_var)
        loss = criterion(output, target_var)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_sum += loss.item() * input.size(0)
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target_var.data.view_as(pred)).sum().item()

    return {
        'loss': loss_sum / len(loader.dataset),
        'accuracy': correct / len(loader.dataset) * 100.0,
    }

def remove_bar():
    import sys
    sys.stdout.write("\033[F") #back to previous line
    sys.stdout.write("\033[K") #clear line

def one_sample_pred(loader, model, **kwargs):
    preds = []

    for i, (input, target) in enumerate(loader):
        input = input.cuda()
        output = model(input, **kwargs)
        log_probs = torch.nn.functional.log_softmax(output, dim=1)
        preds.append(log_probs.cpu().data.numpy())

    return np.vstack(preds)

def eval(loader, model, criterion):
    loss_sum = 0.0
    correct = 0.0

    model.eval()

    with torch.no_grad():
        for i, (input, target) in enumerate(loader):
            input = input.cuda()
            target = target.cuda()
            input_var = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target)

            output = model(input_var)
            loss = criterion(output, target_var)

            loss_sum += loss.item() * input.size(0)
            pred = output.data.max(1, keepdim=True)[1]
            correct += (pred == target_var.data.view_as(pred)).sum().item()

        return {
            'loss': loss_sum / len(loader.dataset),
            'accuracy': correct / len(loader.dataset) * 100.0,
        }

def predictions(test_loader, model, **kwargs):
    model.eval()
    preds = []
    targets = []
    for input, target in test_loader:
        input = input.cuda()
        output = model(input, **kwargs)
        log_probs = torch.nn.functional.log_softmax(output, dim=1)
        preds.append(log_probs.cpu().data.numpy())
        targets.append(target.numpy())
    return np.vstack(preds), np.concatenate(targets)

def get_data_randaugment(args):
    # Only validation data!
    if args.dataset in ['CIFAR10', 'CIFAR100']:
        method_name = args.method.split('_')[0]
        transform_train_cifar = transforms.Compose([
            BetterRandAugment(args.N, args.M),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_test_cifar = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        print('Loading dataset %s from %s' % (args.dataset, args.data_path))
        ds = getattr(torchvision.datasets, args.dataset)
        path = os.path.join(args.data_path, args.dataset.lower())
        train_set = ds(path, train=True, download=True, transform=transform_train_cifar)
        tr = transform_test_cifar if not args.aug_test else transform_train_cifar
        test_set = ds(path, train=False, download=True, transform=tr)
        loaders = {
            'train': torch.utils.data.DataLoader(
                train_set,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.num_workers,
                pin_memory=True
            ),
            'test': torch.utils.data.DataLoader(
                test_set,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=True
            )
        }
        num_classes = np.max(train_set.targets)+1
    else:
        raise Exception('Unknown dataset "%s"' % args.dataset)

    return loaders, num_classes


def _check_bn(module, flag):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        flag[0] = True


def check_bn(model):
    flag = [False]
    model.apply(lambda module: _check_bn(module, flag))
    return flag[0]


def reset_bn(module):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        module.running_mean = torch.zeros_like(module.running_mean)
        module.running_var = torch.ones_like(module.running_var)


def _get_momenta(module, momenta):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        momenta[module] = module.momentum


def _set_momenta(module, momenta):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        module.momentum = momenta[module]


def bn_update(loader, model, verbose=False, subset=None, **kwargs):
    """
        BatchNorm buffers update (if any).
        Performs 1 epochs to estimate buffers average using train dataset.

        :param loader: train dataset loader for buffers average estimation.
        :param model: model being update
        :return: None
    """
    if not check_bn(model):
        return
    model.train()
    momenta = {}
    model.apply(reset_bn)
    model.apply(lambda module: _get_momenta(module, momenta))
    n = 0
    num_batches = len(loader)

    with torch.no_grad():
        if subset is not None:
            num_batches = int(num_batches * subset)
            loader = itertools.islice(loader, num_batches)
        if verbose:

            loader = tqdm.tqdm(loader, total=num_batches)
        for input, _ in loader:
            input = input.cuda(non_blocking=True)
            input_var = torch.autograd.Variable(input)
            b = input_var.data.size(0)

            momentum = b / (n + b)
            for module in momenta.keys():
                module.momentum = momentum

            model(input_var, **kwargs)
            n += b

    model.apply(lambda module: _set_momenta(module, momenta))

