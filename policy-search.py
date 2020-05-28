import os
import numpy as np
import glob
import argparse
import torch
from torch.nn.functional import log_softmax
from scipy.special import logsumexp
from tqdm import tqdm
import time
from sklearn.model_selection import StratifiedShuffleSplit
from metrics import metrics_kfold
import multiprocessing

def logmeansoftmax(data, classes_axis, members_axis, b=None):
    if b is None:
        n_samples = data.shape[members_axis]
    else:
        n_samples = np.array(b).sum()
    logsoftmaxed = log_softmax(torch.Tensor(data), dim=classes_axis).data.numpy()
    logmean = logsumexp(logsoftmaxed, axis=members_axis, b=b) - np.log(n_samples)
    return logmean

def eval_metrics(args):
    new_i, preds, targets, ens_preds, group_indices, select_ts = args
    if ens_preds is None:
        cur_ens_preds = preds
    else:
        dstacked = np.dstack([ens_preds, preds])
        b = (len(group_indices), 1)
        cur_ens_preds = logmeansoftmax(dstacked, classes_axis=1, members_axis=2, b=b)
    return new_i, cur_ens_preds, metrics_kfold(cur_ens_preds, targets, temp_scale=select_ts, n_splits=1, n_runs=1)

def select_greedily_on_ens(all_preds, all_targets, with_replacement, search_set_len, backward=False, select_only=None, select_ts=True, select_by='ll', num_workers=1):
    if backward:
        raise NotImplementedError

    val_preds = all_preds[:, :search_set_len, :]
    val_targets = all_targets['search_set']

    group_indices = []
    group_preds = None
    if select_only is None:
        select_only = val_preds.shape[0]
    pool = None
    if num_workers > 1:
        pool = multiprocessing.Pool(num_workers)
    for new_member_i in range(select_only):
        print(new_member_i, end=' ', flush=True)
        start = time.time()
        best_metric = None
        best_i = None
        best_ens_preds = None

        new_preds_and_metrics = []

        args = [(new_i, val_preds[new_i, :, :], val_targets, group_preds, group_indices, select_ts)
    for new_i in range(val_preds.shape[0])
    if (with_replacement or new_i not in group_indices)]

        if num_workers <= 1:
            new_metrics = map(eval_metrics, args)
        else:
            new_metrics = pool.imap(eval_metrics, args)

        for new_i, cur_ens_preds, cur_metrics in new_metrics:
            cur_metric = cur_metrics[select_by]
            if best_i is None or cur_metric > best_metric:
                best_i = new_i
                best_metric = cur_metric
                best_metrics = cur_metrics
                best_ens_preds = cur_ens_preds

        group_indices.append(best_i)
        group_preds = best_ens_preds
        print('(%d - %.4f, %.4f, %.4f - %.2f sec) %s'%(best_i, best_metrics['acc'], best_metrics['ll'], best_metrics['temperature'] if 'temperature' in best_metrics else None, time.time() - start, short_names[best_i]), flush=True)
    print()
    pool.close()
    return np.array(group_indices)

parser = argparse.ArgumentParser(description='Policy search script')
parser.add_argument('--predictions_dir', type=str, required=True,
    help='Directory with predicitons')
parser.add_argument('--output', type=str, required=True,
    help='Ouput file name')
parser.add_argument('--select_by', type=str, default='ll',
    help='Metric for policy search (ll or acc, default: ll)')
parser.add_argument('--no_select_ts', action='store_true', default=False,
    help='Disable calibration during search')
parser.add_argument('--select_num', type=int, default=100, metavar='N',
    help='Length of the resulting policy (default: 100)')
parser.add_argument('--num_workers', type=int, default=12, metavar='N',
    help='number of workers')

args = parser.parse_args()
files = [f for f in os.listdir(args.predictions_dir) if os.path.isfile(os.path.join(args.predictions_dir, f))]

dataset = files[0].split('-')[0]
arch = files[0].split('-')[1]

print('Found %d files' % len(files))

all_preds = []
all_names = []
len(files)
for f in files:
    if not f.startswith('%s-%s' % (dataset, arch)):
        continue
    try:
        all_preds.append(list(np.load(os.path.join(args.predictions_dir, f)).items())[0][1])
        all_names.append(f)
    except:
        pass
print('%d files are suitable' % len(all_preds))
all_preds = np.vstack([x[None, :, :] for x in all_preds])
print('Shape of all predictions:', all_preds.shape)

if dataset == 'CIFAR10':
    from torchvision.datasets import CIFAR10 as CIFAR
else:
    from torchvision.datasets import CIFAR100 as CIFAR
cifar = CIFAR('data', train=False, download=True)
targets = np.array(cifar.targets)

cifar_train = CIFAR('data', train=True, download=True)

sss = StratifiedShuffleSplit(n_splits=1, test_size=5000, random_state=0)
sss = sss.split(list(range(len(cifar_train.data))), cifar_train.targets)
train_idx, valid_idx = next(sss)

cifar_train.data = cifar_train.data[valid_idx]
cifar_train.targets = list(np.array(cifar_train.targets)[valid_idx])
cifar = cifar_train
targets = np.array(cifar_train.targets)

search_set_size = 5000
all_targets = {}
all_targets['search_set'] = targets[:search_set_size]

short_names = [x.split('#')[1] for x in all_names]

res = None
res = select_greedily_on_ens(all_preds, all_targets, with_replacement=True, search_set_len=search_set_size,
         backward=False, select_only=args.select_num, select_ts=not args.no_select_ts, select_by=args.select_by, num_workers=args.num_workers)
np.savez(args.output, np.array([eval(short_names[i]) for i in res], dtype=object))
