import glob
import torchvision.datasets as datasets
from sklearn.model_selection import StratifiedShuffleSplit

import os
import numpy as np
import argparse
import time

from selection.metrics_select import compute_test_metrics, metrics_kfold, to_probs, to_logits, logmeansoftmax

import multiprocessing


def eval_metrics(args):
    new_i, preds, searchset_index, targets, ens_preds, group_indices, select_by = args

    preds = np.load(preds)['arr_0'][searchset_index]
    assert len(preds) == len(targets)

    if ens_preds is None:
        cur_ens_preds = preds
    else:
        dstacked = np.dstack([ens_preds, preds])
        b = (len(group_indices), 1)
        cur_ens_preds = logmeansoftmax(dstacked, classes_axis=1, members_axis=2, b=b)

    metrics_ = metrics_kfold(cur_ens_preds, targets, temp_scale=True, n_splits=1, n_runs=1)
    return new_i, cur_ens_preds, metrics_


def select_greedily_on_ens(all_preds_, all_targets, with_replacement, select_only=5, select_by='ll', num_workers=1, model=None):
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=0)
    searchset_index, test_index = next(sss.split(np.array(range(len(all_targets))), np.array(all_targets)))
    val_targets = np.array(all_targets)[searchset_index]
    print('searchset_index', searchset_index[:10])
    print('test_index', test_index[:10])

    group_indices = []
    group_preds = None
    pool = None
    if num_workers > 1:
        pool = multiprocessing.Pool(num_workers)
    for new_member_i in range(select_only):
        all_preds = all_preds_
        print(new_member_i, end=' ', flush=True)
        start = time.time()
        best_metric = None
        best_i = None
        best_ens_preds = None

        args = [(new_i, all_preds[new_i], searchset_index, val_targets, group_preds, group_indices, select_by) for new_i
                in range(len(all_preds)) if (with_replacement or new_i not in group_indices)]

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

        group_indices.append((best_i, all_preds[best_i].split('/')[-1]))
        group_preds = best_ens_preds
        print('(%d - %.4f, %.4f - %.2f sec), %s' % (
        best_i, best_metrics['acc'], best_metrics['ll'], time.time() - start, all_preds[best_i].split('/')[-1]), flush=True)
        np.save('./imagenet/trained_pols/%s_%s' % (model, select_by), np.array(list(map(lambda x: x[1].split('#')[1], group_indices))))
    print()
    pool.close()


    return np.array(group_indices)


parser = argparse.ArgumentParser(description='Ensembling script')
parser.add_argument('--model', type=str, default='efficientnet_b2')
parser.add_argument('--select_by', type=str, default='ll')
parser.add_argument('--select_only', type=int, default=5)
parser.add_argument('--num_workers', type=int, default=5)
parser.add_argument('--logits_dir', type=str, default='./logits/')
args = parser.parse_args()
print('Args:', args)

valdir = os.path.join('~/imagenet/raw-data', 'val')
targets = datasets.ImageFolder(valdir).targets
logits_path = os.path.abspath(args.logits_dir) + '/%s/ImageNet*' % args.model
logits = glob.glob(logits_path)
print('Logits read', len(logits), 'in', logits_path)

res = select_greedily_on_ens(logits, targets, with_replacement=True, select_only=args.select_only,
    select_by=args.select_by, num_workers=args.num_workers, model=args.model)

print(list(res))
