import os
import time
import torch
import numpy as np
from scipy.special import logsumexp
from utils.ood_loaders import CIFAR_ood
from utils.utils import Logger, get_sd, get_model, get_targets, bn_update
from utils.randaugment import BetterRandAugment
from utils import utils
import torchvision
import torchvision.transforms as transforms
import models
from collections import defaultdict
import metrics
import argparse

from sklearn.model_selection import StratifiedShuffleSplit

import warnings
warnings.filterwarnings("ignore")

def get_parser_ens():
    parser = argparse.ArgumentParser(description='Script for obtaining predictions and ensembling on augmentations')
    parser.add_argument(
        '--models', type=str, nargs='+', help='List of models to evaluate')
    parser.add_argument(
        '--log_dir', type=str, default='./logs/')
    parser.add_argument(
        '--policy', type=str, default=None,
        help='Path to the augmentation policy with RandAugment transforms')
    parser.add_argument(
        '--fname', type=str, default='unnamed', required=False,
        help='the fname will be appended to the name of log file')
    parser.add_argument(
        '--dataset', type=str, default='CIFAR10',
        help='CIFAR10 / CIFAR100')
    parser.add_argument(
        '--data_path', type=str, default='../data', metavar='PATH',
        help='Location of the corrupted dataset')
    parser.add_argument(
        '--batch_size', type=int, default=256, metavar='N', help='input batch size (default: 256)')
    parser.add_argument(
        '--num_workers', type=int, default=4, metavar='N', help='number of workers (default: 4)')
    parser.add_argument(
        '--N', type=int, default=3, metavar='N', help='number of randaugmentations')
    parser.add_argument(
        '--M', type=float, default=5, metavar='M', help='Maximum magnitude of randaugmentations')
    parser.add_argument('--bn_update', action='store_true', default=False)
    parser.add_argument('--num_tta', type=int, default=100, metavar='N', help='number of sample for test time augmentation')
    parser.add_argument('--no_tta', action='store_true', default=False)
    parser.add_argument('--valid', action='store_true', default=False)
    parser.add_argument('--silent', action='store_true', default=False, help='Do not save predictions')
    parser.add_argument('--verbose', action='store_true', default=False, help='Verbose augmentations')
    parser.add_argument('--true_m0', action='store_true', default=False, help='Do not apply any randaugment transform for M < 0.5')
    parser.add_argument('--fix_sign', action='store_true', default=False, help='Disable random sign of Contrast, Color, Brightness and Sharpnes')
    parser.add_argument('--transforms', type=int, nargs='+', default=None, help='List of the transform indices used in BetterRandAugment (default: use all transforms)')
    parser.add_argument('--ood_exp_name', type=str, default='test', help='Name of the experiment')
    return parser

def one_sample_pred(loader, model, **kwargs):
    preds = []
    model.eval()

    for i, (input, target) in enumerate(loader):
        input = input.cuda()
        with torch.no_grad():
            output = model(input, **kwargs)
        log_probs = torch.nn.functional.log_softmax(output, dim=1)
        preds.append(log_probs.cpu().data.numpy())

    return np.vstack(preds)

def main():
    torch.backends.cudnn.benchmark = True
    args = get_parser_ens().parse_args()
    args.method = 'randaugment'
    args.aug_test = True
    print('>> Data-augmentation is terned *ON* !')

    print(args.models)
    print('Using the following snapshots:')
    print('\n'.join(args.models))

    args.dataset = args.models[0].split('/')[-1].split('-')[0]
    args.model = args.models[0].split('/')[-1].split('-')[1]
    print(args.model, args.dataset)

    num_tta = args.num_tta
    samples_per_policy = 1
    if args.policy is not None:
        policy = np.load(args.policy, allow_pickle=True)['arr_0']
        if args.num_tta > len(policy):
            num_tta = len(policy)
            samples_per_policy = args.num_tta // num_tta

    # loaders, num_classes = get_data_randaugment(args)
    path = os.path.join(args.data_path, args.dataset.lower())

    ds = CIFAR_ood

    if args.dataset == 'CIFAR10':
        args.num_classes = 10
    elif args.dataset == 'CIFAR100':
        args.num_classes = 100
    else:
        raise NotImplementedError

    ood_transforms_list = ['no_transform']
    ood_severity_list = [0.]

    ood_transforms_list = utils.ood_transforms
    ood_severity_list = np.arange(5)
    ood_ind_acc = defaultdict(lambda: defaultdict(list))
    ood_ind_ll = defaultdict(lambda: defaultdict(list))
    ood_ens_acc = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    ood_ens_ll = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    model_cfg = getattr(models, args.model)
    # TODO: add flag for random M
    print('WARNING: using random M')

    if args.no_tta:
        print('\033[93m'+'TTA IS DISABLED!'+'\033[0m')


    logger = Logger(base=args.log_dir)

    model = get_model(args)

    if args.valid:
        train_set = ds(path, train=True, download=True, transform=model_cfg.transform_train)
        sss = StratifiedShuffleSplit(n_splits=1, test_size=5000, random_state=0)
        train_idx = np.array(list(range(len(train_set.data))))
        sss = sss.split(train_idx, train_set.targets)
        train_idx, valid_idx = next(sss)

    for ood_transform in ood_transforms_list:
        print('#' * 120)
        print('Transform for OOD data: ', ood_transform)
        for ood_severity in ood_severity_list:
            print('*' * 80)
            print('OOD transform severity: ', ood_severity)
            full_ens_preds = []
            for try_ in range(num_tta):
                start = time.time()

                current_policy = None
                if args.policy is not None:
                    current_policy = policy[try_]
                    if current_policy is None:
                        current_policy = []

                if args.no_tta:
                    transform_train = model_cfg.transform_test
                    current_transform = 'None'
                    print('\033[93m'+'Using the following transform:'+'\033[0m')
                    print('\033[93m'+current_transform+'\033[0m')
                else:
                    transform_train = transforms.Compose([BetterRandAugment(args.N, args.M, True, False, transform=current_policy, verbose=args.verbose, true_m0=args.true_m0, randomize_sign=not args.fix_sign, used_transforms=args.transforms),
                                                          model_cfg.transform_train])
                    current_transform = transform_train.transforms[0].get_transform_str()
                    print('\033[93m'+'Using the following transform:'+'\033[0m')
                    print('\033[93m'+current_transform+'\033[0m')

                if args.valid:
                    print('\033[93m'+'Using the following objects for validation:'+'\033[0m')
                    print(train_idx, valid_idx)

                    test_set = ds(path, train=True, download=True, transform=transform_train)
                    test_set.data = test_set.data[valid_idx]
                    test_set.targets = list(np.array(test_set.targets)[valid_idx])
                    test_set.train = False
                else:
                    test_set = ds(path, args.data_path, ood_transform, ood_severity, args.dataset,
                                  train=False, download=True, transform=transform_train)

                loaders = {
                    # 'train': torch.utils.data.DataLoader(
                        # train_set,
                        # batch_size=args.batch_size,
                        # shuffle=True,
                        # num_workers=0,
                        # pin_memory=True
                    # ),
                    'test': torch.utils.data.DataLoader(
                        test_set,
                        batch_size=args.batch_size,
                        shuffle=False,
                        num_workers=args.num_workers,
                        pin_memory=True
                    )
                }

                # targets = get_targets(loaders['test'], args)

                # Load the model and update BN statistics (if run with a single model)
                if len(args.models) == 1:
                    try:
                        model.load_state_dict(get_sd(args.models[0], args))
                    except RuntimeError:
                        model = torch.nn.DataParallel(model).cuda()
                        model.load_state_dict(get_sd(args.models[0], args))
                    if args.bn_update:
                        bn_update(loaders['train'], model)
                        print('BatchNorm statistics updated!')

                log_probs = []
                ns = 0
                for fname in args.models:
                    # Load the model and update BN if several models are supplied
                    if len(args.models) > 1:
                        try:
                            model.load_state_dict(get_sd(fname, args))
                        except RuntimeError:
                            if hasattr(model, 'module'):
                                model.module.load_state_dict(get_sd(fname, args))
                            else:
                                model = torch.nn.DataParallel(model).cuda()
                                model.load_state_dict(get_sd(fname, args))
                        if args.bn_update:
                            bn_update(loaders['train'], model)
                            print('BatchNorm statistics updated!')
                    for _ in range(samples_per_policy):
                        ones_log_prob = one_sample_pred(loaders['test'], model)
                        log_probs.append(ones_log_prob)
                        ns += 1

                log_prob = logsumexp(np.dstack(log_probs), axis=2) - np.log(ns)
                full_ens_preds.append(log_prob)

                fname = '%s-%s-%s-%s.npz' % (args.dataset, args.model, args.method, '-'.join([os.path.basename(f) for f in args.models]) + args.fname + '#'+current_transform+'#' + 'N%d-M%d'%(args.N, args.M))
                if len(fname) > 255:
                    fname = '%s-%s-%s-%s.npz' % (args.dataset, args.model, args.method, os.path.basename(args.models[0]) + '-' +
                                                 '-'.join([os.path.basename(f)[-5:] for f in args.models[1:]]) + args.fname + '#'+current_transform+'#' + 'N%d-M%d'%(args.N, args.M))
                fname = os.path.join(args.log_dir, fname)
                if not args.silent:
                    np.savez(fname, log_prob)
                    print('\033[93m'+'Saved to ' + fname +'\033[0m')

                print('Last aug metrics: ', end='')
                ind_metrics = logger.add_metrics_ts(ns-1, log_probs, np.array(test_set.targets), args, time_=start, return_metrics=True)
                ood_ind_acc[ood_transform][str(ood_severity)].append(ind_metrics['acc'])
                ood_ind_ll[ood_transform][str(ood_severity)].append(ind_metrics['ll'])

                if num_tta == 1:
                    for i in range(99):
                        ood_ind_acc[ood_transform][str(ood_severity)].append(ind_metrics['acc'])
                        ood_ind_ll[ood_transform][str(ood_severity)].append(ind_metrics['ll'])
                print('Full ens metrics: ', end='')
                ens_metrics = logger.add_metrics_ts(try_, full_ens_preds, np.array(test_set.targets), args, time_=start, return_metrics=True)
                ood_ens_acc[ood_transform][str(ood_severity)][str(try_)].append(ens_metrics['acc'])
                ood_ens_ll[ood_transform][str(ood_severity)][str(try_)].append(ens_metrics['ll'])

                if num_tta == 1:
                    for i in range(99):
                        ood_ens_acc[ood_transform][str(ood_severity)][str(try_)].append(ens_metrics['acc'])
                        ood_ens_ll[ood_transform][str(ood_severity)][str(try_)].append(ens_metrics['ll'])
                # logger.save(args)

                # os.makedirs('./.megacache', exist_ok=True)
                # logits_pth = '.megacache/logits_%s-%s-%s-%s-%s'
                # logits_pth = logits_pth % (args.dataset, args.model, args.method, ns+1, try_)
                # log_prob = logsumexp(np.dstack(log_probs), axis=2) - np.log(ns+1)
                # np.save(logits_pth, log_prob)

                print('---%s--- ends' % try_, flush=True)
                # print('Time: %.2f' % (time.time() - start))

    full_exp_name = args.dataset + '-' + args.models[0].split('/')[-1].split('-')[1] + '-' +\
                    str(args.num_tta) + '-' + args.ood_exp_name
    full_res = {'ind_acc': utils.default_to_regular(ood_ind_acc),
                'ind_ll': utils.default_to_regular(ood_ind_ll),
                'ens_acc': utils.default_to_regular(ood_ens_acc),
                'ens_ll': utils.default_to_regular(ood_ens_ll),
                }
    np.save(os.path.join(args.log_dir, full_exp_name), full_res)

main()




