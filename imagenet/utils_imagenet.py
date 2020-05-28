import os
import time
import glob
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
from imagenet.randaugment_imagenet import BetterRandAugment

from PIL import Image

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

class Logger:
    def __init__(self, base='./logs/'):
        self.res = []
        self.base = base
        os.makedirs(base, exist_ok=True)
        self.df = None

    def add(self, ns, metrics, args, info='', end='\n', silent=False):
        for m in metrics:
            self.res += [[args.dataset, args.model, args.method, ns, m, metrics[m], info]]
        if not silent:
            print('ns %s: acc %.5f, nll %.4f' % (ns, metrics['acc'], metrics['ll']), flush=True, end=end)

    def save(self, args, silent=True):
        self.df = pd.DataFrame(
            self.res, columns=['dataset', 'model', 'method', 'n_samples', 'metric', 'value', 'info'])
        dir = '%s-%s-%s-%s.cvs' % (args.dataset, args.model, args.method, args.fname)
        dir = os.path.join(self.base, dir)
        if not silent:
            print('Saved to:', dir, flush=True)
        self.df.to_csv(dir)

    def print(self):
        print(self.df, flush=True)

    def add_metrics_ts(self, ns, log_probs, targets, args, time_=0, info="", return_metrics=False):
        log_prob = logsumexp(np.dstack(log_probs), axis=2) - np.log(ns+1)
        if return_metrics:
            metrics = metrics_kfold(log_prob, targets, n_splits=2, n_runs=2, disable=('misclass_MI_auroc', 'sce', 'ace'), temp_scale=False)
            self.add(ns+1, metrics, args, end=' ', info=info)
            return metrics
        metrics_ts = metrics_kfold(log_prob, targets, n_splits=2, n_runs=2, temp_scale=True)
        self.add(ns+1, metrics_ts, args, end=' ', info=info)
        self.save(args, silent=True)
        print("time: %.3f" % (time.time() - time_))

def get_parser_ens():
    parser = argparse.ArgumentParser(description='Ensembling script')
    parser.add_argument(
        '--models_dir', type=str, nargs='+', default='~/megares')
    parser.add_argument(
        '--log_dir', type=str, default='./logs/')
    parser.add_argument(
        '--policy', type=str, default=None,
        help='The exported policy with RandAugment transforms (for get_predictions_randaugment only)')
    parser.add_argument(
        '--fname', type=str, default='unnamed', required=False,
        help='the fname will be added to the name of log file')
    parser.add_argument(
        '--dataset', type=str, default='CIFAR10',
        help='not influence dropout, onenet, and')
    parser.add_argument('--aug_test', action='store_true', default=False)
    parser.add_argument(
        '--data_path', type=str, default='../data', metavar='PATH',
        help='Works for CIFARs, ImegeNet path is hardcoded to ~/imagenet')
    parser.add_argument(
        '--im_size', type=int, default=260)
    parser.add_argument(
        '--model', type=str, default='')
    parser.add_argument(
        '--batch_size', type=int, default=256, metavar='N', help='input batch size (default: 256)')
    parser.add_argument(
        '--num_workers', type=int, default=44, metavar='N', help='number of workers (default: 4)')
    parser.add_argument(
        '--N', type=int, default=3, metavar='N', help='number of randaugmentations (only for ens-randaugment)')
    parser.add_argument(
        '--M', type=float, default=5, metavar='M', help='magnitude of randaugmentations (only for ens-randaugment)')
    parser.add_argument(
        '--bnN', type=int, default=3, metavar='N', help='number of randaugmentations (only for ens-randaugment)')
    parser.add_argument(
        '--bnM', type=float, default=5, metavar='M', help='magnitude of randaugmentations (only for ens-randaugment)')
    parser.add_argument('--bn_update', action='store_true', default=False)
    parser.add_argument('--num_tta', type=int, default=100, metavar='N', help='number of sample for test time augmentation')
    parser.add_argument('--no_tta', action='store_true', default=False)
    parser.add_argument('--valid', action='store_true', default=False)
    parser.add_argument('--silent', action='store_true', default=False, help='Do not save predictions in get_predictions_randaugment')
    parser.add_argument('--verbose', action='store_true', default=False, help='Verbose augmentations for get_predictions_randaugment')
    parser.add_argument('--true_m0', action='store_true', default=False, help='Do not apply any transform for M < 0.5; only for get_predictions_randaugment.py')
    parser.add_argument('--fix_sign', action='store_true', default=False, help='Disable random sign of Contrast, Color, Brightness and Sharpness; only for get_predictions_randaugment.py')

    return parser