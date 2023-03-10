import argparse
import os
import time
import torch
import torch.nn.functional as F
import torchvision
import sys
sys.path.append('.')
import numpy as np
import models
from utils import utils
import tabulate
import numpy as np
from utils.randaugment import BetterRandAugment
import torchvision.transforms as transforms
from scipy.special import logsumexp
from metrics import metrics_kfold
from sklearn.model_selection import StratifiedShuffleSplit


parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--dir', type=str, default=None, required=True, help='training directory (default: None)')
parser.add_argument('--fname', type=str, default=None, required=True, help='checkpoint and outputs file name')

parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset name (default: CIFAR10)')
parser.add_argument('--data_path', type=str, default=None, required=True, metavar='PATH',
                    help='path to datasets location')
parser.add_argument('--batch_size', type=int, default=128, metavar='N', help='input batch size (default: 128)')
parser.add_argument('--num_workers', type=int, default=4, metavar='N', help='number of workers (default: 4)')
parser.add_argument('--model', type=str, default=None, required=True, metavar='MODEL',
                    help='model name (default: None)')

parser.add_argument('--resume', type=str, default=None, metavar='CKPT',
                    help='checkpoint to resume training from (default: None)')

parser.add_argument('--epochs', type=int, default=200, metavar='N', help='number of epochs to train (default: 200)')
parser.add_argument('--save_freq', type=int, default=25, metavar='N', help='Save model once every N epochs (default: 25)')
parser.add_argument('--eval_freq', type=int, default=5, metavar='N', help='Evaluate once every N epochs (default: 5)')
parser.add_argument('--lr_init', type=float, default=0.1, metavar='LR', help='initial learning rate (default: 0.1)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum (default: 0.9)')
parser.add_argument('--wd', type=float, default=1e-4, help='weight decay (default: 1e-4)')
parser.add_argument('--schedule_linear', action='store_true', default=False, help='Use legacy linearl learning rate')

parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
parser.add_argument('--N', type=int, default=3, metavar='N', help='number of randaugmentations')
parser.add_argument('--M', type=float, default=5, metavar='M', help='magnitude of randaugmentations')
parser.add_argument('--randaugment', action='store_true', default=False, help='train with randaugment')
parser.add_argument('--num_tta', type=int, default=20, metavar='N', help='number of samples for test time augmentation')
parser.add_argument('--valid_size', type=int, default=0, metavar='N', help='Size of the validation set (default: empty validation set)')
parser.add_argument('--transforms', type=int, nargs='+', default=None, help='List of the transform indices used in BetterRandAugment (default: use all transforms)')

global_time = time.time()

args = parser.parse_args()
args.num_workers *= torch.cuda.device_count()

print('Preparing directory %s' % args.dir)
os.makedirs(args.dir, exist_ok=True)

torch.backends.cudnn.benchmark = True

print('Using model %s' % args.model)
model_cfg = getattr(models, args.model)

print('Loading dataset %s from %s' % (args.dataset, args.data_path))
ds = getattr(torchvision.datasets, args.dataset)
path = os.path.join(args.data_path, args.dataset.lower())
if args.randaugment:
    ra = BetterRandAugment(args.N, args.M, True, used_transforms=args.transforms)
    print('Using transforms:', 'all' if args.transforms is None else args.transforms)
    print('Using BetterRandAugment!')
    model_cfg.transform_train = transforms.Compose([ra, model_cfg.transform_train])
else:
    print('Using vanilla augmentation!')

train_set = ds(path, train=True, download=True, transform=model_cfg.transform_train)

train_idx = np.array(list(range(len(train_set.data))))
valid_idx = np.array([], dtype=np.int32)

if args.valid_size > 0:
    sss = StratifiedShuffleSplit(n_splits=1, test_size=args.valid_size, random_state=0)
    sss = sss.split(train_idx, train_set.targets)
    train_idx, valid_idx = next(sss)
    print(train_idx, valid_idx)

    train_set.data = train_set.data[train_idx]
    train_set.targets = list(np.array(train_set.targets)[train_idx])

valid_set = ds(path, train=True, download=True, transform=model_cfg.transform_test)

valid_set.data = valid_set.data[valid_idx]
valid_set.targets = list(np.array(valid_set.targets)[valid_idx])
valid_set.train = False

test_set = ds(path, train=False, download=True, transform=model_cfg.transform_test)
tta_set = ds(path, train=False, download=True, transform=model_cfg.transform_train)
loaders = {
    'train': torch.utils.data.DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    ),
    'valid': torch.utils.data.DataLoader(
        valid_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    ) if args.valid_size > 0 else None,
    'test': torch.utils.data.DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    ),
    'tta': torch.utils.data.DataLoader(
        tta_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
}
num_classes = 10 if args.dataset == 'CIFAR10' else 100

print('Preparing model')
model = model_cfg.base(*model_cfg.args, num_classes=num_classes, **model_cfg.kwargs)
if torch.cuda.device_count() > 1:
  print("Using", torch.cuda.device_count(), "GPUs!")
  model = torch.nn.DataParallel(model)
model.cuda()

def schedule_linear(epoch):
    t = (epoch) / args.epochs
    lr_ratio = 0.01
    if t <= 0.5:
        factor = 1.0
    elif t <= 0.9:
        factor = 1.0 - (1.0 - lr_ratio) * (t - 0.5) / 0.4
    else:
        factor = lr_ratio
    return args.lr_init * factor

def schedule(epoch):
    t = (epoch) / args.epochs
    return args.lr_init * (2. ** -int(t * 13))

if args.schedule_linear:
    schedule = schedule_linear

criterion = F.cross_entropy
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=args.lr_init,
    momentum=args.momentum,
    weight_decay=args.wd
)

start_epoch = 0
if args.resume is not None:
    print('Resume training from %s' % args.resume)
    checkpoint = torch.load(args.resume)
    start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])

columns = ['ep', 'lr', 'tr_loss', 'tr_acc', 'val_loss', 'val_acc', 'te_loss', 'te_acc', 'tta_loss', 'tta_acc', 'time']

for epoch in range(start_epoch, args.epochs):
    time_ep = time.time()

    lr = schedule(epoch)
    utils.adjust_learning_rate(optimizer, lr)
    model.train()
    train_res = utils.train_epoch(loaders['train'], model, criterion, optimizer)
    test_res = {'loss': None, 'accuracy': None}
    valid_res = {'loss': None, 'accuracy': None}
    tta_res = {'loss': None, 'accuracy': None}
    if (epoch + 1) % args.eval_freq == 0:
        with torch.no_grad():
            model.eval()
            # Testing
            log_prob, targets = utils.predictions(loaders['test'], model)
            metrics_ts = metrics_kfold(log_prob, targets, n_splits=2, n_runs=5, temp_scale=True)
            test_res['loss'] = -metrics_ts['ll']
            test_res['accuracy'] = metrics_ts['acc']
            # Validation
            if args.valid_size > 0:
                log_prob, targets = utils.predictions(loaders['valid'], model)
                metrics_ts = metrics_kfold(log_prob, targets, n_splits=2, n_runs=5, temp_scale=True)
                valid_res['loss'] = -metrics_ts['ll']
                valid_res['accuracy'] = metrics_ts['acc']
            # Test-time augmentation
            log_probs = []
            for i in range(args.num_tta):
                log_prob, targets = utils.predictions(loaders['tta'], model)
                log_probs.append(log_prob)
            log_prob = logsumexp(np.dstack(log_probs), axis=2) - np.log(args.num_tta)
            metrics_ts = metrics_kfold(log_prob, targets, n_splits=2, n_runs=5, temp_scale=True)
            tta_res['loss'] = -metrics_ts['ll']
            tta_res['accuracy'] = metrics_ts['acc']
            model.train()

    time_ep = time.time() - time_ep
    values = [epoch + 1, lr, train_res['loss'], train_res['accuracy'], valid_res['loss'], valid_res['accuracy'], test_res['loss'], test_res['accuracy'], tta_res['loss'], tta_res['accuracy'], time_ep]
    table = tabulate.tabulate([values], columns, tablefmt='simple', floatfmt='8.4f')
    if epoch % 40 == 0:
        table = table.split('\n')
        table = '\n'.join([table[1]] + table)
    else:
        table = table.split('\n')[2]
    print(table)

test_res = utils.eval(loaders['test'], model, criterion)
print('Final test res:', test_res)
test_preds, test_targets = utils.predictions(loaders['test'], model)
torch.save(model.state_dict(), os.path.join(args.dir, args.fname+'.pt'))
np.savez(os.path.join(args.dir, args.fname), test_preds=test_preds, test_targets=test_targets)
print('Global time: ', time.time() - global_time)
