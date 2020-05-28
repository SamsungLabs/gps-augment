import os
import time
import torch, argparse
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from imagenet.utils_imagenet import Logger, get_parser_ens, ood_transforms, default_to_regular
from imagenet.randaugment_imagenet import BetterRandAugment, getRandomResizedCropAndInterpolationdef
import torchvision.transforms as transforms
from timm.models import create_model
import torchvision.datasets as datasets
from PIL import Image
from collections import defaultdict

import warnings
warnings.filterwarnings("ignore")


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
    model2imsize   = {
        'resnet50': 224,
        'efficientnet_b2': 260,
        'tf_efficientnet_b5': 456,
        'tf_efficientnet_l2_ns': 800,
        'tf_efficientnet_l2_ns_475': 475
    }
    model2crop_pct = {
        'resnet50': 0.875,
        'efficientnet_b2': 0.875,
        'tf_efficientnet_b5': 0.934,
        'tf_efficientnet_l2_ns': 0.961,
        'tf_efficientnet_l2_ns_475': 0.936}
    torch.backends.cudnn.benchmark = True

    parser = argparse.ArgumentParser(description='Ensembling script for OOD')
    parser.add_argument('--model', type=str, default='resnet50')
    parser.add_argument('--num_tta', type=int, default=1000)
    parser.add_argument('--mode', type=str, default='ra')
    parser.add_argument('--fix_sign', action='store_true', default=False)
    parser.add_argument('--log_dir', type=str, default='./logits/')
    parser.add_argument('--data_path', type=str, default='../data/ImageNetC/', metavar='PATH', help='Path to ImageNet-C files')
    parser.add_argument('--fname', type=str, default='RApreds', required=False)
    parser.add_argument('--N', type=int, default=3, metavar='N')
    parser.add_argument('--M', type=float, default=5, metavar='M')
    parser.add_argument('--batch_size', type=int, default=500)
    parser.add_argument('--num_workers', type=int, default=30)
    parser.add_argument('--report_ens', action='store_true', default=False)
    parser.add_argument('--use_val', action='store_true', default=False)
    parser.add_argument('--policy', type=str, default=False)

    parser.add_argument('--corruptions', type=int, nargs='+', default=None, help='Indices of corruptions to evaluate on, default: all')
    parser.add_argument('--ood_exp_name', type=str, default='test', help='Name of experiment')
    parser.add_argument('--ood_path', type=str, default='./ood_res/', help='Folder to store results')

    args = parser.parse_args()
    args.dataset = 'ImageNet'
    args.method = 'randaugment'
    args.num_tta = 1 if args.mode == 'cc' else args.num_tta
    args.im_size = model2imsize[args.model]
    if args.corruptions is None:
        args.corruptions = np.arange(19)

    print(args)

    logger = Logger(base=args.log_dir)

    print('Creating model:', args.model)
    model = create_model(args.model, num_classes=1000, in_chans=3, pretrained=True)
    model = torch.nn.DataParallel(model).cuda()

    if args.policy:
        policy = np.load(args.policy, allow_pickle=True)
        print('Policy: ', policy)
        print('Len: ', len(policy))

    ood_ens_acc = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    ood_ens_ll = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))

    ood_transforms_name = ''
    ood_transforms_list = []
    for transform_ind in args.corruptions:
        ood_transforms_name += str(transform_ind) + '_'
        ood_transforms_list.append(ood_transforms[transform_ind])
    ood_transforms_name = ood_transforms_name[:-1]

    for ood_transform in ood_transforms_list:
        print('#' * 120)
        print('Transform for OOD data: ', ood_transform)
        for ood_severity in range(5):
            print('*' * 80)
            print('OOD transform severity: ', ood_severity + 1)
            full_ens_preds = []
            for try_ in range(args.num_tta if not args.policy else len(policy)):
                start = time.time()

                current_policy = policy[try_] if args.policy else None

                if args.mode == 'cc':
                    if try_ == 0: print('\033[93m'+'=== Central crop mode ==='+'\033[0m')
                    transform_train = transforms.Compose([
                        transforms.Resize(int(args.im_size/model2crop_pct[args.model]), Image.BICUBIC),
                        transforms.CenterCrop(args.im_size),
                        transforms.ToTensor(),
                        transforms.Normalize(
                            mean=torch.tensor((0.485, 0.456, 0.406)), std=torch.tensor((0.229, 0.224, 0.225)))])
                elif args.mode == 'cf':
                    if try_ == 0: print('\033[93m'+'=== Random crops and flips mode ==='+'\033[0m')
                    transform_train = transforms.Compose([
                        getRandomResizedCropAndInterpolationdef(args.im_size, scale=(0.08, 1.0)),
                        transforms.ToTensor(),
                        transforms.Normalize(
                            mean=torch.tensor((0.485, 0.456, 0.406)), std=torch.tensor((0.229, 0.224, 0.225)))])
                else:
                    if try_ == 0: print('\033[93m'+'=== RandAugment mode ==='+'\033[0m')
                    rand_aug = BetterRandAugment(
                        args.N, args.M, True, False, transform=current_policy,
                        verbose=False, true_m0=False,
                        randomize_sign=not args.fix_sign, image_size=args.im_size)
                    transform_train = transforms.Compose(
                        [rand_aug, transforms.ToTensor(),
                         transforms.Normalize(
                             mean=torch.tensor((0.485, 0.456, 0.406)), std=torch.tensor((0.229, 0.224, 0.225)))])

                    current_transform = transform_train.transforms[0].get_transform_str()
                    # print('\033[93m'+'Using the following transform:'+current_transform+'\033[0m')

                valdir = os.path.join(args.data_path, ood_transform, str(ood_severity + 1))
                test_set = datasets.ImageFolder(valdir, transform_train)

                if args.use_val:
                    # print('\033[93m'+'Using val/test split'+'\033[0m')
                    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=0)
                    searchset_index, test_index = next(sss.split(np.array(range(len(test_set.targets))), np.array(test_set.targets)))
                    test_set.imgs = np.array(test_set.imgs)[test_index]
                    test_set.targets = np.array(test_set.targets)[test_index]
                    test_set.samples = np.array(test_set.samples)[test_index]
                    # print('searchset_index', searchset_index[:10])
                    # print('test_index', test_index[:10])

                loaders = {
                    'test': torch.utils.data.DataLoader(
                        test_set,
                        batch_size=args.batch_size,
                        shuffle=False,
                        num_workers=args.num_workers,
                        pin_memory=True
                    )
                }

                log_prob = one_sample_pred(loaders['test'], model)
                full_ens_preds.append(log_prob)

                if args.mode == 'ra':
                    fname = '%s-%s-%s-%s.npz'% (
                        'ImageNet', args.model, args.method,
                        args.fname + '#'+current_transform+'#' + 'N%d-M%d'%(args.N, args.M))
                    fname = os.path.join(args.log_dir, args.model, fname)

                    if not args.use_val:
                        # np.savez(fname, log_prob)
                        print('\033[93m'+'Saved to ' + fname +'\033[0m')

                # print('Last aug metrics: ', end='')
                # logger.add_metrics_ts(0, [log_prob], np.array(test_set.targets), args, time_=start)
                if args.report_ens:
                    print('Full ens metrics: ', end='')
                    ens_metrics = logger.add_metrics_ts(try_, full_ens_preds, np.array(test_set.targets), args, time_=start, return_metrics=True)
                    ood_ens_acc[ood_transform][str(ood_severity)][str(try_)] = ens_metrics['acc']
                    ood_ens_ll[ood_transform][str(ood_severity)][str(try_)] = ens_metrics['ll']

                # print('---%s--- ends' % try_, flush=True)

            full_exp_name = args.dataset + '-' + args.model + '-' +\
                            str(args.num_tta) + '-' + ood_transforms_name + '-' + args.ood_exp_name
            full_res = {
                        'ens_acc': default_to_regular(ood_ens_acc),
                        'ens_ll': default_to_regular(ood_ens_ll),
                        }
            np.save(os.path.join(args.ood_path, full_exp_name), full_res)

main()




