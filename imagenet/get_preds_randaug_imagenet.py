import os
import time
import torch, argparse
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from imagenet.utils_imagenet import Logger
from imagenet.randaugment_imagenet import BetterRandAugment, getRandomResizedCropAndInterpolationdef
import torchvision.transforms as transforms
from timm.models import create_model
import torchvision.datasets as datasets
from PIL import Image

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

    parser = argparse.ArgumentParser(description='Ensembling script')
    parser.add_argument('--model', type=str, default='resnet50')
    parser.add_argument('--num_samples', type=int, default=1000)
    parser.add_argument('--mode', type=str, default='ra')
    parser.add_argument('--fix_sign', action='store_true', default=False)
    parser.add_argument('--logits_dir', type=str, default='./logits/')
    parser.add_argument('--log_dir', type=str, default='./logs/')
    parser.add_argument('--data_path', type=str, default='~/imagenet/raw-data', metavar='PATH')
    parser.add_argument('--fname', type=str, default='unnamed', required=False)
    parser.add_argument('--N', type=int, default=3, metavar='N')
    parser.add_argument('--M', type=float, default=5, metavar='M')
    parser.add_argument('--batch_size', type=int, default=500)
    parser.add_argument('--num_workers', type=int, default=30)
    parser.add_argument('--report_ens', action='store_true', default=False)
    parser.add_argument('--use_val', action='store_true', default=False)
    parser.add_argument('--policy', type=str, default=False)

    args = parser.parse_args()
    args.dataset = 'ImageNet'
    args.method = args.mode
    args.num_samples = 1  if args.mode == 'cc'  else args.num_samples
    args.num_samples = 5  if args.mode == '5c'  else args.num_samples
    args.num_samples = 10 if args.mode == '10c' else args.num_samples

    args.im_size = model2imsize[args.model]

    print(args)

    logger = Logger(base=args.log_dir)

    print('Creating model:', args.model)
    model = create_model(args.model, num_classes=1000, in_chans=3, pretrained=True)
    model = torch.nn.DataParallel(model).cuda()

    if args.policy:
        policy = np.load(args.policy, allow_pickle=True)
        print('Policy: ', policy)
        print('Len: ', len(policy))

    full_ens_preds = []
    for try_ in range(args.num_samples if not args.policy else len(policy)):
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
        elif args.mode == 'ra':
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
        elif args.mode == '5c':
            transform_train = transforms.Compose([
                transforms.Resize(int(args.im_size/model2crop_pct[args.model]), Image.BICUBIC),
                transforms.FiveCrop(args.im_size),
                transforms.Lambda(lambda crops: crops[try_]),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=torch.tensor((0.485, 0.456, 0.406)), std=torch.tensor((0.229, 0.224, 0.225)))])
        elif args.mode == '10c':
            transform_train = transforms.Compose([
                transforms.Resize(int(args.im_size/model2crop_pct[args.model]), Image.BICUBIC),
                transforms.TenCrop(args.im_size),
                transforms.Lambda(lambda crops: crops[try_]),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=torch.tensor((0.485, 0.456, 0.406)), std=torch.tensor((0.229, 0.224, 0.225)))])

        elif args.mode == 'aa':
            if try_ == 0: print("Uning AA")
            from autoaugment import augmentation_transforms
            epoch_policy = [
             [('Posterize', 0.4, 8.0), ('Rotate', 0.6, 9.0)],
             [('Solarize', 0.6, 5.0), ('AutoContrast', 0.6, 5.0)],
             [('Equalize', 0.8, 8.0), ('Equalize', 0.6, 3.0)],
             [('Posterize', 0.6, 7.0), ('Posterize', 0.6, 6.0)],
             [('Equalize', 0.4, 7.0), ('Solarize', 0.2, 4.0)],
             [('Equalize', 0.4, 4.0), ('Rotate', 0.8, 8.0)],
             [('Solarize', 0.6, 3.0), ('Equalize', 0.6, 7.0)],
             [('Posterize', 0.8, 5.0), ('Equalize', 1.0, 2.0)],
             [('Rotate', 0.2, 3.0), ('Solarize', 0.6, 8.0)],
             [('Equalize', 0.6, 8.0), ('Posterize', 0.4, 6.0)],
             [('Rotate', 0.8, 8.0), ('Color', 0.4, 0.0)],
             [('Rotate', 0.4, 9.0), ('Equalize', 0.6, 2.0)],
             [('Equalize', 0.0, 7.0), ('Equalize', 0.8, 8.0)],
             [('Invert', 0.6, 4.0), ('Equalize', 1.0, 8.0)],
             [('Color', 0.6, 4.0), ('Contrast', 1.0, 8.0)],
             [('Rotate', 0.8, 8.0), ('Color', 1.0, 2.0)],
             [('Color', 0.8, 8.0), ('Solarize', 0.8, 7.0)],
             [('Sharpness', 0.4, 7.0), ('Invert', 0.6, 8.0)],
             [('ShearX', 0.6, 5.0), ('Equalize', 1.0, 9.0)],
             [('Color', 0.4, 0.0), ('Equalize', 0.6, 3.0)],
             [('Equalize', 0.4, 7.0), ('Solarize', 0.2, 4.0)],
             [('Solarize', 0.6, 5.0), ('AutoContrast', 0.6, 5.0)],
             [('Invert', 0.6, 4.0), ('Equalize', 1.0, 8.0)],
             [('Color', 0.6, 4.0), ('Contrast', 1.0, 8.0)],
             [('Equalize', 0.8, 8.0), ('Equalize', 0.6, 3.0)]]

            auto_aug = lambda x: augmentation_transforms.apply_policy(epoch_policy[try_], x)
            transform_train = transforms.Compose(
            [getRandomResizedCropAndInterpolationdef(args.im_size, scale=(0.08, 1.0)),
             auto_aug,
             transforms.ToPILImage(),
             transforms.ToTensor(),
             transforms.Normalize(
                 mean=torch.tensor((0.485, 0.456, 0.406)), std=torch.tensor((0.229, 0.224, 0.225)))])
        else:
            raise Exception('Unknown mode %s' % args.mode)

        valdir = os.path.join(args.data_path, 'val')
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
            fname = os.path.join(args.logits_dir, args.model, fname)

            if not args.use_val:
                np.savez(fname, log_prob)
                print('\033[93m'+'Saved to ' + fname +'\033[0m')

        # print('Last aug metrics: ', end='')
        # logger.add_metrics_ts(0, [log_prob], np.array(test_set.targets), args, time_=start)
        if args.report_ens:
            print('Full ens metrics: ', end='')
            logger.add_metrics_ts(try_, full_ens_preds, np.array(test_set.targets), args, time_=start, info=args.policy)
    logger.save(args, silent=False)

        # print('---%s--- ends' % try_, flush=True)

main()




