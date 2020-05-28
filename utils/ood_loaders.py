from __future__ import print_function
import os
import os.path
import numpy as np
import sys
import torchvision

if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle


class CIFAR_ood(torchvision.datasets.cifar.CIFAR10):
    def __init__(self, root, ood_path, ood_transform, severity, dataset, train=True,
                 transform=None, target_transform=None, download=False):

        super(CIFAR_ood, self).__init__(root, train=train, transform=transform,
                                        target_transform=target_transform, download=download)

        self.data = []
        self.targets = []

        if dataset == 'CIFAR10':
            dataset_name = 'CIFAR-10-C'
        elif dataset == 'CIFAR100':
            dataset_name = 'CIFAR-100-C'
        else:
            raise NotImplementedError

        self.data = np.load(os.path.join(ood_path, dataset_name, ood_transform + '.npy'))[10000 * severity:10000 * (severity + 1)]
        self.targets = np.load(os.path.join(ood_path, dataset_name, 'labels.npy'))

        # self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        # self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

        self._load_meta()