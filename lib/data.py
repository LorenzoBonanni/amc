# Code for "AMC: AutoML for Model Compression and Acceleration on Mobile Devices"
# Yihui He*, Ji Lin*, Zhijian Liu, Hanrui Wang, Li-Jia Li, Song Han
# {jilin, songhan}@mit.edu

import os
import random
import numpy as np
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.io import read_image

from lib.classes import IMAGENET2012_CLASSES, TEST_FILE_TO_ID, IMAGENET_2012_LABELS


class MyDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.base_path = image_paths
        self.image_names = os.listdir(image_paths)
        self.transform = transform
        self.get_class_label = self.get_class_label_test if 'test' in image_paths else self.get_class_label_train

    def get_class_label_test(self, image_name):
        label_id = image_name.split('_')[-1].split('.')[0]
        y = IMAGENET2012_CLASSES[label_id]
        return y

    def get_class_label_train(self, image_name):
        y = IMAGENET_2012_LABELS[TEST_FILE_TO_ID[image_name]]
        return y

    def __getitem__(self, index):
        image_path = self.base_path + '/' + self.image_names[index]
        x = read_image(image_path)
        y = self.get_class_label(image_path.split('/')[-1])
        if self.transform is not None:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.base_path)

    class MyDataset(Dataset):
        def __init__(self, image_paths, transform=None):
            self.base_path = image_paths
            self.image_names = os.listdir(image_paths)
            self.transform = transform
            self.get_class_label = self.get_class_label_test if 'test' in image_paths else self.get_class_label_train

        def get_class_label_test(self, image_name):
            label_id = image_name.split('_')[-1].split('.')[0]
            y = IMAGENET2012_CLASSES[label_id]
            return y

        def get_class_label_train(self, image_name):
            label_id = TEST_FILE_TO_ID[image_name]
            y = IMAGENET_2012_LABELS[label_id]
            return y

        def __getitem__(self, index):
            image_path = self.base_path + '/' + self.image_names[index]
            x = read_image(image_path)
            y = self.get_class_label(image_path.split('/')[-1])
            if self.transform is not None:
                x = self.transform(x)
            return x, y

        def __len__(self):
            return len(self.image_names)


def get_dataset(dset_name, batch_size, n_worker, data_root='../../data'):
    cifar_tran_train = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
    cifar_tran_test = [
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
    print('=> Preparing data..')
    if dset_name == 'cifar10':
        transform_train = transforms.Compose(cifar_tran_train)
        transform_test = transforms.Compose(cifar_tran_test)
        trainset = torchvision.datasets.CIFAR10(root=data_root, train=True, download=True, transform=transform_train)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True,
                                                   num_workers=n_worker, pin_memory=True, sampler=None)
        testset = torchvision.datasets.CIFAR10(root=data_root, train=False, download=True, transform=transform_test)
        val_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False,
                                                 num_workers=n_worker, pin_memory=True)
        n_class = 10
    elif dset_name == 'imagenet':
        # get dir
        traindir = os.path.join(data_root, 'train')
        valdir = os.path.join(data_root, 'val')

        # preprocessing
        input_size = 224
        imagenet_tran_train = [
            transforms.RandomResizedCrop(input_size, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
        imagenet_tran_test = [
            transforms.Resize(int(input_size / 0.875)),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]

        train_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(traindir, transforms.Compose(imagenet_tran_train)),
            batch_size=batch_size, shuffle=True,
            num_workers=n_worker, pin_memory=True, sampler=None)

        val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(valdir, transforms.Compose(imagenet_tran_test)),
            batch_size=batch_size, shuffle=False,
            num_workers=n_worker, pin_memory=True)
        n_class = 1000

    else:
        raise NotImplementedError

    return train_loader, val_loader, n_class


def get_split_dataset(dset_name, batch_size, n_worker, val_size, data_root='../data',
                      use_real_val=True, shuffle=True):
    '''
        split the train set into train / val for rl search
    '''
    if shuffle:
        index_sampler = SubsetRandomSampler
    else:  # every time we use the same order for the split subset
        class SubsetSequentialSampler(SubsetRandomSampler):
            def __iter__(self):
                return (self.indices[i] for i in torch.arange(len(self.indices)).int())

        index_sampler = SubsetSequentialSampler

    print('=> Preparing data: {}...'.format(dset_name))
    if dset_name == 'cifar10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        trainset = torchvision.datasets.CIFAR100(root=data_root, train=True, download=True, transform=transform_train)
        if use_real_val:  # split the actual val set
            valset = torchvision.datasets.CIFAR10(root=data_root, train=False, download=True, transform=transform_test)
            n_val = len(valset)
            assert val_size < n_val
            indices = list(range(n_val))
            np.random.shuffle(indices)
            _, val_idx = indices[val_size:], indices[:val_size]
            train_idx = list(range(len(trainset)))  # all train set for train
        else:  # split the train set
            valset = torchvision.datasets.CIFAR10(root=data_root, train=True, download=True, transform=transform_test)
            n_train = len(trainset)
            indices = list(range(n_train))
            # now shuffle the indices
            np.random.shuffle(indices)
            assert val_size < n_train
            train_idx, val_idx = indices[val_size:], indices[:val_size]

        train_sampler = index_sampler(train_idx)
        val_sampler = index_sampler(val_idx)

        train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=False,
                                                   sampler=train_sampler,
                                                   num_workers=n_worker, pin_memory=True)
        val_loader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False, sampler=val_sampler,
                                                 num_workers=n_worker, pin_memory=True)
        n_class = 10
    elif dset_name == 'imagenet':
        train_dir = os.path.join(data_root, 'train')
        test_dir = os.path.join(data_root, 'test')
        # transform = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_convnets_processing_utils')
        transform = torch.nn.Sequential(
            transforms.Resize(380),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        )
        train_dataset = MyDataset(train_dir, transform)
        d = MyDataset(test_dir, transform)
        # random.shuffle(d.image_names)
        i_names = d.image_names
        # test_data = i_names[:int(len(i_names) * 0.9)]
        val_data = i_names[int(len(i_names) * 0.9):]
        # test_dataset = MyDataset(test_dir, transform)
        # test_dataset.image_names = test_data
        val_dataset = MyDataset(test_dir, transform)
        val_dataset.image_names = val_data

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=n_worker,
                                                   pin_memory=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, num_workers=n_worker,
                                                 pin_memory=True)

        n_class = 1000
    else:
        raise NotImplementedError

    return train_loader, val_loader, n_class
