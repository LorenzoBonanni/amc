# Code for "AMC: AutoML for Model Compression and Acceleration on Mobile Devices"
# Yihui He*, Ji Lin*, Zhijian Liu, Hanrui Wang, Li-Jia Li, Song Han
# {jilin, songhan}@mit.edu

import os

import numpy as np
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from transformers import EfficientNetImageProcessor
from PIL import Image
from lib.classes import IMAGENET2012_CLASSES


class MyDataset(Dataset):
    def __init__(self, image_paths, labeltoid, transform=None):
        self.base_path = image_paths
        self.image_names = os.listdir(image_paths)
        self.transform = transform
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.labeltoid = labeltoid

    def get_class_label(self, image_name):
        label_id = image_name.split('_')[0]
        y = IMAGENET2012_CLASSES[label_id]
        return y

    def __getitem__(self, index):
        image_path = self.base_path + '/' + self.image_names[index]
        x = Image.open(image_path)
        x_np = np.asarray(x)
        if x_np.ndim == 2:
            print("ERROR")
        y = self.get_class_label(image_path.split('/')[-1])
        y = torch.as_tensor(self.labeltoid[y])
        if self.transform is not None:
            x = self.transform(x)
            x = x.convert_to_tensors('pt')
            x = x['pixel_values']
        return x, y

    def __len__(self):
        return len(self.image_names)


def get_dataset(net, dset_name, batch_size, n_worker, data_root='../../data'):
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
        train_dir = os.path.join(data_root, 'train')
        val_dir = os.path.join(data_root, 'val')
        # transform = torchvision.transforms.Compose([
        #     transforms.Resize(380),
        #     transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                          std=[0.229, 0.224, 0.225]),
        #     # transforms.ToTensor()
        # ]
        # )
        transform = EfficientNetImageProcessor.from_pretrained("google/efficientnet-b4")
        train_dataset = MyDataset(image_paths=train_dir, transform=transform, labeltoid=net.config.label2id)
        val_dataset = MyDataset(image_paths=val_dir, transform=transform, labeltoid=net.config.label2id)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=n_worker,
                                                   pin_memory=True, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, num_workers=n_worker,
                                                 pin_memory=True, shuffle=True)

        n_class = 1000

    else:
        raise NotImplementedError

    return train_loader, val_loader, n_class


def get_split_dataset(net, dset_name, batch_size, n_worker, val_size, data_root='../data',
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
        val_dir = os.path.join(data_root, 'val')
        transform = EfficientNetImageProcessor.from_pretrained("google/efficientnet-b4")
        train_dataset = MyDataset(image_paths=train_dir, transform=transform, labeltoid=net.config.label2id)
        val_dataset = MyDataset(image_paths=val_dir, transform=transform, labeltoid=net.config.label2id)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=n_worker,
                                                   pin_memory=True, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, num_workers=n_worker,
                                                 pin_memory=True, shuffle=True)

        n_class = 1000
    else:
        raise NotImplementedError

    return train_loader, val_loader, n_class
