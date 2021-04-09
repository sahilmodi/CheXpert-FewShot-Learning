from __future__ import print_function
from dataset.chexpert_helper import get_data
from yacs.config import CfgNode

import os
from PIL import Image
import numpy as np

import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset


class Chexpert(Dataset):
    """support Chexpert dataset"""
    def __init__(self, args, partition='train', pretrain=True, is_sample=False, k=4096, transform=None):
        super(Dataset, self).__init__()
        self.data_root = args.data_root
        self.partition = partition
        self.data_aug = args.data_aug
        self.pretrain = pretrain

        if transform is None:
            if self.partition == 'train' and self.data_aug:
                self.transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToPILImage(),
                    transforms.RandomAffine(
                        degrees=(-15, 15),
                        translate=(0.05, 0.05),
                        scale=(0.95, 1.05)
                    ),
                    transforms.ToTensor(),
                ])
            else:
                self.transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                ])
        else:
            self.transform = transform

        # build a config node
        cfg = CfgNode()
        cfg.BATCH_SIZE = 128
        cfg.NUM_WORKERS = 1
        cfg.PATH = self.data_root
    
        print("- Starting to get data in dictionary form")
        data = get_data(cfg, partition)
        print("- Finished getting data in dictionary form")

        self.imgs = data["data"]
        self.labels = data["labels"]

        # pre-process for contrastive sampling
        self.k = k
        self.is_sample = is_sample
        if self.is_sample:
            self.labels = np.asarray(self.labels)
            self.labels = self.labels - np.min(self.labels)
            num_classes = np.max(self.labels) + 1

            self.cls_positive = [[] for _ in range(num_classes)]
            for i in range(len(self.imgs)):
                self.cls_positive[self.labels[i]].append(i)

            self.cls_negative = [[] for _ in range(num_classes)]
            for i in range(num_classes):
                for j in range(num_classes):
                    if j == i:
                        continue
                    self.cls_negative[i].extend(self.cls_positive[j])

            self.cls_positive = [np.asarray(self.cls_positive[i]) for i in range(num_classes)]
            self.cls_negative = [np.asarray(self.cls_negative[i]) for i in range(num_classes)]
            self.cls_positive = np.asarray(self.cls_positive)
            self.cls_negative = np.asarray(self.cls_negative)

    def __getitem__(self, item):
        img = np.asarray(self.imgs[item]).astype('uint8')
        img = self.transform(torch.Tensor(img))
        target = self.labels[item] - min(self.labels)

        if not self.is_sample:
            return img, target, item
        else:
            pos_idx = item
            replace = True if self.k > len(self.cls_negative[target]) else False
            neg_idx = np.random.choice(self.cls_negative[target], self.k, replace=replace)
            sample_idx = np.hstack((np.asarray([pos_idx]), neg_idx))
            return img, target, item, sample_idx

    def __len__(self):
        return len(self.labels)


class MetaChexpert(Chexpert):

    def __init__(self, args, partition='train', train_transform=None, test_transform=None, fix_seed=True):
        super(MetaChexpert, self).__init__(args, partition, False)
        self.fix_seed = fix_seed
        self.n_ways = args.n_ways
        self.n_shots = args.n_shots
        self.n_queries = args.n_queries
        self.n_test_runs = args.n_test_runs
        self.n_aug_support_samples = args.n_aug_support_samples
        if train_transform is None:
            self.train_transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToPILImage(),
                    transforms.RandomAffine(
                        degrees=(-15, 15),
                        translate=(0.05, 0.05),
                        scale=(0.95, 1.05)
                    ),
                    transforms.ToTensor(),
                ])

        else:
            self.train_transform = train_transform

        if test_transform is None:
            self.test_transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                ])
        else:
            self.test_transform = test_transform
        
        self.data = {}
        for idx in range(self.imgs.shape[0]):
            if self.labels[idx] not in self.data:
                self.data[self.labels[idx]] = []
            self.data[self.labels[idx]].append(self.imgs[idx])
        self.classes = list(self.data.keys())

    def __getitem__(self, item):
        if self.fix_seed:
            np.random.seed(item)
        cls_sampled = np.random.choice(self.classes, self.n_ways, False)
        support_xs = []
        support_ys = []
        query_xs = []
        query_ys = []
        for idx, cls in enumerate(cls_sampled):
            imgs = np.asarray(self.data[cls]).astype('uint8')
            support_xs_ids_sampled = np.random.choice(range(imgs.shape[0]), self.n_shots, False)
            support_xs.append(imgs[support_xs_ids_sampled])
            support_ys.append([idx] * self.n_shots)
            query_xs_ids = np.setxor1d(np.arange(imgs.shape[0]), support_xs_ids_sampled)
            query_xs_ids = np.random.choice(query_xs_ids, self.n_queries, False)
            query_xs.append(imgs[query_xs_ids])
            query_ys.append([idx] * query_xs_ids.shape[0])
        support_xs, support_ys, query_xs, query_ys = np.array(support_xs), np.array(support_ys), np.array(
            query_xs), np.array(query_ys)

        num_ways, n_queries_per_way, channel, height, width = query_xs.shape
        query_xs = query_xs.reshape((num_ways * n_queries_per_way, channel, height, width))
        query_ys = query_ys.reshape((num_ways * n_queries_per_way,))

        support_xs = support_xs.reshape((-1, channel, height, width))
        if self.n_aug_support_samples > 1:
            support_xs = np.tile(support_xs, (self.n_aug_support_samples, 1, 1, 1))
            support_ys = np.tile(support_ys.reshape((-1,)), (self.n_aug_support_samples))
        support_xs = np.split(support_xs, support_xs.shape[0], axis=0)
        query_xs = query_xs.reshape((-1, channel, height, width))
        query_xs = np.split(query_xs, query_xs.shape[0], axis=0)

        support_xs = torch.stack(list(map(lambda x: self.train_transform(torch.Tensor(x).squeeze(0)), support_xs)))
        query_xs = torch.stack(list(map(lambda x: self.test_transform(torch.Tensor(x).squeeze(0)), query_xs)))

        return support_xs, support_ys, query_xs, query_ys

    def __len__(self):
        return self.n_test_runs


if __name__ == '__main__':
    args = lambda x: None
    args.n_ways = 5
    args.n_shots = 1
    args.n_queries = 5
    args.data_root = '/home/koyejolab/CheXpert/CheXpert-v1.0-small'
    args.data_aug = True
    args.n_test_runs = 5
    args.n_aug_support_samples = 1
    chexpert = Chexpert(args, 'train')
    print(len(chexpert))
    print(chexpert.__getitem__(500)[0].shape)

    metachexpert = MetaChexpert(args, 'val')
    print(len(metachexpert))
    print("support_xs: ", metachexpert.__getitem__(500)[0].size())
    print("support_ys: ", metachexpert.__getitem__(500)[1].shape)
    print("query_xs: ", metachexpert.__getitem__(500)[2].shape)
    print("query_ys: ", metachexpert.__getitem__(500)[3].shape)
