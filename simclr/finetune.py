import sys
from pathlib import Path

from torch._C import autocast_decrement_nesting
sys.path.append(str(Path(__file__).absolute().parent.parent))

import numpy as np
import os
import yaml
import argparse
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
import warnings

import torch
import torchvision
from torchvision import datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn


from utils.misc import set_seed
from simclr.data_aug.chexpert_dataset import ChexpertDatasetFinetune

def get_stl10_data_loaders(download, shuffle=False, batch_size=256):
    train_dataset = datasets.STL10('./data', split='train', download=download, transform=transforms.ToTensor())
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=0, drop_last=False, shuffle=shuffle)
    test_dataset = datasets.STL10('./data', split='test', download=download, transform=transforms.ToTensor())
    test_loader = DataLoader(test_dataset, batch_size=2*batch_size, num_workers=10, drop_last=False, shuffle=shuffle)
    return train_loader, test_loader

def get_cifar10_data_loaders(download, shuffle=False, batch_size=256):
    train_dataset = datasets.CIFAR10('./data', train=True, download=download, transform=transforms.ToTensor())
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=0, drop_last=False, shuffle=shuffle)
    test_dataset = datasets.CIFAR10('./data', train=False, download=download, transform=transforms.ToTensor())
    test_loader = DataLoader(test_dataset, batch_size=2*batch_size, num_workers=10, drop_last=False, shuffle=shuffle)
    return train_loader, test_loader

def get_chexpert_data_loaders(config, training_data_size):
    train_dataset = ChexpertDatasetFinetune(config['data'], 224, num=training_data_size)
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], num_workers=12, drop_last=False, shuffle=True)
    test_dataset = ChexpertDatasetFinetune(config['data'], 224, split="test")
    test_loader = DataLoader(test_dataset, batch_size=2*config['batch_size'], num_workers=12, drop_last=False)
    return train_loader, test_loader

def get_auc(labels, y):
        try:
            return metrics.roc_auc_score(labels.cpu().numpy(), y.detach().cpu().numpy(), average='weighted')
        except ValueError:
            return np.nan

def get_prc(labels, y):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            return metrics.average_precision_score(labels.cpu().numpy(), y.detach().cpu().numpy(), average='weighted')

def accuracy(output, target):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        '''
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
        '''
        auc = get_auc(target, output)
        prc = get_prc(target, output)
        return auc, prc


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('-d', "--dir", type=Path, help="Path to model directory.")
    ap.add_argument('-g', "--gpu", type=int, help="Which gpu to use.")
    ap.add_argument('-tds', "--training_size", type=int, default=1000, help="How much training data to use.")
    ap.add_argument('-s', "--seed", type=int, default=0, help="Set random seed.")
    return ap.parse_args()


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args = parse_args()
    with open(os.path.join(args.dir / 'config.yml')) as file:
        config = yaml.load(file)
    set_seed(args.seed)

    # check if gpu training is available
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    if not config['disable_cuda'] and torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        args.device = torch.device('cuda')
        cudnn.deterministic = True
        cudnn.benchmark = True

    print("Using device:", device)

    if config['arch'] == 'resnet18':
        model = torchvision.models.resnet18(pretrained=False, num_classes=5).to(device)
    elif config['arch'] == 'resnet50':
        model = torchvision.models.resnet50(pretrained=False, num_classes=5).to(device)


    checkpoint = torch.load(args.dir / 'checkpoint_0200.pth.tar', map_location=device)
    state_dict = checkpoint['state_dict']

    for k in list(state_dict.keys()):
        if k.startswith('backbone.'):
            if k.startswith('backbone') and not k.startswith('backbone.fc'):
                # remove prefix
                state_dict[k[len("backbone."):]] = state_dict[k]
        del state_dict[k]

    log = model.load_state_dict(state_dict, strict=False)
    assert log.missing_keys == ['fc.weight', 'fc.bias']

    if config['dataset_name'] == 'cifar10':
        train_loader, test_loader = get_cifar10_data_loaders(download=True)
    elif config['dataset_name'] == 'stl10':
        train_loader, test_loader = get_stl10_data_loaders(download=True)
    elif config['dataset_name'] == 'chexpert':
        train_loader, test_loader = get_chexpert_data_loaders(config, args.training_size)
    print("Dataset:", config['dataset_name'])

    # freeze all layers but the last fc
    for name, param in model.named_parameters():
        if name not in ['fc.weight', 'fc.bias']:
            param.requires_grad = False

    parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    assert len(parameters) == 2  # fc.weight, fc.bias

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=0.0008)
    criterion = torch.nn.BCEWithLogitsLoss().to(device)

    epochs = 200
    for epoch in range(epochs):
        # top1_train_accuracy = 0
        auc_, prc_ = 0, 0
        for counter, (x_batch, y_batch) in enumerate(train_loader):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            auc, prc = accuracy(logits, y_batch)
            auc_ += auc
            prc_ += prc
            # top1_train_accuracy += top1[0]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # top1_train_accuracy /= (counter + 1)
        auc_ /= (counter + 1)
        prc_ /= (counter + 1)
        # top1_accuracy = 0
        # top5_accuracy = 0
        auc_val, prc_val = 0, 0
        for counter, (x_batch, y_batch) in enumerate(test_loader):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(x_batch)
        
            auc, prc = accuracy(logits, y_batch)
            # top1_accuracy += top1[0]
            # top5_accuracy += top5[0]
            auc_val += auc 
            prc_val += prc
        
        # top1_accuracy /= (counter + 1)
        # top5_accuracy /= (counter + 1)
        auc_val /= (counter + 1)
        prc_val /= (counter + 1)
        print(f"Epoch {epoch}\tAUC {auc_.item():.5f}\tPRC: {prc_.item():.5f}\tAUC_val: {auc_val.item():.5f}\tPRC_val: {prc_val.item():.5f}")
    torch.save(model.state_dict(), args.dir / f"checkpoint_ft_{int(auc_val.item()*10000):04d}.pth.tar")