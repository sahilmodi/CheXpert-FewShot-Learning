import os, sys
from pathlib import Path
sys.path.append(str(Path(__file__).absolute().parent.parent))
import yaml
import copy
import argparse
import warnings
import numpy as np
import sklearn.metrics as metrics

import torch
import torchvision
from torchvision import datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import torch.nn as nn

from utils.misc import set_seed
from simclr.data_aug.chexpert_dataset import ChexpertDatasetFinetune

def get_chexpert_data_loaders(config, training_data_size):
    train_dataset = ChexpertDatasetFinetune(config['data'], 224, num=training_data_size)
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], num_workers=12, drop_last=False, shuffle=True)
    
    val_dataset = ChexpertDatasetFinetune(config['data'], 224, split='valid')
    val_loader = DataLoader(train_dataset, batch_size=config['batch_size'], num_workers=12, drop_last=False)
    
    test_dataset = ChexpertDatasetFinetune(config['data'], 224, split="test")
    test_loader = DataLoader(test_dataset, batch_size=2*config['batch_size'], num_workers=12, drop_last=False)
    return train_loader, val_loader, test_loader

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
    ap.add_argument("-n", "--name", type=str, required=True, help="Name of the run.")
    ap.add_argument('-e', "--epochs", type=int, default=200, help="Number of epochs to finetune.")
    ap.add_argument('-g', "--gpu", type=int, help="Which gpu to use.")
    ap.add_argument('-tds', "--training_size", type=int, default=1000, help="How much training data to use.")
    ap.add_argument('-s', "--seed", type=int, default=0, help="Set random seed.")
    ap.add_argument('-m', "--mixup", action='store_true', help="Use mixup training.")
    ap.add_argument('-c', "--confidence_tempering", action='store_true', help="Use confidence tampering.")
    ap.add_argument('-st', "--self_training", type=str, help="Path to teacher model for model distillation.")
    return ap.parse_args()


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args = parse_args()
    with open(os.path.join(args.dir / 'config.yml')) as file:
        config = yaml.safe_load(file)
    set_seed(args.seed)
    output_dir = args.dir / args.name
    output_dir.mkdir(exist_ok=True)

    # check if gpu training is available
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    if not config['disable_cuda'] and torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        args.device = torch.device('cuda')
        cudnn.deterministic = True
        cudnn.benchmark = True

    print("Using device:", torch.cuda.get_device_name())

    if config['arch'] == 'resnet18':
        model = torchvision.models.resnet18(pretrained=False, num_classes=5).to(device)
    elif config['arch'] == 'resnet50':
        model = torchvision.models.resnet50(pretrained=False, num_classes=5).to(device)

    # mixup parameters
    if args.training_size == 1000:
        mixup_alpha = 0.6
        beta_c = 0.2
    elif args.training_size == 12500:
        mixup_alpha = 0.3
        beta_c = 0.2
    elif args.training_size == 20000:
        mixup_alpha = 0.2
        beta_c = 0.1

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

    if config['dataset_name'] == 'chexpert':
        train_loader, val_loader, test_loader = get_chexpert_data_loaders(config, args.training_size)
    print("Dataset:", config['dataset_name'])

    # freeze all layers but the last fc
    if args.self_training:
        teacher_state_dict = torch.load(args.self_training)
        teacher_model = copy.deepcopy(model)
        teacher_model.load_state_dict(teacher_state_dict)
        teacher_model.eval()
    else:
        for name, param in model.named_parameters():
            if name not in ['fc.weight', 'fc.bias']:
                param.requires_grad = False
        parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
        assert len(parameters) == 2  # fc.weight, fc.bias

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=0.0008)
    criterion = torch.nn.BCEWithLogitsLoss().to(device)

    best_auc, best_epoch = 0, 0
    for epoch in range(args.epochs):
        model.train()
        auc_, prc_ = 0, 0
        for counter, (x_batch, y_batch) in enumerate(train_loader):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            if args.self_training:
                y_batch_t = torch.sigmoid(teacher_model(x_batch))
                # y_batch_t = 0.5*y_batch + 0.5*torch.sigmoid(1e8 * (y_batch_t - 0.5))

            # print(y_batch_t[0])
            # print(y_batch[0])
            # exit()

            logits = model(x_batch)
            loss = criterion(logits, y_batch if not args.self_training else y_batch_t)
            auc, prc = accuracy(logits, y_batch)
            auc_ += auc
            prc_ += prc

            if args.mixup:
                x_bar = x_batch 
                labels_ = y_batch 

                # generate mixup parameter
                lambda_ = np.random.beta(mixup_alpha, mixup_alpha)

                inds1 = torch.arange(x_bar.shape[0])
                inds2 = torch.randperm(x_bar.shape[0])

                x_tilde = lambda_ * x_bar[inds1] + (1. - lambda_) * x_bar[inds2]

                # forward pass
                y_bar = model(x_tilde)
                
                loss_fn = nn.BCEWithLogitsLoss() 
                loss_mixup = lambda_ * loss_fn(y_bar, labels_[inds1]) + (1. - lambda_) * loss_fn(y_bar, labels_[inds2])
                loss_mixup = loss_mixup.sum()
                
                loss = loss + loss_mixup
            
            if args.confidence_tempering:
                y_ = torch.sigmoid(logits)
                pcs = torch.mean(y_, axis=0)
                rct = torch.log((0.35 / pcs) + (pcs / 0.75))
                loss += args.beta_c * torch.sum(rct)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        auc_ /= (counter + 1)
        prc_ /= (counter + 1)
        
        # Validate
        model.eval()
        auc_val, prc_val = 0, 0
        for counter, (x_batch, y_batch) in enumerate(val_loader):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(x_batch)
        
            auc, prc = accuracy(logits, y_batch)
            auc_val += auc 
            prc_val += prc

        auc_val /= (counter + 1)
        prc_val /= (counter + 1)
        print(f"Epoch {epoch}\tAUC {auc_.item():.5f}\tPRC: {prc_.item():.5f}\tAUC_val: {auc_val.item():.5f}\tPRC_val: {prc_val.item():.5f}")
        if auc_val > best_auc:
            best_auc = auc_val
            best_epoch = epoch
            torch.save(model.state_dict(), output_dir / f"model_best.pth.tar")


    # Test
    model.load_state_dict(torch.load(output_dir / "model_best.pth.tar"))
    print(f"- Loaded model from epoch {best_epoch}")
    args.best_epoch = best_epoch
    model.eval()
    auc_tst, prc_tst = 0, 0
    for counter, (x_batch, y_batch) in enumerate(test_loader):
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        logits = model(x_batch)
    
        auc, prc = accuracy(logits, y_batch)
        auc_tst += auc 
        prc_tst += prc
    
    auc_tst /= (counter + 1)
    prc_tst /= (counter + 1)
    results = f"[TEST]: AUC_tst: {auc_tst.item():.5f}\tPRC_tst: {prc_tst.item():.5f}"
    print(results)

    # Save checkpoint at the end
    with open(output_dir / f"finetuning.yml", 'w') as f:
        cfg = {}
        for k, v in args.__dict__.items():
            if not isinstance(v, (int, str, bool, float)):
                v = str(v)
            cfg[k] = v
        cfg["Test Results"] = results
        yaml.dump(cfg, f, default_flow_style=False)
