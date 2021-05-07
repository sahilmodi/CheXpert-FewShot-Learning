import os, sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch.backends.cudnn as cudnn

import logging
import pickle
import argparse
import numpy as np

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, SVHN
from dataset.imbalance_chexpert import ChexpertDataset
from torchvision import transforms
import models
import random

from utils.long_tail_config import _C as cfg


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'

set_seed(543)

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))


parser = argparse.ArgumentParser(description='Apply standard trained model to generate labels on unlabeled data')
parser.add_argument('--dataset', default='chexpert', choices=['cifar10', 'svhn', 'chexpert'])
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet32', choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names))
parser.add_argument('--loss_type', default="CE", type=str, choices=['CE', 'Focal', 'LDAM'])
parser.add_argument('--gpu', default=0, type=int, help='GPU id to use')
# load trained models
parser.add_argument('--resume', type=str, default='')
# data related
parser.add_argument('--data_dir', default='./data', type=str,
                    help='directory that has unlabeled data')
parser.add_argument('--data_filename', default='ti_80M_selected.pickle', type=str)
parser.add_argument('--output_dir', default='./data', type=str)
parser.add_argument('--output_filename', default='pseudo_labeled_chexpert.pickle', type=str)
parser.add_argument('--config', type=str, default='./config/longtail_base.yaml')


args = parser.parse_args()

cfg.merge_from_file(args.config)
cfg.freeze()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(args.data_dir, 'prediction.log')),
        logging.StreamHandler()
    ])
logger = logging.getLogger()

logging.info('Prediction on unlabeled data')
logging.info('Args: %s', args)


# Loading unlabeled data
if args.dataset == 'cifar10':
    with open(os.path.join(args.data_dir, args.data_filename), 'rb') as f:
        data = pickle.load(f)

# Loading model
print(f"===> Creating model '{args.arch}'")
num_classes = 10 if args.dataset != "chexpert" else 32
use_norm = True if args.loss_type == 'LDAM' else False
model = models.__dict__[args.arch](num_classes=num_classes, use_norm=use_norm)

if args.gpu is not None:
    torch.cuda.set_device(args.gpu)
    model = model.cuda()
else:
    model = torch.nn.DataParallel(model).cuda()

assert args.resume is not None
if os.path.isfile(args.resume):
    print(f"===> Loading checkpoint '{args.resume}'")
    checkpoint = torch.load(args.resume, map_location=torch.device(f'cuda:{str(args.gpu)}'))
    print(checkpoint)
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in checkpoint['state_dict'].items():
        if 'linear' in k:
            new_state_dict[k.replace('linear', 'fc')] = v
        else:
            new_state_dict[k] = v
    model.load_state_dict(new_state_dict)
    print(f'===> Checkpoint weights found in total: [{len(list(new_state_dict.keys()))}]')
else:
    # train base classifier for CheXpert
    ds_path = Path(cfg.DATA.PATH)
    bs = cfg.DATA.BATCH_SIZE
    dataset = ChexpertDataset(ds_path / "train.csv", "train", labeled=True)
    dataloader = DataLoader(dataset, batch_size=bs, num_workers=min(os.cpu_count(), 12), shuffle=True)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), weight_decay=1e-3)
    criterion = torch.nn.CrossEntropyLoss()

    print("Started Training")
    for epoch in range(5):
        running_loss = 0.0
        for i, (ims, labels) in enumerate(dataloader):
            ims, labels = ims.cuda(), labels.cuda()
            optimizer.zero_grad()
            outputs = model(ims)
            loss = criterion(outputs, torch.max(labels.type(torch.cuda.LongTensor), dim=1)[1])
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 100 == 99:    # print every 100 mini-batches
                print('[Epoch: %d, Batch: %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0
    print("Finished Training")
    torch.save({'state_dict': model.state_dict()}, "base_clf_chexpert.pt")


            
    # raise ValueError(f"No checkpoint found at '{args.resume}'")

cudnn.benchmark = True

model.eval()

mean = [0.4914, 0.4822, 0.4465] if args.dataset.startswith('cifar') else [.5, .5, .5]
std = [0.2023, 0.1994, 0.2010] if args.dataset.startswith('cifar') else [.5, .5, .5]
transform_val = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

batch_size = 200
num_workers = 100
if args.dataset == 'cifar10':
    unlabeled_data = CIFAR10('./data', train=False, transform=transform_val)
    unlabeled_data.data = data['data']
    unlabeled_data.targets = list(data['extrapolated_targets'])
elif args.dataset == "svhn":
    unlabeled_data = SVHN('./data', split='extra', transform=transform_val)
elif args.dataset == "chexpert":
    ds_path = Path(cfg.DATA.PATH)
    unlabeled_data =ChexpertDataset(ds_path / "train.csv", "train", labeled=False)
    batch_size = 10
    num_workers = 2

data_loader = torch.utils.data.DataLoader(unlabeled_data,
                                          batch_size=batch_size,
                                          num_workers=num_workers,
                                          pin_memory=True)

# Running model on unlabeled data
predictions, truths = [], []
for i, (batch, targets) in enumerate(data_loader):
    _, preds = torch.max(model(batch.cuda()), dim=1)
    predictions.append(preds.cpu().numpy())
    if args.dataset == 'svhn':
        truths.append(targets.cpu().numpy())

    if (i+1) % 10 == 0:
        print('Done %d/%d' % (i+1, len(data_loader)))

new_extrapolated_targets = np.concatenate(predictions)
if args.dataset == 'svhn':
    ground_truth = np.concatenate(truths)
    new_targets = dict(extrapolated_targets=new_extrapolated_targets,
                       ground_truth=ground_truth,
                       prediction_model=args.resume)
else:
    new_targets = dict(extrapolated_targets=new_extrapolated_targets,
                       prediction_model=args.resume)

out_path = os.path.join(args.output_dir, args.output_filename)
assert(not os.path.exists(out_path))
with open(out_path, 'wb') as f:
    pickle.dump(new_targets, f)
