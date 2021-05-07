import os, sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch 
import argparse

from net import RML
from robust_warp_loss import RobustWarpLoss
from dataloader import build_dataloader

from utils.config import _C as cfg
from utils.transforms import get_transforms

# def train(cfg, train_loader):
#     model = RML()
#     model.train()
    
#     for epoch in range(50): # 50 epochs, hard coded for now
#         running_loss = 0.0
#         for i, (ims, labels) in enumerate(train_loader):
#             ims, labels = ims.cuda(), labels.cuda()
#             optimizer.zero_grad()

parser = argparse.ArgumentParser(description='Train a model')
parser.add_argument('--config', type=str, default='./config/rml_base.yaml')
args = parser.parse_args()

cfg.merge_from_file(args.config)
cfg.freeze()

model = RML()
tr_labeled, tr_unlabeled = build_dataloader("train", model)