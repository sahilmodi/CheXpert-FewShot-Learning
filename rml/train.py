import os, sys
from pathlib import Path
sys.path.append()
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

tr_labeled, tr_unlabeled = build_dataloader("train")