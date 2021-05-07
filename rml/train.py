import os, sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch 
from torchvision import transforms
import argparse
from PIL import Image

from net import RML
from robust_warp_loss import RobustWarpLoss
from dataloader import build_dataloader
from tqdm import tqdm

from utils.config import _C as cfg
from utils.transforms import get_transforms

def train():
    model = RML()
    model.train()

    tr_labeled, tr_unlabeled = build_dataloader("train", model)

    optimizer = torch.optim.SGD(model.parameters(), lr=cfg.DATA.BASE_LR, momentum=0.9, weight_decay=cfg.DATA.weight_decay)
    criterion = RobustWarpLoss(cfg.DATA.LABELED_SIZE, cfg.DATA.UNLABELED_SIZE)
    
    t = tqdm(range(80))
    for epoch in t: # 80 epochs, hard coded for now
        running_loss = 0.0
        if epoch % 20 == 19: # reinitialize dummy labels
            _, tr_unlabeled = build_dataloader("train", model)
        for i, (ims, labels) in enumerate(tr_labeled):
            ims, labels = ims.cuda(), labels.cuda()
            optimizer.zero_grad()
            outputs = model(ims)
            loss = RobustWarpLoss(outputs, labels, tr_unlabeled.S)
            loss.backward()
            optimizer.step()
            postfix_dict = {"Loss": loss.item()}
            t.set_postfix(postfix_dict)
    t.close()


parser = argparse.ArgumentParser(description='Train a model')
parser.add_argument('--config', type=str, default='./config/rml_base.yaml')
args = parser.parse_args()

cfg.merge_from_file(args.config)
cfg.freeze()

