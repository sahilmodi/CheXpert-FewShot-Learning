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
    model.cuda()
    model.train()

    tr_labeled, tr_unlabeled = build_dataloader("train", model)

    optimizer = torch.optim.SGD(model.parameters(), lr=cfg.SOLVER.BASE_LR, momentum=0.9, weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    criterion = RobustWarpLoss(cfg.DATA.LABELED_SIZE, cfg.DATA.UNLABELED_SIZE)
    
    t = tqdm(range(80))
    for epoch in t: # 80 epochs, hard coded for now
        running_loss = 0.0
        if epoch % 20 == 19: # reinitialize dummy labels
            _, tr_unlabeled = build_dataloader("train", model)
        for i, (labeled, unlabeled) in enumerate(zip(tr_labeled, tr_unlabeled)):
            labeled_ims, labeled_targets = labeled
            unlabeled_ims, unlabeled_targets, s = unlabeled

            labeled_ims, labeled_targets = labeled_ims.cuda(), labeled_targets.cuda()
            unlabeled_ims, unlabeled_targets = unlabeled_ims.cuda(), unlabeled_targets.cuda()

            optimizer.zero_grad()
            labeled_outputs = model(labeled_ims)
            unlabeled_outputs = model(unlabeled_ims)

            loss = criterion(torch.cat((labeled_outputs, unlabeled_outputs)), torch.cat((labeled_targets, unlabeled_targets)), s)
            loss.backward()

            optimizer.step()

            postfix_dict = {"Loss": loss.item()}
            t.set_postfix(postfix_dict)
    t.close()

    torch.save(model.state_dict(), f"rml_{cfg.DATA.LABELED_SIZE}.pth")


parser = argparse.ArgumentParser(description='Train a model')
parser.add_argument('--config', type=str, default='./config/rml_base.yaml')
args = parser.parse_args()

cfg.merge_from_file(args.config)
cfg.freeze()

train()