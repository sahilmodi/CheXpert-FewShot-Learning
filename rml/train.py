import os, sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
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
from utils.metrics import get_auc, get_prc


def train():
    model = RML()
    # model.load_state_dict(torch.load("./rml_1000_final.pth"))
    model.cuda()

    model.eval()
    tr_labeled, tr_unlabeled = build_dataloader("train", model)
    val_loader, _ = build_dataloader("val", None)

    optimizer = torch.optim.SGD(model.parameters(), lr=cfg.SOLVER.BASE_LR, momentum=0.9, weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    criterion = RobustWarpLoss(cfg.DATA.BATCH_SIZE, cfg.DATA.BATCH_SIZE)
    
    t = tqdm(range(80))
    for epoch in t: # 80 epochs, hard coded for now
        running_loss = 0.0
        if epoch % 20 == 19: # reinitialize dummy labels
            model.eval()
            _, tr_unlabeled = build_dataloader("train", model)
        model.train()
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
        
        if epoch % 10 == 9:
            model.eval()
            auc_ = []
            prc_ = []
            for ims, labels in val_loader:
                ims, labels = ims.cuda(), labels.cuda()
                with torch.no_grad():
                    outputs = model(ims)
                auc_.append(get_auc(labels, outputs))
                prc_.append(get_prc(labels, outputs))
            print("EVAL")
            print(f"- [{epoch}]: AUC: {np.nanmean(auc_):0.4f} | PRC: {np.nanmean(prc_):0.4f}")
            torch.save(model.state_dict(), f"rml_{cfg.DATA.LABELED_SIZE}_{epoch:02d}.pth")
            # exit()
            
    t.close()

    torch.save(model.state_dict(), f"rml_{cfg.DATA.LABELED_SIZE}_final.pth")


parser = argparse.ArgumentParser(description='Train a model')
parser.add_argument('--config', type=str, default='./config/rml_base.yaml')
parser.add_argument('--gpu', type=int, default=0)
args = parser.parse_args()

cfg.merge_from_file(args.config)
cfg.freeze()

torch.cuda.set_device(args.gpu)

train()