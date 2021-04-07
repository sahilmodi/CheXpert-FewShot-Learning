import warnings
import numpy as np
from tqdm import tqdm
from pathlib import Path
import sklearn.metrics as metrics

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from utils.config import _C as cfg

class Trainer():
    def __init__(self, model, optimizer, train_loader, val_loader, scheduler, output_dir, iterations) -> None:
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.mixup_alpha = cfg.SOLVER.MIXUP_ALPHA

        self.device = torch.device('cuda')
        self.batch_size = cfg.DATA.BATCH_SIZE
        self.iterations = iterations
        self.max_iters = cfg.SOLVER.NUM_ITERS
        self.val_interval = cfg.SOLVER.VAL_INTERVAL
        self.output_dir = Path(output_dir)

        self.writer = SummaryWriter(self.output_dir, flush_secs=60)

        self.loss_fn = nn.BCEWithLogitsLoss()
        self.bce_loss = nn.BCELoss()

    def train(self):
        t = tqdm(range(self.max_iters))
        t.update(self.iterations)
        for epoch in t:
            self.train_epoch(t)
            self.scheduler.step()
        t.close()
    
    def train_epoch(self, t):
        self.model.train()

        for batch_idx, (imgs, labels) in enumerate(self.train_loader):
            imgs, labels = imgs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad(set_to_none=True)

            y = self.model(imgs)
            loss = self.loss_fn(y, labels)

            auc = self.get_auc(labels, y)
            prc = self.get_prc(labels, y)

            if self.mixup_alpha:
                # what is alpha --> see mixup reference
                # "additional samples" --> is there a regular non-mixup loss?
                # y_bar equation in paper is not used?

                # generate mixup parameter
                lambda_ = np.random.beta(self.mixup_alpha, self.mixup_alpha)

                inds1 = torch.arange(self.batch_size)
                inds2 = torch.randperm(self.batch_size)

                x_bar = lambda_ * imgs[inds1] + (1. - lambda_) * imgs[inds2]

                # forward pass
                y_bar = self.model(x_bar)
                
                loss_mixup = lambda_ * self.bce_loss(y_bar, labels[inds1]) + (1. - lambda_) * self.bce_loss(y_bar, labels[inds2])
                loss_mixup = loss_mixup.sum()

                loss += loss_mixup

            # backprop
            loss.backward()
            self.optimizer.step()
            self.iterations += 1
                
            if batch_idx % 10 == 0 and batch_idx != 0:
                self.writer.add_scalar("train/loss", loss.item(), self.iterations)
                self.writer.add_scalar("train/auc", auc, self.iterations)
                self.writer.add_scalar("train/prc", prc, self.iterations)
            
            postfix_map = {
                "loss": loss.item(), 
                "W-AUC": auc,
                "W-PRC": prc
            }
            t.set_postfix(postfix_map, refresh=False)
            t.update()
            if self.iterations >= self.max_iters:
                break
        
        # validation
        self.validate()

        # save model
        torch.save(self.model.state_dict(), self.output_dir / 'model-{:d}.pth'.format(self.iteration))
        

    def validate(self):
        self.model.eval()
        losses, aucs, prcs = [], [], []

        for batch_idx, (imgs, labels) in enumerate(tqdm(self.val_loader, position=1, leave=False)):
            imgs, labels = imgs.to(self.device), labels.to(self.device) 

            # forward pass
            with torch.no_grad():
                y = self.model(imgs)
                loss = self.loss_fn(y, labels)
                
            aucs.append(self.get_auc(labels, y))
            prcs.append(self.get_prc(labels, y))

            if self.mixup_alpha:
                # what is alpha --> see mixup reference
                # "additional samples" --> is there a regular non-mixup loss?
                # y_bar equation in paper is not used?

                # generate mixup parameter
                lambda_ = np.random.beta(self.mixup_alpha, self.mixup_alpha)

                inds1 = torch.arange(self.batch_size)
                inds2 = torch.randperm(self.batch_size)

                x_bar = lambda_ * imgs[inds1] + (1. - lambda_) * imgs[inds2]

                # forward pass
                y_bar = self.model(x_bar)
                
                loss_mixup = lambda_ * self.bce_loss(y_bar, labels[inds1]) + (1. - lambda_) * self.bce_loss(y_bar, labels[inds2])
                loss_mixup = loss_mixup.sum()

                loss += loss_mixup

            losses.append(loss.item())
        self.writer.add_scalar("val/loss", np.mean(losses), self.iterations)
        self.writer.add_scalar("val/auc", np.mean(aucs), self.iterations)
        self.writer.add_scalar("val/prc", np.mean(prcs), self.iterations)
                
    def get_auc(self, labels, y):
        try:
            return metrics.roc_auc_score(labels.cpu().numpy(), y.detach().cpu().numpy(), average='weighted')
        except ValueError:
            return np.nan

    def get_prc(self, labels, y):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            return metrics.average_precision_score(labels.cpu().numpy(), y.detach().cpu().numpy(), average='weighted')