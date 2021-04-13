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
    def __init__(self, model, optimizer, train_loader, train_loader_unlabeled, val_loader, test_loader, 
                 scheduler, iterations, output_dir, teacher) -> None:
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.train_loader_u = train_loader_unlabeled
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = torch.device('cuda')

        # Self Training:
        self.teacher = teacher
        self.self_training = cfg.SOLVER.SELF_TRAINING
        cfg_trn_node = cfg.STUDENT if (self.self_training and not self.teacher) else cfg.TEACHER

        # Training Parameters
        self.batch_size = cfg.DATA.BATCH_SIZE
        self.num_epochs = cfg_trn_node.EPOCHS
        self.iterations_per_epoch = len(self.train_loader)
        self.max_iters = self.num_epochs * self.iterations_per_epoch
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.iterations = iterations
        
        # Training Hyperparameters
        self.beta_u = cfg_trn_node.BETA_U
        self.beta_l = cfg_trn_node.BETA_L
        self.beta_c = cfg_trn_node.BETA_C
        self.mixup_alpha = cfg.SOLVER.MIXUP_ALPHA

        # Logging
        self.train_recording_interval_per_epoch = 10
        self.train_recording_interval = self.iterations_per_epoch // self.train_recording_interval_per_epoch
        self.output_dir = Path(output_dir)
        self.writer = SummaryWriter(self.output_dir, flush_secs=60)


    def train(self):
        t = tqdm(range(self.max_iters), dynamic_ncols=True)
        t.update(self.iterations)
        for epoch in range(self.num_epochs):
            self.train_epoch(t)
            self.scheduler.step()
            if self.iterations >= self.max_iters:
                break
        t.close()
    
    def train_epoch(self, t):
        self.model.train()
        unlabeled_iterator = iter(self.train_loader_u)
        for batch_idx, (imgs, labels) in enumerate(self.train_loader):
            imgs, labels = imgs.to(self.device), labels.to(self.device)
            
            try:
                imgs_unlabeled = unlabeled_iterator.next().to(self.device)
            except StopIteration:
                unlabeled_iterator = iter(self.train_loader_u)
                imgs_unlabeled = unlabeled_iterator.next().to(self.device)

            self.optimizer.zero_grad(set_to_none=True)
            y = self.model(imgs)
            loss = self.bce_loss(y, labels)

            auc = self.get_auc(labels, y)
            prc = self.get_prc(labels, y)

            if self.mixup_alpha:
                # what is alpha --> see mixup reference
                # "additional samples" --> is there a regular non-mixup loss?
                # y_bar equation in paper is not used?

                # generate mixup parameter
                lambda_ = np.random.beta(self.mixup_alpha, self.mixup_alpha)

                inds1 = torch.arange(imgs.shape[0])
                inds2 = torch.randperm(imgs.shape[0])

                x_bar = lambda_ * imgs[inds1] + (1. - lambda_) * imgs[inds2]

                # forward pass
                y_bar = self.model(x_bar)
                
                loss_mixup = lambda_ * self.bce_loss(y_bar, labels[inds1]) + (1. - lambda_) * self.bce_loss(y_bar, labels[inds2])
                loss_mixup = loss_mixup.sum()

                if self.teacher and self.self_training:
                    loss = loss_mixup
                else:
                    loss += loss_mixup


            if self.self_training and not self.teacher:
                pass

            if self.beta_c:
                y_ = torch.sigmoid(y)
                pcs = torch.mean(y_, axis=0)
                rct = torch.log((0.35 / pcs) + (pcs / 0.75))
                loss += self.beta_c * torch.sum(rct)

            # backprop
            loss.backward()
            self.optimizer.step()
            self.iterations += 1
            
            if batch_idx % self.train_recording_interval == 0:
                global_step = int(self.iterations / self.iterations_per_epoch * self.train_recording_interval_per_epoch)
                self.writer.add_scalar("train/loss", loss.item(), global_step)
                self.writer.add_scalar("train/auc", auc, global_step)
                self.writer.add_scalar("train/prc", prc, global_step)

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
        self.validate(split='val')
        self.validate(split='test')

        # save model
        torch.save(self.model.state_dict(), self.output_dir / f'model-{self.iterations:05d}.pth')

    def validate(self, split='val'):
        assert split in ['val', 'test']

        self.model.eval()
        losses, aucs, prcs = [], [], []

        for batch_idx, (imgs, labels) in enumerate(tqdm(eval(f'self.{split}_loader'), position=1, leave=False)):
            imgs, labels = imgs.to(self.device), labels.to(self.device) 

            # forward pass
            with torch.no_grad():
                y = self.model(imgs)
                loss = self.bce_loss(y, labels)
                
            aucs.append(self.get_auc(labels, y))
            prcs.append(self.get_prc(labels, y))

            if self.mixup_alpha:
                # what is alpha --> see mixup reference
                # "additional samples" --> is there a regular non-mixup loss?
                # y_bar equation in paper is not used?

                # generate mixup parameter
                lambda_ = np.random.beta(self.mixup_alpha, self.mixup_alpha)

                inds1 = torch.arange(imgs.shape[0])
                inds2 = torch.randperm(imgs.shape[0])

                x_bar = lambda_ * imgs[inds1] + (1. - lambda_) * imgs[inds2]

                # forward pass
                y_bar = self.model(x_bar)
                
                loss_mixup = lambda_ * self.bce_loss(y_bar, labels[inds1]) + (1. - lambda_) * self.bce_loss(y_bar, labels[inds2])
                loss_mixup = loss_mixup.sum()

                loss += loss_mixup

            losses.append(loss.item())
        
        self.writer.add_scalar(f"{split}/loss", np.mean(losses), self.iterations // self.iterations_per_epoch)
        self.writer.add_scalar(f"{split}/auc", np.nanmean(aucs), self.iterations // self.iterations_per_epoch)
        self.writer.add_scalar(f"{split}/prc", np.mean(prcs), self.iterations // self.iterations_per_epoch)
                
    def get_auc(self, labels, y):
        try:
            return metrics.roc_auc_score(labels.cpu().numpy(), y.detach().cpu().numpy(), average='weighted')
        except ValueError:
            return np.nan

    def get_prc(self, labels, y):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            return metrics.average_precision_score(labels.cpu().numpy(), y.detach().cpu().numpy(), average='weighted')
