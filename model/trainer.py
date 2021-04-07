import numpy as np
from tqdm import tqdm
from pathlib import Path

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

        self.device = torch.device(cfg.DATA.DEVICE)
        self.batch_size = cfg.DATA.BATCH_SIZE
        self.iterations = iterations
        self.max_iters = cfg.SOLVER.NUM_ITERS
        self.val_interval = cfg.SOLVER.VAL_INTERVAL
        self.output_dir = Path(output_dir)

        self.writer = SummaryWriter(self.output_dir, flush_secs=60)

    def train(self):
        t = tqdm(range(1, self.max_iters + 1))
        t.update(self.iterations)
        for epoch in t:
            self.train_epoch()
            self.scheduler.step()
    
    def train_epoch(self):
        self.model.train()

        print_every, val_every = 100, 300
        train_loss, train_acc = 0, 0    
        cross_entropy_loss = nn.CrossEntropyLoss()
        
        labels_, ys, train_loss, train_acc = [], [], [], []
        if self.mixup:
            ys_mixup, train_loss_mixup, train_acc_mixup = [], [], []
            
        for batch_idx, (imgs, labels) in enumerate(tqdm(self.train_loader, position=1, leave=False)):
            imgs, labels = imgs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad(set_to_none=True)

            y = self.model(imgs)
            loss = cross_entropy_loss(y, labels)

            # accuracy
            pred = y.argmax(dim=1, keepdim=True) 
            
            labels_.append(labels.detach().cpu().numpy())
            ys.append((F.softmax(y, dim=1)).detach().cpu().numpy())
            train_loss += [loss.item()]
            train_acc += [pred.eq(labels.view_as(pred)).float().mean().item()]

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
                
                bce_loss = nn.BCELoss()
                loss_mixup = lambda_ * bce_loss(y_bar, labels[inds1]) + (1. - lambda_) * bce_loss(y_bar, labels[inds2])
                loss_mixup = loss_mixup.sum()

                loss += loss_mixup

                train_loss_mixup += [loss_mixup.item()]
                pred_mixup = y_bar.argmax(dim=1, keepdim=True)
                ys_mixup.append((F.softmax(y_bar, dim=1)).detach().cpu().numpy())
                train_acc_mixup += [pred_mixup.eq(labels.view_as(pred_mixup)).float().mean().item()]

            # backprop
            loss.backward()
            self.optimizer.step()
            self.iterations += 1
                
            if batch_idx % print_every == 0 and batch_idx != 0:
                # log(train_writer, optimizer, iteration, train_loss, train_acc)
                # log_aps(train_writer, iteration, acts, ys)
                            
                train_loss, train_acc = [], []
                labels_, ys = [], []           
                if self.mixup_alpha:
                    train_loss_mixup, train_acc_mixup = [], []
                    ys_mixup = []
        
        # validation
        if self.mixup_alpha: 
            val_act_loss, val_act_acc, val_labels, val_ys, \
                    val_loss_mixup, val_acc_mixup, ys_mixup = self.validate()
        else:
            val_act_loss, val_act_acc, val_labels, val_ys = self.validate()

        # log(val_writer, optimizer, iteration, val_loss, val_act_acc)
        # log_aps(val_writer, iteration, val_labels, val_ys)
                
        # save model
        torch.save(self.model.state_dict(), self.output_dir / 'model-{:d}.pth'.format(self.iteration))
        

    def validate(self):
        self.model.eval()

        val_loss, val_acc, j = 0, 0, 0
        cross_entropy_loss = nn.CrossEntropyLoss()
        
        labels_, ys, val_loss, val_acc = [], [], [], []
        if self.mixup:
            ys_mixup, val_loss, mixup = [], [], []
        
        with torch.no_grad():
            for batch_idx, (imgs, labels) in enumerate(self.val_loader):
                imgs, labels = imgs.to(self.device), labels.to(self.device) 

                # forward pass
                y = self.model(imgs)
                loss = cross_entropy_loss(y, labels)
                
                # accuracy
                pred = y.argmax(dim=1, keepdim=True)        # get the index of the max log-probability

                labels_.append(labels.detach().cpu().numpy())
                ys.append((F.softmax(y, dim=1)).detach().cpu().numpy())
                val_loss += [loss.item()]
                val_acc += [pred.eq(a_tm1.view_as(pred)).float().mean().item()]

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
                    
                    bce_loss = nn.BCELoss()
                    loss_mixup = lambda_ * bce_loss(y_bar, labels[inds1]) + (1. - lambda_) * bce_loss(y_bar, labels[inds2])
                    loss_mixup = loss_mixup.sum()

                    loss += loss_mixup

                    val_loss_mixup += [loss_mixup.item()]
                    pred_mixup = y_bar.argmax(dim=1, keepdim=True)
                    ys_mixup.append((F.softmax(y_bar, dim=1)).detach().cpu().numpy())
                    val_acc_mixup += [pred_mixup.eq(labels.view_as(pred_mixup)).float().mean().item()]

                j += 1
                if j == print_every:
                    break
                
        if self.mixup_alpha:
            return val_loss, val_acc, labels_, ys, val_loss_mixup, val_acc_mixup, ys_mixup
        else:
            return val_loss, val_acc, labels_, ys