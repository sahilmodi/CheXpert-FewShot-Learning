import warnings
import numpy as np
from tqdm import tqdm
from pathlib import Path
import sklearn.metrics as metrics

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from utils.config import _C as cfg

class Trainer():
    def __init__(self, model, optimizer, train_loader, train_loader_unlabeled, val_loader, test_loader, 
                 scheduler, output_dir, teacher_model=None) -> None:
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.train_loader_u = train_loader_unlabeled
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = torch.device('cuda')

        # Self Training:
        self.teacher_model = teacher_model
        self.self_training = cfg.SOLVER.SELF_TRAINING
        self.student = self.self_training and self.teacher_model is not None
        self.teacher = not self.student
        self.mode = 'teacher' if self.teacher else 'student'
        cfg_trn_node = cfg.STUDENT if (self.self_training and not self.teacher) else cfg.TEACHER
        
        # Training Parameters
        self.batch_size = cfg.DATA.BATCH_SIZE
        self.num_epochs = cfg_trn_node.EPOCHS
        self.iterations_per_epoch = len(self.train_loader)
        self.max_iters = self.num_epochs * self.iterations_per_epoch
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.kldiv_loss = nn.KLDivLoss(reduction='batchmean')
        self.iterations = 0
        
        # Training Hyperparameters
        self.beta_l = cfg_trn_node.BETA_L
        self.beta_u = cfg_trn_node.BETA_U
        self.beta_c = cfg_trn_node.BETA_C
        self.mixup_alpha = cfg.SOLVER.MIXUP_ALPHA

        # Logging
        self.train_recording_interval_per_epoch = 10
        self.train_recording_interval = self.iterations_per_epoch // self.train_recording_interval_per_epoch
        self.output_dir = Path(output_dir)
        self.writer = SummaryWriter(self.output_dir / self.mode, flush_secs=60)

        # Parameter checks
        if self.self_training:
            assert self.mixup_alpha, "Self-Training can only be run with a non-zero mixup alpha."
        if self.student:
            self.teacher_model.eval()


    def train(self):
        print(f"Training the {self.mode}...")
        t = tqdm(range(self.max_iters), dynamic_ncols=True)
        for epoch in range(self.num_epochs):
            self.train_epoch(t, epoch)
            self.scheduler.step()
            if self.iterations >= self.max_iters:
                break
        t.close()
        self.validate(split='test')
        self.writer.flush()
    
    def train_epoch(self, t, epoch):
        self.model.train()
        unlabeled_iterator = iter(self.train_loader_u)
        for batch_idx, (imgs, labels) in enumerate(self.train_loader):
            imgs, labels = imgs.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad(set_to_none=True)
            y = self.model(imgs)
            loss = self.bce_loss(y, labels)

            auc = self.get_auc(labels, y)
            prc = self.get_prc(labels, y)

            if self.student:
                gamma = 0.5
                # try:
                #     imgs_unlabeled = unlabeled_iterator.next().to(self.device)
                # except StopIteration:
                #     unlabeled_iterator = iter(self.train_loader_u)
                #     imgs_unlabeled = unlabeled_iterator.next().to(self.device)
                # with torch.no_grad():
                    # yhat_u = self.teacher_model(imgs_unlabeled).sigmoid()
                yhat_u = []
                imgs_unlabeled = unlabeled_iterator.next().to(self.device)
                with torch.no_grad():
                    for b in range(0, imgs_unlabeled.shape[0], imgs.shape[0]):
                        minibatch = imgs_unlabeled[b:b+imgs.shape[0]]
                        out = self.teacher_model(minibatch)
                        yhat_u.append(out)
                yhat_u = torch.cat(yhat_u, dim=0)
                yhat_u = (1-gamma)*yhat_u + gamma*torch.sigmoid(1e8 * (yhat_u - 0.5))
                yhat_u /= yhat_u.sum(dim=1, keepdim=True)

            if self.mixup_alpha:
                # what is alpha --> see mixup reference
                # "additional samples" --> is there a regular non-mixup loss?
                # y_bar equation in paper is not used?

                # generate mixup parameter
                lambda_ = np.random.beta(self.mixup_alpha, self.mixup_alpha)

                inds1 = torch.arange(imgs.shape[0])
                inds2 = torch.randperm(imgs.shape[0])

                x_bar = imgs if self.teacher else imgs_unlabeled
                labels_ = labels if self.teacher else yhat_u

                x_tilde = lambda_ * x_bar[inds1] + (1. - lambda_) * x_bar[inds2]

                # forward pass
                y_bar = self.model(x_tilde)
                
                loss_fn = self.bce_loss
                if self.student:
                    loss_fn = self.kldiv_loss
                    y_bar = y_bar.sigmoid()
                    y_bar = torch.log(y_bar / y_bar.sum(dim=1, keepdim=True))

                loss_mixup = lambda_ * loss_fn(y_bar, labels_[inds1]) + (1. - lambda_) * loss_fn(y_bar, labels_[inds2])
                loss_mixup = loss_mixup.sum()
                if loss_mixup < 0:
                    print("NEGATIVE LOSS")
                    print(labels_[0])
                    print(labels_[0].sum())
                    print(y_bar[0])
                    print(loss_mixup)
                    exit()

                if self.teacher and self.self_training:
                    loss = loss_mixup
                else:
                    loss = self.beta_l*loss + self.beta_u*loss_mixup

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
        
        # self.validate(split='val')

        # save model
        torch.save(self.model.state_dict(), self.output_dir / f'model-{self.iterations:05d}_{self.mode}.pth')

    def validate(self, split='val'):
        assert split in ['val', 'test']

        self.model.eval()
        losses, aucs, prcs = [], [], []

        for batch_idx, (imgs, labels) in enumerate(tqdm(eval(f'self.{split}_loader'), position=1, leave=False, dynamic_ncols=True)):
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
                # Unlabeled data not necessary for validation. Accordingly, L_dist is not applicable, 
                # so just use regular mixup loss for logging purposes.

                # generate mixup parameter
                lambda_ = np.random.beta(self.mixup_alpha, self.mixup_alpha)

                inds1 = torch.arange(imgs.shape[0])
                inds2 = torch.randperm(imgs.shape[0])

                x_bar = imgs
                labels_ = labels

                x_tilde = lambda_ * x_bar[inds1] + (1. - lambda_) * x_bar[inds2]

                # forward pass
                y_bar = self.model(x_tilde)
               
                loss_fn = self.bce_loss
                loss_mixup = lambda_ * loss_fn(y_bar, labels_[inds1]) + (1. - lambda_) * loss_fn(y_bar, labels_[inds2])
                loss_mixup = loss_mixup.sum()

                loss = self.beta_l*loss + self.beta_u*loss_mixup

            if self.beta_c:
                y_ = torch.sigmoid(y)
                pcs = torch.mean(y_, axis=0)
                rct = torch.log((0.35 / pcs) + (pcs / 0.75))
                loss += self.beta_c * torch.sum(rct)

            losses.append(loss.item())
        
        self.writer.add_scalar(f"{split}/loss", np.mean(losses), self.iterations // self.iterations_per_epoch)
        self.writer.add_scalar(f"{split}/auc", np.nanmean(aucs), self.iterations // self.iterations_per_epoch)
        self.writer.add_scalar(f"{split}/prc", np.mean(prcs), self.iterations // self.iterations_per_epoch)

        if split == 'test':
            print(f"\nLoss: {np.mean(losses):.3f} | W-AUC: {np.nanmean(aucs):.3f} | W-PRC: {np.mean(prcs):.3f}\n")
                
    def get_auc(self, labels, y):
        try:
            return metrics.roc_auc_score(labels.cpu().numpy(), y.detach().cpu().numpy(), average='weighted')
        except ValueError:
            return np.nan

    def get_prc(self, labels, y):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            return metrics.average_precision_score(labels.cpu().numpy(), y.detach().cpu().numpy(), average='weighted')
