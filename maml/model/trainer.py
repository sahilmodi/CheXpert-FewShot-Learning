import warnings
import numpy as np
from tqdm import tqdm
from pathlib import Path
import sklearn.metrics as metrics

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from utils.config import _C as C
from maml.model.net import Net

import  torch
from    torch import nn
from    torch import optim
from    torch.nn import functional as F
from    torch.utils.data import TensorDataset, DataLoader
from    torch import optim
import  numpy as np

from    copy import deepcopy



class Trainer(nn.Module):
    def __init__(self, ):
        super(Trainer, self).__init__()

        self.update_lr = C.MAML.ALPHA
        self.meta_lr = C.MAML.BETA
        self.n_way = C.MAML.N_WAY
        self.k_shot = C.MAML.K_SHOT
        self.k_query = C.MAML.K_QUERY
        self.num_updates = C.MAML.N_INNER_UPDATES_TRN
        self.num_updates_test = C.MAML.N_INNER_UPDATES_TST

        self.net = Net()
        self.optim = optim.Adam(self.net.parameters(), lr=self.meta_lr)

    def forward(self, x_support, y_support, x_query, y_query):
        """
        :param x_support:   [b, setsz, c_, h, w]
        :param y_support:   [b, setsz]
        :param x_query:   [b, querysz, c_, h, w]
        :param y_query:   [b, querysz]
        :return:
        """
        task_num, setz, c_, h, w = x_support.shape
        querysz = x_query.shape[1]

        self.net.train()
        losses_q = [0 for _ in range(self.num_updates + 1)]  # losses_q[i] is the loss on step i
        corrects = [0 for _ in range(self.num_updates + 1)]

        for i in range(task_num):
            # 1. run the i-th task and compute loss for k=0
            logits = self.net(x_support[i])
            loss = F.cross_entropy(logits, y_support[i])
            grad = torch.autograd.grad(loss, self.net.parameters())
            fast_weights = list(map(lambda p: (p[1][0], p[1][1] - self.update_lr * p[0]), zip(grad, self.net.named_parameters())))

            # this is the loss and accuracy before first update
            with torch.no_grad():
                # [setsz, nway]
                logits_q = self.net(x_query[i])
                loss_q = F.cross_entropy(logits_q, y_query[i])
                losses_q[0] += loss_q

                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                correct = torch.eq(pred_q, y_query[i]).sum().item()
                corrects[0] = corrects[0] + correct

            # this is the loss and accuracy after the first update
            # [setsz, nway]
            logits_q = self.net(x_query[i], fast_weights)
            loss_q = F.cross_entropy(logits_q, y_query[i])
            losses_q[1] += loss_q
            # [setsz]
            with torch.no_grad():
                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                correct = torch.eq(pred_q, y_query[i]).sum().item()
                corrects[1] = corrects[1] + correct

            for k in range(1, self.num_updates):
                # 1. run the i-th task and compute loss for k=1~K-1
                logits = self.net(x_support[i], fast_weights)
                loss = F.cross_entropy(logits, y_support[i])
                # 2. compute grad on theta_pi
                grad = torch.autograd.grad(loss, [param for _, param in fast_weights])
                # 3. theta_pi = theta_pi - train_lr * grad
                fast_weights = list(map(
                    lambda p: (p[1][0], p[1][1] - self.update_lr * p[0]), 
                    zip(grad, fast_weights)
                ))

                logits_q = self.net(x_query[i], fast_weights)
                # loss_q will be overwritten and just keep the loss_q on last update step.
                loss_q = F.cross_entropy(logits_q, y_query[i])
                losses_q[k + 1] += loss_q

                with torch.no_grad():
                    pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                    correct = torch.eq(pred_q, y_query[i]).sum().item()  # convert to numpy
                    corrects[k + 1] = corrects[k + 1] + correct



        # end of all tasks
        # sum over all losses on query set across all tasks
        loss_q = losses_q[-1] / task_num

        # optimize theta parameters
        self.optim.zero_grad(set_to_none=True)
        loss_q.backward()
        self.optim.step()

        accs = np.array(corrects) / (querysz * task_num)
        # if accs[1] == 1:
        #     print(loss_q.item())
        #     print(pred_q)
        #     print(y_query[-1])
            # exit()
        return accs


    def finetunning(self, x_support, y_support, x_query, y_query):
        """
        :param x_support:   [setsz, c_, h, w]
        :param y_support:   [setsz]
        :param x_query:   [querysz, c_, h, w]
        :param y_query:   [querysz]
        :return:
        """
        assert len(x_support.shape) == 4, x_support.shape

        querysz = x_query.shape[0]
        corrects = [0 for _ in range(self.num_updates + 1)]

        # in order to not ruin the state of running_mean/variance and bn_weight/bias
        # we finetunning on the copied model instead of self.net
        self.net.eval()
        net = self.net
        # net = deepcopy(self.net)

        # 1. run the i-th task and compute loss for k=0
        logits = net(x_support)
        loss = F.cross_entropy(logits, y_support)
        grad = torch.autograd.grad(loss, net.parameters()) 
        fast_weights = list(map(lambda p: (p[1][0], p[1][1] - self.update_lr * p[0]), zip(grad, self.net.named_parameters())))

        # this is the loss and accuracy before first update
        with torch.no_grad():
            # [setsz, nway]
            logits_q = net(x_query)
            # [setsz]
            pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
            # scalar
            correct = torch.eq(pred_q, y_query).sum().item()
            corrects[0] = corrects[0] + correct

        # this is the loss and accuracy after the first update
        with torch.no_grad():
            # [setsz, nway]
            logits_q = net(x_query, fast_weights)
            # [setsz]
            pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
            # scalar
            correct = torch.eq(pred_q, y_query).sum().item()
            corrects[1] = corrects[1] + correct

        for k in range(1, self.num_updates_test):
            # 1. run the i-th task and compute loss for k=1~K-1
            logits = net(x_support, fast_weights)
            loss = F.cross_entropy(logits, y_support)
            # 2. compute grad on theta_pi
            grad = torch.autograd.grad(loss, fast_weights)
            # 3. theta_pi = theta_pi - train_lr * grad
            fast_weights = list(map(
                    lambda p: (p[1][0], p[1][1] - self.update_lr * p[0]), 
                    zip(grad, fast_weights)
            ))

            logits_q = net(x_query, fast_weights)
            # loss_q will be overwritten and just keep the loss_q on last update step.
            loss_q = F.cross_entropy(logits_q, y_query)

            with torch.no_grad():
                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                correct = torch.eq(pred_q, y_query).sum().item()  # convert to numpy
                corrects[k + 1] = corrects[k + 1] + correct


        del net
        accs = np.array(corrects) / querysz
        # if accs[0] == 0:
        #     print(loss.item())
        #     print(pred_q)
        #     print(y_query)
            # exit()
        return accs


def main():
    pass

if __name__ == '__main__':
    main()