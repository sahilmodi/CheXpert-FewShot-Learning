import torch
import torch.nn as nn
import random 
import numpy as np


class RobustWarpLoss(nn.Module):
    def __init__(self, l, u, kappa=5):
        super(RobustWarpLoss, self).__init__()
        self.l = l
        self.u = u
        self.L = [1./1] # rank weights
        self.kappa = kappa
        for i in range(1, self.l):
            self.L.append(self.L[i - 1] + (1. / (i + 1)))
    
    def ramp_loss(self, t, s=-0.8):
        return min(1 - s, max(0, 1 - t))

    def forward(self, inp, target, S):
        batch_size = target.size()[0]
        num_labels = target.size()[1]
        max_num_trials = num_labels - 1
        loss = 0.

        for i in range(batch_size):
            if i <= self.l:
                s_i = 1
            else:
                s_i = S[i - self.l]
            for j in range(num_labels):
                if target[i, j] == 1:
                    neg_labels_idx = np.array([idx for idx, v in enumerate(target[i, :]) if v == 0])
                    neg_idx = np.random.choice(neg_labels_idx, replace=False)
                    sample_score_margin = 1 - inp[i, j] + inp[i, neg_idx]
                    num_trials = 0

                    while sample_score_margin < 0 and num_trials < max_num_trials:
                        neg_idx = np.random.choice(neg_labels_idx, replace=False)
                        num_trials += 1
                        sample_score_margin = 1 - inp[i, j] + inp[i, neg_idx]

                    r_j = np.floor(max_num_trials / num_trials)
                    weight = self.L[r_j]

                    for k in range(num_labels):
                        if target[i, k] == 0:
                            score_margin = 1 - inp[i, j] + inp[i, k]
                            loss += (s_i * weight * self.ramp_loss(score_margin))

                for l in range(num_labels):
                    for m in range(self.l + self.u):
                        loss += self.kappa * (s_i * self.ramp_loss(target[m, l] * inp[m, l]))

        return loss
