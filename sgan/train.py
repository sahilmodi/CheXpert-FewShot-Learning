import torch
import torch.nn as nn

from sgan.models import *
from sgan.dataloader import *

cuda = True if torch.cuda.is_available() else False

g_net = BasicGenerator().cuda()
g_net.apply(weights_init)
d_net = BasicDiscriminator().cuda()
d_net.apply(weights_init)

dl_labeled, dl_unlabeled = build_dataloader('train')

n_epochs = 10
for epoch in range(n_epochs):
    n_batches = 5
    for i in range(n_batches):
        # train D on supervised data
        # get samples from labeled dataset and predict

        # train D on unsupervised data
        # get samples from unlabeled dataset and predict

        # train D on generated data
        # get samples from generator

        # update D

        # train G on random noise
        # get random noise vector, generate fake data,
        # use discriminator output

        # update G
