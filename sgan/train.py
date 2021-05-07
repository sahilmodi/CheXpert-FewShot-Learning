import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.optim as optim

from models import *
from dataloader import *

cuda = True if torch.cuda.is_available() else False

g_net = BasicGenerator().cuda()
g_net.apply(weights_init)
d_net = BasicDiscriminator(k=2).cuda()
d_net.apply(weights_init)

dl_labeled, dl_unlabeled = build_dataloader('train')

print("labeled size: ", len(dl_labeled))
print("unlabeled size: ", len(dl_unlabeled))
train_features, train_labels = next(iter(dl_labeled))

criterion = nn.CrossEntropyLoss()
bce = nn.BCELoss()
softmax = nn.Softmax(dim=1)
optimizerD = optim.Adam(d_net.parameters(), lr=0.0002)
optimizerG = optim.Adam(g_net.parameters(), lr=0.0002)

# set up for labeled > unlabeled
n_epochs = 1
for epoch in range(n_epochs):
    unlabeled_iter = iter(dl_unlabeled)
    for i, labeled_data in enumerate(dl_labeled):
        try:
            unlabeled_data = next(unlabeled_iter)
        except StopIteration:
            unlabeled_iter = iter(dl_unlabeled)
            unlabeled_data = next(dl_unlabeled)

        # Training D
        optimizerD.zero_grad()

        # Train D with labeled images
        supervised_imgs, supervised_labels = labeled_data
        batch_size = supervised_labels.shape[0]
        supervised_imgs = supervised_imgs.cuda()
        supervised_labels = supervised_labels.cuda()
        supervised_pred = d_net(supervised_imgs)
        labeled_loss = criterion(supervised_pred, supervised_labels)
        labeled_loss.backward()

        # Train D with unlabeled images
        unsupervised_imgs, unsupervised_labels = unlabeled_data
        unsupervised_imgs = unsupervised_imgs.cuda()
        unsupervised_labels = unsupervised_labels.cuda()
        unsupervised_pred = d_net(unsupervised_imgs)
        unsupervised_pred = softmax(unsupervised_pred)
        unlabeled_loss = bce(unsupervised_pred[:, 2], torch.zeros(batch_size).cuda())
        unlabeled_loss.backward()

        # Train D with generated images
        z_input = generate_noise(batch_size).cuda()
        generated_imgs = g_net(z_input)
        generated_pred = d_net(generated_imgs)
        generated_labels = torch.from_numpy(np.ones(batch_size) * 2).long().cuda()
        generated_loss = criterion(generated_pred, generated_labels)
        generated_loss.backward()

        total_loss = labeled_loss + unlabeled_loss + generated_loss
        print("Epoch %d iter %d" % (epoch, i))
        print("labeled loss: %f" % labeled_loss)
        print("unlabeled loss: %f" % unlabeled_loss)
        print("generated loss: %f" % generated_loss)
        print("total loss: %f" % total_loss)
        optimizerD.step()

        # Training G
        g_net.zero_grad()

        # Train G with generated images
        z_input = generate_noise(batch_size).cuda()
        generated_imgs = g_net(z_input)
        generated_pred = d_net(generated_imgs)
        generated_pred = softmax(generated_pred)
        generated_loss = bce(generated_pred[:, 2], torch.ones(batch_size).cuda())
        generated_loss.backward()
        print("generator loss: %f" % generated_loss)
        print()
        optimizerG.step()



