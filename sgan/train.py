import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from models import *
from dataloader import *

cuda = True if torch.cuda.is_available() else False

import os

experiment_name = str(cfg.DATA.LABELED_SIZE)
if not os.path.exists("/saved_models_%s" % experiment_name):
    os.makedirs("/saved_models_%s" % experiment_name)

if not os.path.exists("/train_losses_%s" % experiment_name):
    os.makedirs("/train_losses_%s" % experiment_name)

g_net = BasicGenerator().cuda()
g_net.apply(weights_init)
d_net = BasicDiscriminator(k=2).cuda()
d_net.apply(weights_init)

dl_labeled, dl_unlabeled = build_dataloader('train')
dl_val, _ = build_dataloader('valid')

ce_loss = nn.CrossEntropyLoss()
bce_loss = nn.BCELoss()
softmax = nn.Softmax(dim=1)

beta1 = 0.5
optimizerD = optim.Adam(d_net.parameters(), lr=0.0002, betas=(beta1, 0.999))
optimizerG = optim.Adam(g_net.parameters(), lr=0.002, betas=(beta1, 0.999))

D_labeled_losses = []
D_unlabeled_losses = []
D_generated_losses = []
G_losses = []
val_acc_list = []

labeled_weight = 10

n_epochs = 10
g_steps = 5


def val_acc(discriminator):
    discriminator.eval()
    total = 0
    correct = 0
    for _, val_data in enumerate(dl_val):
        val_imgs, val_labels = val_data
        val_imgs = val_imgs.cuda()
        val_labels = val_labels.cuda()
        outputs = discriminator(val_imgs)
        predicted = torch.argmax(outputs.data, 1)
        total += val_labels.size(0)
        correct += (predicted == val_labels).sum().item()
    return 100 * correct / total

iter = 0
# set up for len(labeled) > len(unlabeled)
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
        d_net.train()

        # Train D with labeled images
        supervised_imgs, supervised_labels = labeled_data
        batch_size = supervised_labels.shape[0]
        supervised_imgs = supervised_imgs.cuda()
        supervised_labels = supervised_labels.cuda()
        supervised_pred = d_net(supervised_imgs)
        # print("Supervised output")
        # print(supervised_pred)
        # print(supervised_labels)
        labeled_loss = ce_loss(supervised_pred, supervised_labels) * labeled_weight
        labeled_loss.backward()

        # Train D with unlabeled images
        unsupervised_imgs, unsupervised_labels = unlabeled_data
        unsupervised_imgs = unsupervised_imgs.cuda()
        unsupervised_labels = unsupervised_labels.cuda()
        unsupervised_pred = d_net(unsupervised_imgs)
        unsupervised_pred = softmax(unsupervised_pred)
        # print("Unsupervised output")
        # print(unsupervised_pred)
        unlabeled_loss = bce_loss(unsupervised_pred[:, 2], torch.zeros(batch_size).cuda())
        unlabeled_loss.backward()

        # Train D with generated images
        z_input = generate_noise(batch_size).cuda()
        generated_imgs = g_net(z_input)
        generated_pred = d_net(generated_imgs)
        generated_labels = torch.from_numpy(np.ones(batch_size) * 2).long().cuda()
        # print("Generated output")
        # print(generated_pred)
        # print(generated_labels)
        generated_loss = ce_loss(generated_pred, generated_labels)
        generated_loss.backward()

        total_loss = labeled_loss / labeled_weight + unlabeled_loss + generated_loss
        print("Epoch %d iter %d" % (epoch, i))
        print("labeled loss: %f" % (labeled_loss / 10))
        print("unlabeled loss: %f" % unlabeled_loss)
        print("generated loss: %f" % generated_loss)
        print("total loss: %f" % total_loss)
        D_labeled_losses.append(labeled_loss / labeled_weight)
        D_unlabeled_losses.append(unlabeled_loss)
        D_generated_losses.append(generated_loss)
        optimizerD.step()

        total_gen_loss = 0
        for _ in range(g_steps):
            # Training G
            optimizerG.zero_grad()

            # Train G with generated images
            z_input = generate_noise(batch_size).cuda()
            generated_imgs = g_net(z_input)
            generated_pred = d_net(generated_imgs)
            generated_pred = softmax(generated_pred)
            # print(generated_pred)
            generated_loss = bce_loss(1 - generated_pred[:, 2], torch.ones(batch_size).cuda())
            total_gen_loss += generated_loss
            generated_loss.backward()

            optimizerG.step()
        G_losses.append(total_gen_loss / g_steps)
        print("generator loss: %f" % (total_gen_loss / g_steps))
        print()

        if iter % 50 == 0:
            # save discriminator & generator every 50 iterations
            torch.save(d_net.state_dict(), "/saved_models_%s/d_%d.pth" % (experiment_name, iter))
            torch.save(g_net.state_dict(), "/saved_models_%s/g_%d.pth" % (experiment_name, iter))

        if iter % 10:
            # calculate validation accuracy every 10 iterations
            acc = val_acc(d_net)
            val_acc_list.append(acc)
            print("Validation accuracy: %f" % acc)

        iter += 1

loss_path = "/train_losses_%s" % experiment_name
np.save('%s/D_labeled_losses.npy' % loss_path, np.array(D_labeled_loss))
np.save('%s/D_unlabeled_losses.npy' % loss_path, np.array(D_unlabeled_losses))
np.save('%s/D_generated_losses.npy' % loss_path, np.array(D_generated_losses))
np.save('%s/G_losses.npy' % loss_path, np.array(G_losses))
np.save('%s/val_acc.npy' % loss_path, np.array(val_acc_list))
