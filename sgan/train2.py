import os
import warnings
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from models import *
from dataloader import *

# cuda = True if torch.cuda.is_available() else False
torch.cuda.set_device(2)
torch.set_num_threads(1)

import os
import sklearn.metrics as metrics

experiment_name = str(cfg.DATA.LABELED_SIZE)
if not os.path.exists("saved_models_%s" % experiment_name):
    os.makedirs("saved_models_%s" % experiment_name)

if not os.path.exists("train_losses_%s" % experiment_name):
    os.makedirs("train_losses_%s" % experiment_name)

# g_net = BasicGenerator().cuda()
g_net = Generator().cuda()
g_net.apply(weights_init)
# d_net = BasicDiscriminator(k=2).cuda()
d_net = Discriminator().cuda()
d_net.apply(weights_init)

dl_labeled, dl_unlabeled = build_dataloader('train')
dl_val, _ = build_dataloader('test')

ce_loss = nn.CrossEntropyLoss()
bce_loss = nn.BCELoss()
bce_loss_with_logits = nn.BCEWithLogitsLoss()
softmax = nn.Softmax(dim=1)

beta1 = 0.5
optimizerD = optim.Adam(d_net.parameters(), lr=0.0002, betas=(beta1, 0.999))
# optimizerD = optim.SGD(d_net.parameters(), lr=0.0002, momentum=0.9)
optimizerG = optim.Adam(g_net.parameters(), lr=0.0002, betas=(beta1, 0.999))

D_labeled_losses = []
D_unlabeled_losses = []
D_generated_losses = []
G_losses = []
val_acc_list = []

labeled_weight = 10

n_epochs = 10
g_steps = 1

writer = SummaryWriter('sgan/tensorboard_logs/run0_tanh_batchnorm_gsteps1_lrSame_DCGAN_64_multilabel_1k/', flush_secs=60)
# writer = SummaryWriter('sgan/tensorboard_logs/debug/', flush_secs=60)

def get_auc(labels, y):
    try:
        return metrics.roc_auc_score(labels.cpu().numpy(), y.detach().cpu().numpy(), average='weighted')
    except ValueError:
        return np.nan

def get_prc(labels, y):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        return metrics.average_precision_score(labels.cpu().numpy(), y.detach().cpu().numpy(), average='weighted')

def val_acc(discriminator):
    discriminator.eval()
    total = 0
    correct = 0
    count0, count1, count2 = 0, 0, 0
    aucs, prcs = [], []
    for _, val_data in enumerate(dl_val):
        val_imgs, val_labels = val_data
        val_imgs = val_imgs.cuda()
        val_labels = val_labels.cuda()
        outputs = discriminator(val_imgs)
        '''
        predicted = torch.argmax(outputs.data, 1)
        count0 += torch.sum(predicted == 0).item() # len(predicted[predicted == 2]) 
        count1 += torch.sum(predicted == 1).item() # len(predicted[predicted == 2]) 
        count2 += torch.sum(predicted == 2).item() # len(predicted[predicted == 2]) 
        total += val_labels.size(0)
        correct += (predicted == val_labels).sum().item()
        '''
        aucs.append(get_auc(val_labels, outputs.squeeze()))
        prcs.append(get_prc(val_labels, outputs.squeeze()))
    return np.nanmean(aucs), np.nanmean(prcs)  

iterations = 0
# set up for len(labeled) > len(unlabeled)
for epoch in range(n_epochs):
    # unlabeled_iter = iter(dl_unlabeled)
    labeled_iter = iter(dl_labeled)
    for i, unlabeled_data in enumerate(dl_unlabeled):
        try:
            labeled_data = next(labeled_iter)
        except StopIteration:
            labeled_iter = iter(dl_labeled)
            labeled_data = next(labeled_iter)

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
        # labeled_loss = ce_loss(supervised_pred.squeeze(), supervised_labels) * labeled_weight
        labeled_loss = bce_loss_with_logits(supervised_pred.squeeze(), supervised_labels) * labeled_weight
        labeled_loss.backward()

        # Train D with unlabeled images
        # unsupervised_imgs, unsupervised_labels = unlabeled_data
        unsupervised_imgs = unlabeled_data
        unsupervised_imgs = unsupervised_imgs.cuda()
        # unsupervised_labels = unsupervised_labels.cuda()
        unsupervised_pred = d_net(unsupervised_imgs)
        # unsupervised_pred = softmax(unsupervised_pred)
        # print("Unsupervised output")
        # print(unsupervised_pred)
        # unlabeled_loss = bce_loss(unsupervised_pred[:, 2].squeeze(), torch.zeros(batch_size).cuda())
        unlabeled_loss = bce_loss_with_logits(unsupervised_pred[:,5].squeeze(), torch.zeros(batch_size).cuda())
        unlabeled_loss.backward()

        # Train D with generated images
        # z_input = generate_noise(batch_size).cuda()
        z_input = torch.randn(batch_size, 100, 1, 1).cuda()
        generated_imgs = g_net(z_input)
        generated_pred = d_net(generated_imgs)
        # generated_labels = torch.from_numpy(np.ones(batch_size) * 2).long().cuda()
        # generated_labels = torch.from_numpy(np.ones(batch_size)).long().cuda()
        generated_labels = torch.ones(batch_size).cuda()
        # print("Generated output")
        # print(generated_pred)
        # print(generated_labels)
        # generated_loss = ce_loss(generated_pred.squeeze(), generated_labels)
        generated_loss = bce_loss_with_logits(generated_pred[:, 5].squeeze(), generated_labels)
        generated_loss.backward()

        total_loss = labeled_loss / labeled_weight + unlabeled_loss + generated_loss
        print("Epoch %d iter %d" % (epoch, i))
        print("labeled loss: %f" % (labeled_loss / 10))
        print("unlabeled loss: %f" % unlabeled_loss)
        print("generated loss: %f" % generated_loss)
        print("total loss: %f" % total_loss)

        # tensorboard
        writer.add_scalar('labeled loss', labeled_loss / 10, iterations)
        writer.add_scalar('unlabeled loss', unlabeled_loss, iterations)
        writer.add_scalar('generated loss', generated_loss, iterations)
        writer.add_scalar('total loss', total_loss, iterations)
        
        D_labeled_losses.append(labeled_loss / labeled_weight)
        D_unlabeled_losses.append(unlabeled_loss)
        D_generated_losses.append(generated_loss)
        optimizerD.step()

        total_gen_loss = 0
        for _ in range(g_steps):
            # Training G
            optimizerG.zero_grad()

            # Train G with generated images
            # z_input = generate_noise(batch_size).cuda()
            z_input = torch.randn(batch_size, 100, 1, 1).cuda()
            generated_imgs = g_net(z_input)
            generated_pred = d_net(generated_imgs)
            # generated_pred = softmax(generated_pred)
            # print(generated_pred)
            # generated_loss = bce_loss(1 - generated_pred[:, 2].squeeze(), torch.ones(batch_size).cuda())
            generated_loss = bce_loss_with_logits(1 - generated_pred[:, 5].squeeze(), torch.ones(batch_size).cuda())
            total_gen_loss += generated_loss
            generated_loss.backward()

            optimizerG.step()
        G_losses.append(total_gen_loss / g_steps)
        print("generator loss: %f" % (total_gen_loss / g_steps))
        print()

        if iterations % 50 == 0:
            # save discriminator & generator every 50 iterations
            torch.save(d_net.state_dict(), "saved_models_%s/d_%d.pth" % (experiment_name, iterations))
            torch.save(g_net.state_dict(), "saved_models_%s/g_%d.pth" % (experiment_name, iterations))

        if iterations % 10 == 0:
            # calculate validation accuracy every 10 iterations
            # acc, count0, count1, count2 = val_acc(d_net)
            auc, prc = val_acc(d_net)
            # writer.add_scalar('val_acc', acc, iterations)
            '''
            writer.add_scalar('distribution/count0', count0, iterations)
            writer.add_scalar('distribution/count1', count1, iterations)
            writer.add_scalar('distribution/count2', count2, iterations)
            '''
            writer.add_scalar('auc', auc, iterations)
            writer.add_scalar('prc', prc, iterations)
            writer.add_image('vis/generator_output', generated_imgs[0], iterations)

        iterations += 1

loss_path = "/train_losses_%s" % experiment_name
np.save('%s/D_labeled_losses.npy' % loss_path, np.array(D_labeled_loss))
np.save('%s/D_unlabeled_losses.npy' % loss_path, np.array(D_unlabeled_losses))
np.save('%s/D_generated_losses.npy' % loss_path, np.array(D_generated_losses))
np.save('%s/G_losses.npy' % loss_path, np.array(G_losses))
np.save('%s/val_acc.npy' % loss_path, np.array(val_acc_list))
