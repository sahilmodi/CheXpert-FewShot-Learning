import torch, os
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter
import torchvision.models as models
from PIL import Image
import csv
import numpy as np
import torchvision.transforms as transforms
import torchvision.models as models
import os
import pdb
import time
from itertools import permutations
from absl import app
from absl import flags
import matplotlib.pyplot as plt
# from data_loader import Dataset
from tqdm import tqdm
from sklearn import metrics

FLAGS = flags.FLAGS

flags.DEFINE_integer('gpu', 4, 'Which GPU to use.')
flags.DEFINE_string('logdir', 'runs_debug', 'Name of tensorboard logdir')
flags.DEFINE_string('logdir_prefix', '/newdata01/arjung2/d4rl_forward_regression_runs/', 'Name of tensorboard logdir')
flags.DEFINE_integer('batch_size', 128, 'batch_size')
flags.DEFINE_float('weight_decay', 0.00001, 'Weight decay in optimizer')
flags.DEFINE_float('lr', 1e-3, 'Learning rate')
flags.DEFINE_integer('lr_decay_every', 15000, 'Learning rate')
flags.DEFINE_integer('step_size', 1, 'number of epochs to wait before dropping lr')
flags.DEFINE_string('model_path', '0', 'path from which to load model')
flags.DEFINE_integer('iteration_start', 0, 'number of iterations to start at if loading model')
flags.DEFINE_boolean('use_mixup', False, 'whether to use mixup training or not')
flags.DEFINE_integer('print_every', 25, 'number of iterations to do val for')

'''
TODO
- finish mixup
- val: add mixup, clean up
- implement AUC and PRC
'''
    
def log_aps(writer, iteration, acts, ys):
    acts = np.concatenate(acts, 0)
    ys = np.concatenate(ys, 0)
    num_actions = ys.shape[1]
    aps_y = np.zeros((num_actions, num_actions))
    for i in range(num_actions):
      for j in range(num_actions):
        ap, _, __ = calc_pr(acts == j, ys[:,i])
        aps_y[i,j] = ap[0]
    aps_y = np.max(aps_y, 0)
    for i, p in enumerate(aps_y):
        writer.add_scalar('aps_y/train_{:02d}'.format(i), p, iteration)
        print(f'                 aps_y/{i:02d} [{iteration:6d}]: {p:0.8f}')

def log(writer, optimizer, iteration, act_losses, train_accs):
    print('')
    ks = ['lr', 'action_loss', 'action_acc']
    vs = [optimizer.param_groups[0]['lr'], 
          np.mean(act_losses), np.mean(train_accs)]
    for k, v in zip(ks, vs):
        print('{:>25s} [{:6d}]: {:0.8f}'.format(k, iteration, v))
        writer.add_scalar(f'loss/{k}', v, iteration)
    return   


def train(model, optimizer, epoch, device, train_loader, val_loader,
          train_writer, val_writer, iteration, num_actions, scheduler):
    
    model.train()

    print_every, val_every = 100, 300
    train_loss, train_acc = 0, 0    
    cross_entropy_loss = nn.CrossEntropyLoss()
    
    labels_, ys, train_loss, train_acc = [], [], [], []
    if FLAGS.use_mixup:
        ys_mixup, train_loss_mixup, train_acc_mixup = [], [], []
        
    for batch_idx, (imgs, labels) in enumerate(train_loader):

        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()

        # forward pass 
        y = model(imgs)

        # losses
        loss = cross_entropy_loss(y, labels)

        # accuracy
        pred = y.argmax(dim=1, keepdim=True) 
        
        labels_.append(labels.detach().cpu().numpy())
        ys.append((F.softmax(y, dim=1)).detach().cpu().numpy())
        train_loss += [loss.item()]
        train_acc += [pred.eq(labels.view_as(pred)).float().mean().item()]

        if FLAGS.use_mixup:
            # what is alpha --> see mixup reference
            # "additional samples" --> is there a regular non-mixup loss?
            # y_bar equation in paper is not used?

            # generate mixup parameter
            lambda_ =

            inds1 = torch.arange(FLAGS.batch_size)
            inds2 = torch.randperm(FLAGS.batch_size)

            x_bar = lambda_ * imgs[inds1] + (1 - lambda_) * imgs[inds2]

            # forward pass
            y_bar = model(x_bar)
            
            bce_loss = nn.BCELoss()
            loss_mixup = lambda_ * bce_loss(y_bar, labels[inds1]) + (1 - lambda_) * bce_loss(y_bar, labels[inds2])
            loss_mixup = loss_mixup.sum()

            loss += loss_mixup

            train_loss_mixup += [loss_mixup.item()]
            pred_mixup = y_bar.argmax(dim=1, keepdim=True)
            ys_mixup.append((F.softmax(y_bar, dim=1)).detach().cpu().numpy())
            train_acc_mixup += [pred_mixup.eq(labels.view_as(pred_mixup)).float().mean().item()]

        # backprop
        loss.backward()
        optimizer.step()
        iteration += 1
            
        if batch_idx % print_every == 0 and batch_idx != 0:
            log(train_writer, optimizer, iteration, train_loss, train_acc)
            log_aps(train_writer, iteration, acts, ys)
                        
            train_loss, train_acc = [], []
            labels_, ys = [], []           
            if FLAGS.use_mixup:
                train_loss_mixup, train_acc_mixup = [], []
                ys_mixup = []
        
    # validation
    val_act_loss, val_act_acc, val_acts, val_ys = validate(model, device, val_loader, FLAGS.print_every)

    model.train()

    log(val_writer, optimizer, iteration, val_act_loss, val_act_acc)
    log_aps(val_writer, iteration, val_acts, val_ys)
            
    # save model
    file_name = os.path.join(FLAGS.logdir_prefix, FLAGS.logdir, 'model-{:d}.pth'.format(iteration))
    torch.save(model.state_dict(), file_name)
        
    return iteration
        
        

def validate(model, device, val_loader, print_every):

    model.eval()

    val_loss, val_acc, j = 0, 0, 0
    cross_entropy_loss = nn.CrossEntropyLoss()
    
    acts, ys, val_act_loss, val_acc = [], [], [], []
    
    with torch.no_grad():
        for batch_idx, (o_tm1, a_tm1, o_t) in enumerate(val_loader):

            o_tm1, a_tm1, o_t = o_tm1.to(device), a_tm1.to(device), o_t.to(device)

            # forward pass
            y = model(o_tm1, o_t)

            # losses
            loss = cross_entropy_loss(y, a_tm1)
            
            # accuracy
            pred = y.argmax(dim=1, keepdim=True)        # get the index of the max log-probability

            acts.append(a_tm1.detach().cpu().numpy())
            ys.append((F.softmax(y, dim=1)).detach().cpu().numpy())
            val_act_loss += [loss.item()]
            val_acc += [pred.eq(a_tm1.view_as(pred)).float().mean().item()]

            j += 1
            if j == print_every:
                break

    return val_act_loss, val_acc, acts, ys


def main(argv):
        
    torch.cuda.set_device(FLAGS.gpu)
    torch.set_num_threads(1)
    
    train_dataset = Dataset(train=True)
    train_loader = data.DataLoader(train_dataset, batch_size=FLAGS.batch_size, 
                                   shuffle=True, num_workers=2, drop_last=True)
    
    val_dataset = Dataset(train=False)
    val_loader = data.DataLoader(val_dataset, batch_size=FLAGS.batch_size,  
                                 num_workers=0, drop_last=True)
    
    device = torch.device("cuda")
    model = models.resnet18(pretrained=False).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=FLAGS.lr, 
                                 weight_decay=FLAGS.weight_decay)
 
    if FLAGS.model_path != '0':
        model.load_state_dict(torch.load(FLAGS.model_path, device))
        print("loaded model!")

    train_writer = SummaryWriter(FLAGS.logdir_prefix + FLAGS.logdir + '/train/', flush_secs=60)
    val_writer = SummaryWriter(FLAGS.logdir_prefix + FLAGS.logdir + '/val/', flush_secs=60)
       
    scheduler = StepLR(optimizer, step_size=FLAGS.step_size, gamma=0.1)
    iteration = FLAGS.iteration_start

    for epoch in range(1, 1000):
        print("Train Epoch: ", epoch)
        iteration = train(model, optimizer, epoch, device, train_loader,
                          val_loader, train_writer, val_writer, iteration,
                          num_actions, scheduler)
        scheduler.step()



def calc_pr(gt, out, wt=None):
  if wt is None:
    wt = np.ones((gt.size,1))

  gt = gt.astype(np.float64).reshape((-1,1))
  wt = wt.astype(np.float64).reshape((-1,1))
  out = out.astype(np.float64).reshape((-1,1))

  gt = gt*wt
  tog = np.concatenate([gt, wt, out], axis=1)*1.
  ind = np.argsort(tog[:,2], axis=0)[::-1]
  tog = tog[ind,:]
  cumsumsortgt = np.cumsum(tog[:,0])
  cumsumsortwt = np.cumsum(tog[:,1])
  prec = cumsumsortgt / cumsumsortwt
  rec = cumsumsortgt / np.sum(tog[:,0])

  ap = voc_ap(rec, prec)
  return ap, rec, prec

def voc_ap(rec, prec):
  rec = rec.reshape((-1,1))
  prec = prec.reshape((-1,1))
  z = np.zeros((1,1)) 
  o = np.ones((1,1))
  mrec = np.vstack((z, rec, o))
  mpre = np.vstack((z, prec, z))
  for i in range(len(mpre)-2, -1, -1):
    mpre[i] = max(mpre[i], mpre[i+1])

  I = np.where(mrec[1:] != mrec[0:-1])[0]+1;
  ap = 0;
  for i in I:
    ap = ap + (mrec[i] - mrec[i-1])*mpre[i];
  return ap

if __name__ == '__main__':
    app.run(main)
