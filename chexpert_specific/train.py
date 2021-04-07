import os, sys
import random
import argparse
import numpy as np
from pathlib import Path
from absl import app

import torch
import torchvision.models as models
from torch.optim.lr_scheduler import StepLR

sys.path.append(str(Path(__file__).parent.parent))
from utils.config import _C as cfg
from chexpert_specific.model.dataloader import build_dataloader
from chexpert_specific.model.trainer import Trainer

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True, type=str, default='sample', help='Name of output directory')
    parser.add_argument('--cfg', required=True, type=str, help='Path to config file')
    parser.add_argument('--gpu', type=int, required=True, help='GPU ID')
    parser.add_argument('--init', type=str, help='(optionally) path to pretrained model', default='')
    parser.add_argument('--iteration_start', type=int, default=0, help='(optionally) iteration to resume training')
    parser.add_argument('--seed', type=int, help='set random seed use this command', default=0)
    return parser.parse_args()


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'

'''
TODO
- test mixup
- add mixup stats to writer
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


def main(argv):
    args = parse_args()
    
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    if not torch.cuda.is_available():
        assert NotImplementedError
    print("Using", torch.cuda.get_device_name())
    set_seed(args.seed)
        
    torch.cuda.set_device(args.gpu)

    cfg.merge_from_file(args.cfg)
    cfg.freeze()

    output_dir = Path(cfg.OUTPUT_DIR) / args.output
    
    train_loader = build_dataloader("train")
    val_loader = build_dataloader("val")
    
    device = torch.device("cuda")
    model = models.resnet18(pretrained=False).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.SOLVER.BASE_LR, 
                                 weight_decay=cfg.SOLVER.WEIGHT_DECAY)
 
    if args.init:
        model.load_state_dict(torch.load(args.init, device))
        print("Loaded model!")

    scheduler = StepLR(optimizer, step_size=cfg.SOLVER.SCHEDULER_STEP_SIZE, gamma=0.1)

    kwargs = {
      'device': torch.device('cuda'),
      'model': model,
      'optimizer': optimizer,
      'scheduler': scheduler,
      'train_loader': train_loader,
      'val_loader': val_loader,
      'iterations': args.iteration_start,
      'output_dir': output_dir,
    }
    trainer = Trainer(**kwargs)
    trainer.train()



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
