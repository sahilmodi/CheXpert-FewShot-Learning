import os, sys
import shutil
import random
import argparse
import numpy as np
from glob import glob
from pathlib import Path

import torch
import torchvision.models as models
from torch.optim.lr_scheduler import StepLR

sys.path.append(str(Path(__file__).parent.parent))
from utils.config import _C as cfg
from chexpert_specific.model.dataloader import build_dataloader
from chexpert_specific.model.trainer import Trainer
from chexpert_specific.model.net import Net


'''
TODO
- test mixup
- add mixup stats to writer
'''

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', required=True, type=str, help='Path to config file')
    parser.add_argument('--output', required=True, type=str, default='sample', help='Name of output directory')
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


def main():
    args = parse_args()
    
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    if not torch.cuda.is_available():
        assert NotImplementedError
    torch.cuda.set_device(args.gpu)
    print("Using", torch.cuda.get_device_name())
    set_seed(args.seed)

    cfg.merge_from_file(args.cfg)
    cfg.freeze()

    output_dir = Path(cfg.OUTPUT_ROOT_DIR) / args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(args.cfg, output_dir / 'config.yaml')
    
    train_loader = build_dataloader("train")
    val_loader = build_dataloader("valid")
    test_loader = build_dataloader("test")

    print("Train Batches:", len(train_loader), "| Val Batches:", len(val_loader), "| Test Batches:", len(test_loader))
    
    device = torch.device("cuda")
    model = Net().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.SOLVER.BASE_LR, weight_decay=cfg.SOLVER.WEIGHT_DECAY)
 
    if args.init:
        model.load_state_dict(torch.load(args.init, device))
        print("Loaded model!")

    scheduler = StepLR(optimizer, step_size=cfg.SOLVER.SCHEDULER_STEP_SIZE, gamma=0.1)

    kwargs = {
      'model': model,
      'optimizer': optimizer,
      'scheduler': scheduler,
      'train_loader': train_loader,
      'val_loader': val_loader,
      'test_loader': test_loader,
      'iterations': args.iteration_start,
      'output_dir': output_dir,
    }
    trainer = Trainer(**kwargs)

    try:
        trainer.train()
    except BaseException:
        if len(glob(f"{output_dir}/*.pth")) < 1:
            shutil.rmtree(output_dir, ignore_errors=True)
        raise


if __name__ == '__main__':
    main()