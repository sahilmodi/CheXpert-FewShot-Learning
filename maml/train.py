import  torch, os
import  numpy as np
import  scipy.stats
from    torch.utils.data import DataLoader
from    torch.optim import lr_scheduler
from tqdm import tqdm
import  random, sys, pickle
import  argparse

import shutil
from glob import glob
from pathlib import Path
import torchvision.models as models
sys.path.append(str(Path(__file__).parent.parent))
from utils.misc import set_seed
from utils.config import _C as cfg

from maml.model.trainer import Trainer
from maml.model.meta import Meta
from maml.fewshot_dataloader import build_dataloader

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', required=True, type=str, help='Path to config file')
    parser.add_argument('--output', required=True, type=str, default='sample', help='Name of output directory')
    parser.add_argument('--gpu', type=int, required=True, help='GPU ID')
    parser.add_argument('--seed', type=int, help='set random seed use this command', default=0)
    return parser.parse_args()

def mean_confidence_interval(accs, confidence=0.95):
    n = accs.shape[0]
    m, se = np.mean(accs), scipy.stats.sem(accs)
    h = se * scipy.stats.t._ppf((1 + confidence) / 2, n - 1)
    return m, h


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

    device = torch.device('cuda')
    maml = Meta().to(device)

    tmp = filter(lambda x: x.requires_grad, maml.parameters())
    num = sum(map(lambda x: np.prod(x.shape), tmp))
    # print(maml)
    print('Total trainable tensors:', num)

    # batchsz here means total episode number
    '''
    mini = MiniImagenet('/home/i/tmp/MAML-Pytorch/miniimagenet/', mode='train', n_way=args.n_way, k_shot=args.k_spt,
                        k_query=args.k_qry,
                        batchsz=10000, resize=args.imgsz)
    mini_test = MiniImagenet('/home/i/tmp/MAML-Pytorch/miniimagenet/', mode='test', n_way=args.n_way, k_shot=args.k_spt,
                             k_query=args.k_qry,
                             batchsz=100, resize=args.imgsz)
    '''
    train_loader = build_dataloader("train")
    val_loader = build_dataloader("valid")
    test_loader = build_dataloader("test")

    print("Train Batches:", len(train_loader), "| Val Batches:", len(val_loader), "| Test Batches:", len(test_loader))

    postfix_map = {}
    t = tqdm(range(cfg.MAML.EPOCHS), leave=True, dynamic_ncols=True)
    for epoch in t:
        # fetch meta_batchsz num of episode each time
        # db = DataLoader(mini, args.task_num, shuffle=True, num_workers=1, pin_memory=True)

        # for step, (x_spt, y_spt, x_qry, y_qry) in enumerate(db):
        for step, (x_spt, y_spt, x_qry, y_qry) in enumerate(tqdm(train_loader, position=1, dynamic_ncols=True, leave=False)):

            x_spt, y_spt, x_qry, y_qry = x_spt.to(device), y_spt.to(device), x_qry.to(device), y_qry.to(device)
            accs = maml(x_spt, y_spt, x_qry, y_qry)

            if step % 30 == 0:
                postfix_map["trAcc"] = accs

            if step % 500 == 0:  # evaluation
                # db_test = DataLoader(mini_test, 1, shuffle=True, num_workers=1, pin_memory=True)
                accs_all_test = []

                for x_spt, y_spt, x_qry, y_qry in tqdm(val_loader, desc="Validation", position=2, leave=False):
                    x_spt, y_spt, x_qry, y_qry = x_spt.squeeze(0).to(device), y_spt.squeeze(0).to(device), \
                                                 x_qry.squeeze(0).to(device), y_qry.squeeze(0).to(device)

                    accs = maml.finetunning(x_spt, y_spt, x_qry, y_qry)
                    accs_all_test.append(accs)

                # [b, update_step+1]
                accs = np.array(accs_all_test).mean(axis=0).astype(np.float16)
                postfix_map["valAcc"] = accs
            t.set_postfix(postfix_map)


if __name__ == '__main__':
    main()
