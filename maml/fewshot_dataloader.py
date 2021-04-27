import os, sys
import random
import numpy as np
import pandas as pd
from pathlib import Path

import torch
import PIL.Image as Image
from torch.utils.data import Dataset, DataLoader

sys.path.append(str(Path(__file__).absolute().parent.parent))
from utils.config import _C as cfg
from utils.transforms import get_transforms, label_to_binary_class
from utils.misc import set_seed

class ChexpertDataset(Dataset):
    def __init__(self, csv_path: Path, split: str) -> None:
        super(ChexpertDataset, self).__init__()
        self.data_path = Path(csv_path).parent
        self.annotations = pd.read_csv(csv_path).fillna(0)
        self.split = split
        self.height, self.width = 224, 224
        self.class_strings = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion']
        self.transforms = get_transforms(self.height, self.width, split)
        self.annotations['Binary Class'] = self.annotations.apply(lambda row: label_to_binary_class(row, self.class_strings), axis=1)
        
        self.num_tasks = cfg.MAML.N_TASKS
        self.n_way = cfg.MAML.N_WAY
        self.k_shot = cfg.MAML.K_SHOT
        self.k_query = cfg.MAML.K_QUERY
        self.size_support = self.n_way * self.k_shot
        self.size_query = self.n_way * self.k_query
        self.novel_classes = cfg.MAML.NOVEL_CLASSES

        assert len(self.novel_classes) >= self.n_way, f"There should be at least {self.n_way} novel classes."

        novel_classes_idxs = self.annotations['Binary Class'].isin(self.novel_classes)
        if split == 'train':
            self.annotations = self.annotations[~novel_classes_idxs]
        else:
            self.annotations = self.annotations[novel_classes_idxs]

        self.class_df_map = {}
        for cls_name in range(32):
            df = self.annotations[self.annotations['Binary Class'] == cls_name]
            if df.empty:
                continue
            assert len(df) >= self.k_shot + self.k_query, f"Class {cls_name} only has {len(df)} examples, {self.k_shot + self.k_query} expected."
            df = df.sample(frac=1).reset_index(drop=True)
            df = df.iloc[:self.k_shot + self.k_query]
            self.class_df_map[cls_name] = df

        self.class_pool = list(self.class_df_map.keys())
        self.task_classes = np.array([
            np.random.choice(self.class_pool, size=self.n_way, replace=False) for _ in range(self.num_tasks)
        ])

    def __len__(self) -> int:
        return self.num_tasks

    def _load_task_data(self, k, class_annotations):
        x = torch.FloatTensor(k, self.n_way, 3, self.height, self.width)
        y = torch.LongTensor(k, self.n_way)
        for i in np.random.permutation(range(self.n_way)):
            annotations = class_annotations[i]
            for j, row in annotations.iterrows():
                data = self.transforms(Image.open(self.data_path.parent / row['Path'])).repeat(3,1,1)
                x[j, i] = data
                y[j, i] = row['Binary Class']
        return x.reshape(-1, 3, self.height, self.width), y.flatten()

    def __getitem__(self, task_index: int) -> None:
        classes = self.task_classes[task_index]
        annot_support = [self.class_df_map[c].iloc[:self.k_shot] for c in classes]
        annot_query = [self.class_df_map[c].iloc[self.k_shot:].reset_index(drop=True) for c in classes]
        
        support_x, support_y = self._load_task_data(self.k_shot, annot_support)
        query_x, query_y = self._load_task_data(self.k_query, annot_query)

        return support_x, support_y, query_x, query_y


def build_dataloader(split):
    valid_splits = ["train", "valid", "test"]
    assert split in valid_splits, f"{split} should be one of {valid_splits}."

    dataset = ChexpertDataset(Path(cfg.DATA.PATH) / f"{split}.csv", split)
    dl_labeled = DataLoader(dataset, batch_size=1, num_workers=min(os.cpu_count(), 12))
    return dl_labeled


if __name__ == '__main__':
    set_seed(0)

    cfg.merge_from_file("config/maml_base.yaml")
    cfg.freeze()

    ds = ChexpertDataset(Path(cfg.DATA.PATH) / 'train.csv', 'train')
    print(len(ds))
    for t, (sx, sy, qx, qy) in enumerate(ds):
        print(t)
        print(torch.norm(sx).item())
        exit()

