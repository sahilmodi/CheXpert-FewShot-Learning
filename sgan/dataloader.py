import os
import numpy as np
import pandas as pd
from pathlib import Path

import torch
import PIL.Image as Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

from config import _C as cfg
from utils.transforms import get_transforms


class ChexpertDataset(Dataset):
    def __init__(self, csv_path: Path, split: str) -> None:
        super(ChexpertDataset, self).__init__()
        self.data_path = Path(csv_path).parent
        self.annotations = pd.read_csv(csv_path).fillna(0)
        self.split = split
        self.transforms = None
        self.height, self.width = 224, 224
        self.transforms = get_transforms(self.height, self.width, split)
        if split == "train":
            assert cfg.DATA.BATCH_SIZE <= cfg.DATA.LABELED_SIZE, "Batch size must be smaller than train size."
            labeled_size = cfg.DATA.LABELED_SIZE
            self.normal = self.annotations[
                (self.annotations['Atelectasis'] == 0) & (self.annotations['Cardiomegaly'] == 0) & (
                        self.annotations['Consolidation'] == 0) & (self.annotations['Edema'] == 0) & (
                        self.annotations['Pleural Effusion'] == 0)]
            self.abnormal = self.annotations[
                (self.annotations['Atelectasis'] != 0) | (self.annotations['Cardiomegaly'] != 0) | (
                        self.annotations['Consolidation'] != 0) | (self.annotations['Edema'] != 0) | (
                        self.annotations['Pleural Effusion'] != 0)]
            # self.normal = self.normal.sample(n=int(labeled_size / 2)).reset_index(drop=True)
            self.normal = self.normal.sample(n=int(labeled_size / 2))
            self.abnormal = self.abnormal.sample(n=int(labeled_size / 2))
            normal_indices = self.normal.index
            abnormal_indices = self.abnormal.index
            self.annotations = self.annotations.drop(normal_indices).drop(abnormal_indices)
            self.train_annotations = self.normal.append(self.abnormal, ignore_index=True)

    def __len__(self) -> int:
        return self.annotations.shape[0] if self.split != 'train' else self.train_annotations.shape[0]

    def __getitem__(self, index: int):
        annotations = self.annotations if self.split != 'train' else self.train_annotations
        annotation = annotations.iloc[index]
        image = Image.open(self.data_path.parent / annotation['Path'])
        classes = annotation[
            ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion']].values.astype("float32")
        if np.sum(classes) > 0:
            label = 1
        else:
            label = 0
        data = self.transforms(image)
        return data, torch.tensor(label).long()


class ChexpertDatasetUnlabeled(Dataset):
    def __init__(self, csv_path: Path, shuffled_annotations: pd.DataFrame) -> None:
        # shuffled_annotations are remaining annotations not used in labeled
        super(ChexpertDatasetUnlabeled, self).__init__()
        self.data_path = Path(csv_path).parent
        unlabeled_size = cfg.DATA.UNLABELED_SIZE
        self.annotations = shuffled_annotations[:unlabeled_size].reset_index(drop=True)
        self.height, self.width = 224, 224
        self.transforms = get_transforms(self.height, self.width, 'train')

    def __len__(self) -> int:
        return self.annotations.shape[0]

    def __getitem__(self, index: int) -> None:
        annotation = self.annotations.iloc[index]
        image = Image.open(self.data_path.parent / annotation['Path'])
        return self.transforms(image), torch.tensor(2).long()


def build_dataloader(split):
    valid_splits = ["train", "valid", "test"]
    assert split in valid_splits, f"{split} should be one of {valid_splits}."

    ds_path = Path(cfg.DATA.PATH)
    # ds_path = Path("/home/koyejolab/CheXpert/CheXpert-v1.0-small")
    bs = cfg.DATA.BATCH_SIZE

    is_train = split == 'train'
    dataset = ChexpertDataset(ds_path / f"{split}.csv", split)
    dl_labeled = DataLoader(dataset, batch_size=bs, num_workers=min(os.cpu_count(), 12), shuffle=is_train)
    dl_unlabeled = None
    if split == 'train':
        dataset_u = ChexpertDatasetUnlabeled(ds_path / f'{split}.csv', dataset.annotations)
        # bs = cfg.DATA.UNLABELED_SIZE // (cfg.DATA.LABELED_SIZE / bs)
        dl_unlabeled = DataLoader(dataset_u, batch_size=int(bs), num_workers=min(os.cpu_count(), 12), shuffle=is_train)
    return dl_labeled, dl_unlabeled
