import os
import numpy as np
import pandas as pd
from pathlib import Path

import torch
import torch.nn.functional as F
import PIL.Image as Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

from utils.config import _C as cfg
from utils.transforms import get_transforms

class ChexpertDataset(Dataset):
    def __init__(self, csv_path: Path, split: str) -> None:
        super(ChexpertDataset, self).__init__()
        self.data_path = Path(csv_path).parent
        self.annotations = pd.read_csv(csv_path).fillna(0)
        self.train_annotations = None
        self.split = split
        self.transforms = None
        self.height, self.width = 224, 224
        self.transforms = get_transforms(self.height, self.width, split)
        if split == "train":
            assert cfg.DATA.BATCH_SIZE <= cfg.DATA.LABELED_SIZE, "Batch size must be smaller than train size."
            self.annotations = self.annotations.sample(frac=1).reset_index(drop=True)
            self.train_annotations = self.annotations[:cfg.DATA.LABELED_SIZE]
            

    def __len__(self) -> int:
        return self.annotations.shape[0] if self.split != 'train' else self.train_annotations.shape[0]

    def __getitem__(self, index: int) -> None:
        annotations = self.annotations if self.split != 'train' else self.train_annotations
        annotation = annotations.iloc[index]
        image = Image.open(self.data_path.parent / annotation['Path'])
        classes = annotation[['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion']].values.astype("float32")
        data = self.transforms(image)
        return data.repeat(3, 1, 1), torch.from_numpy(classes)


class ChexpertDatasetUnlabeled(Dataset):
    def __init__(self, csv_path: Path, shuffled_annotations: pd.DataFrame, model) -> None:
        super(ChexpertDatasetUnlabeled, self).__init__()
        self.data_path = Path(csv_path).parent
        labeled_size = cfg.DATA.LABELED_SIZE
        unlabeled_size = cfg.DATA.UNLABELED_SIZE
        self.labeled = shuffled_annotations[:labeled_size]
        self.annotations = shuffled_annotations[labeled_size:labeled_size + unlabeled_size].reset_index(drop=True)
        self.S = []
        self.height, self.width = 224, 224
        self.transforms = get_transforms(self.height, self.width, 'train')

        # model = model.cuda() 
        self.assign_nearest(model)
    
    def assign_nearest(self, fe):
        unlabeled_paths, labeled_paths = list(self.annotations["Path"]), list(self.labeled["Path"])
        ftr_extractor = np.vectorize(lambda path: fe.extract_feature(self.transforms(Image.open(self.data_path.parent / path)).repeat(3, 1, 1)))
        unlabeled_ftrs = ftr_extractor(unlabeled_paths)
        labeled_ftrs = ftr_extractor(labeled_paths)
            
        for i, unlabel in enumerate(unlabeled_ftrs):
            dist = torch.norm(labeled_ftrs - unlabel, dim=1, p=None)
            nearest_idx = torch.argmin(dist, axis=0)
            nearest = labeled_ftrs[nearest_idx]
            print(type(nearest), type(unlabel))
            print(nearest)
            print(unlabel)
            s_i = np.exp(F.normalize(torch.norm(nearest - unlabel, dim=1, p=None)))
            self.S.append(nearest)
            print(self.annotations.iloc[i][['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion']])
            print(self.labeled.iloc[nearest_idx][['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion']])
            self.annotations.iloc[i]["Atelectasis"] = self.labeled.iloc[nearest_idx]["Atelectasis"]
            self.annotations.iloc[i]["Cardiomegaly"] = self.labeled.iloc[nearest_idx]["Cardiomegaly"]
            self.annotations.iloc[i]["Consolidation"] = self.labeled.iloc[nearest_idx]["Consolidation"]
            self.annotations.iloc[i]["Edema"] = self.labeled.iloc[nearest_idx]["Edema"]
            self.annotations.iloc[i]["Pleural Effusion"] = self.labeled.iloc[nearest_idx]["Pleural Effusion"]
            print(self.annotations.iloc[i][['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion']])
            exit()

    def __len__(self) -> int:
        return self.annotations.shape[0]

    def __getitem__(self, index: int) -> None:
        annotation = self.annotations.iloc[index]
        image = Image.open(self.data_path.parent / annotation['Path'])
        classes = annotation[['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion']].values.astype("float32")
        data = self.transforms(image)
        return data.repeat(3, 1, 1), torch.from_numpy(classes)


def build_dataloader(split, model):
    valid_splits = ["train", "valid", "test"]
    assert split in valid_splits, f"{split} should be one of {valid_splits}."

    ds_path = Path(cfg.DATA.PATH)
    bs = cfg.DATA.BATCH_SIZE

    is_train = split == 'train'
    dataset = ChexpertDataset(ds_path / f"{split}.csv", split)
    dl_labeled = DataLoader(dataset, batch_size=bs, num_workers=min(os.cpu_count(), 12), shuffle=is_train)
    dl_unlabeled = None
    if split == 'train':
        dataset_u = ChexpertDatasetUnlabeled(ds_path / f'{split}.csv', dataset.annotations, model)
        dl_unlabeled = DataLoader(dataset_u, batch_size=int(bs), num_workers=min(os.cpu_count(), 12), shuffle=is_train)
    return dl_labeled, dl_unlabeled