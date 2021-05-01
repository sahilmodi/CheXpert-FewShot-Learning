import os
import numpy as np
import pandas as pd
from pathlib import Path

import torch
import PIL.Image as Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

from .view_generator import ContrastiveLearningViewGenerator
from .gaussian_blur import GaussianBlur

class ChexpertDataset(Dataset):
    def __init__(self, root_dir: Path, size: str, split="train") -> None:
        super(ChexpertDataset, self).__init__()
        csv_path = Path(root_dir) / f"{split}.csv"
        self.data_path = Path(csv_path).parent
        self.annotations = pd.read_csv(csv_path).fillna(0)
        self.split = split
        self.transforms = None
        self.height, self.width = size, size
        self.transforms = ContrastiveLearningViewGenerator(self.get_simclr_pipeline_transform(size), 2)
        self.annotations = self.annotations.sample(frac=1).reset_index(drop=True)[:15000]

    @staticmethod
    def get_simclr_pipeline_transform(size, s=1):
        """Return a set of data augmentation transformations as described in the SimCLR paper."""
        color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
        data_transforms = transforms.Compose([
                                transforms.RandomResizedCrop(size=size),
                                transforms.RandomHorizontalFlip(),
                                transforms.RandomApply([color_jitter], p=0.8),
                                transforms.RandomGrayscale(p=0.2),
                                GaussianBlur(kernel_size=int(0.1 * size)),
                                transforms.ToTensor()
                        ])
        return data_transforms       

    def __len__(self) -> int:
        return self.annotations.shape[0]

    def __getitem__(self, index: int) -> None:
        annotation = self.annotations.iloc[index]
        image = Image.open(self.data_path.parent / annotation['Path']).convert('RGB')
        classes = annotation[['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion']].values.astype("float32")
        data = self.transforms(image)
        return data, torch.from_numpy(classes)