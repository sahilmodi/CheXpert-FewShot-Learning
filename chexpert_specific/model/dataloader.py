import os
import pandas as pd
from pathlib import Path

import torch
import PIL.Image as Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.transforms import Normalize, ToPILImage

from utils.config import _C as cfg

class ChexpertDataset(Dataset):
    def __init__(self, csv_path: Path, data_augmentation=False) -> None:
        super(ChexpertDataset, self).__init__()
        self.data_path = Path(csv_path).parent
        self.annotations = pd.read_csv(csv_path).fillna(0)
        self.transforms = None
        self.height, self.width = 224, 224
        self.transforms = transforms.Compose([
                transforms.Resize((self.height, self.width)),
                transforms.ToTensor()
            ])
        if data_augmentation:
            self.transforms = transforms.Compose([
                transforms.RandomAffine(
                    degrees=(-15, 15),
                    translate=(0.05, 0.05),
                    scale=(0.95, 1.05)
                ),
                transforms.Resize((self.height, self.width)),
                transforms.ToTensor(),
                transforms.Normalize(128, 64),
                transforms.ToPILImage(),
                transforms.Lambda(lambda x: transforms.functional.equalize(x)),
                transforms.ToTensor(),
            ])

    def __len__(self) -> int:
        return self.annotations.shape[0]

    def __getitem__(self, index: int) -> None:
        annotation = self.annotations.iloc[index]
        image = Image.open(self.data_path.parent / annotation['Path'])
        classes = annotation[['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion']].values.astype("float32")
        data = self.transforms(image)
        return data.repeat(3, 1, 1), torch.from_numpy(classes)


def build_dataloader(split):
    valid_splits = ["train", "valid", "test"]
    assert split in valid_splits, f"{split} should be one of {valid_splits}."

    ds_path = Path(cfg.DATA.PATH)
    bs = cfg.DATA.BATCH_SIZE

    dataset = ChexpertDataset(ds_path / f"{split}.csv", data_augmentation=split=="train")
    return DataLoader(dataset, batch_size=bs, num_workers=min(os.cpu_count(), 12), shuffle=split == "train")