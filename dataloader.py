import numpy as np
import pandas as pd
from pathlib import Path

import torch
import PIL.Image as Image
from torch.utils.data import Dataset

class ChexpertDataset(Dataset):
    def __init__(self, csv_path: Path) -> None:
        super(ChexpertDataset, self).__init__()
        self.csv_path = csv_path
        self.annotations = pd.read_csv(csv_path)

        # Filter out rows which have uncertain annotations (-1)
        self.annotations = self.annotations.iloc[~np.any(self.annotations.values[:, 5:] < 0, axis=1)]

    def __len__(self) -> int:
        return self.annotations.shape[0]

    def __getitem__(self, index: int) -> None:
        annotation = self.annotations.iloc[index]
        image = np.asarray(Image.open(Path("data") / annotation['Path']))
        print(image)



ds = ChexpertDataset('data/CheXpert-v1.0-small/valid.csv')
print(len(ds))
ds[0]
