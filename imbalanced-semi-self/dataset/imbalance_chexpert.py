import os, sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import numpy as np
import pandas as pd
import random

import torch
import PIL.Image as Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import scipy
import pickle

from utils.long_tail_config import _C as cfg
from utils.transforms import get_transforms

def convert_label(label):
    str_repr = "".join(map(str, np.array(label, dtype=np.int8)))
    num_repr = int(str_repr, 2)
    return num_repr

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

class ChexpertDataset(Dataset):
    def __init__(self, csv_path: Path, split: str, labeled=True, rand_number=543) -> None:
        super(ChexpertDataset, self).__init__()
        set_seed(rand_number)
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
            if labeled:
                self.annotations = self.annotations[:cfg.DATA.LABELED_SIZE] 
            else:
                self.annotations = self.annotations[cfg.DATA.LABELED_SIZE:cfg.DATA.LABELED_SIZE + cfg.DATA.UNLABELED_SIZE].reset_index(drop=True) 

    def __len__(self) -> int:
        return self.annotations.shape[0]

    def __getitem__(self, index: int) -> None:
        annotations = self.annotations
        annotation = annotations.iloc[index]
        image = Image.open(self.data_path.parent / annotation['Path'])
        classes = annotation[['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion']].values.astype("float32")
        data = self.transforms(image)
        classes = convert_label(classes)
        label = np.zeros(32)
        label[classes] = 1
        return data.repeat(3, 1, 1), torch.Tensor(label)


class ImbalanceChexpert(Dataset):
    def __init__(self, csv_path: Path, split: str, imb_type="exp", imb_factor=0.01, rand_number=543):
        super(ImbalanceChexpert, self).__init__()
        set_seed(rand_number)
        self.data_path = Path(csv_path).parent
        self.annotations = pd.read_csv(csv_path).fillna(0)
        self.split = split
        self.height, self.width = 224, 224
        self.transforms = get_transforms(self.height, self.width, split)
        self.assign_labels()
        
        if split == "train":
            assert cfg.DATA.BATCH_SIZE <= cfg.DATA.LABELED_SIZE, "Batch size must be smaller than train size."
            self.annotations = self.annotations.sample(frac=1).reset_index(drop=True)
            self.annotations = self.annotations[:cfg.DATA.LABELED_SIZE] # split to be "labelled data"

        img_num_list = self.get_img_num_per_cls(32, imb_type, imb_factor)
        self.gen_imbalanced_data(img_num_list)

    def __len__(self) -> int:
        return self.annotations.shape[0]

    def __getitem__(self, index: int) -> None:
        annotation = self.annotations.iloc[index]
        image = Image.open(self.data_path.parent / annotation['Path'])
        classes = annotation[["Label"]].values.astype("float32")
        data = self.transforms(image)
        return data.repeat(3, 1, 1), torch.from_numpy(classes)

    def assign_labels(self):
        classes = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion']
        df = self.annotations[classes].copy()
        self.annotations["Label"] = df.apply(convert_label, axis=1, result_type="reduce")
    
    def get_img_num_per_cls(self, cls_num, imb_type, imb_factor):
        img_max = len(self.annotations) / cls_num
        img_num_per_cls = []
        if imb_type == 'exp':
            for cls_idx in range(cls_num):
                num = img_max * (imb_factor**(cls_idx / (cls_num - 1.0)))
                img_num_per_cls.append(int(num))
        elif imb_type == 'step':
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max))
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max * imb_factor))
        else:
            img_num_per_cls.extend([int(img_max)] * cls_num)
        return img_num_per_cls

    def gen_imbalanced_data(self, img_num_per_cls):
        new_data = []
        new_targets = []
        targets_np = np.array(self.annotations["Label"], dtype=np.int64)
        classes = np.unique(targets_np)
        self.num_per_cls_dict = dict()
        for the_class, the_img_num in zip(classes, img_num_per_cls):
            self.num_per_cls_dict[the_class] = the_img_num
            idx = np.where(targets_np == the_class)[0]
            print(f"Class {the_class}:\t{len(idx)}")
            np.random.shuffle(idx)
            selec_idx = idx[:the_img_num]
            new_data.append(self.annotations.iloc[selec_idx])
        new_data = pd.concat(new_data)
        self.annotations = new_data

class SemiSupervisedImbalanceChexpert(Dataset):
    def __init__(self, csv_path: Path, unlabeled_pseudo: str, split="train", imb_type="exp", imb_factor=0.01, unlabel_imb_factor=1, rand_number=543):
        super(SemiSupervisedImbalanceChexpert, self).__init__()
        # unlabeled
        set_seed(rand_number)
        self.data_path = Path(csv_path).parent
        self.annotations = pd.read_csv(csv_path).fillna(0)
        self.split = split
        self.height, self.width = 224, 224
        self.transforms = get_transforms(self.height, self.width, split)
        self.assign_labels()

        if split == "train":
            assert cfg.DATA.BATCH_SIZE <= cfg.DATA.LABELED_SIZE, "Batch size must be smaller than train size."
            self.labeled_annotations = self.annotations.sample(frac=1).reset_index(drop=True)
            self.labeled_annotations = self.labeled_annotations[:cfg.DATA.LABELED_SIZE] 
            self.unlabeled_annotations = self.annotations[cfg.DATA.LABELED_SIZE:cfg.DATA.LABELED_SIZE + cfg.DATA.UNLABELED_SIZE].reset_index(drop=True)
        
        self.cls_num = 32
        self.unlabel_size_factor = 5
        self.unlabeled_pseudo = unlabeled_pseudo  # pseudo-labels using model trained on imbalanced data
        self.imb_factor = imb_factor
        self.unlabel_imb_factor = unlabel_imb_factor
        self.num_per_cls_dict = dict()
        img_num_list = self.get_img_num_per_cls(self.cls_num, imb_type, imb_factor)
        img_num_list_unlabeled = self.get_img_num_per_cls_unlabeled(self.cls_num, img_num_list, unlabel_imb_factor)
        self.gen_imbalanced_data(img_num_list, img_num_list_unlabeled)

    def __len__(self) -> int:
        return self.annotations.shape[0]

    def __getitem__(self, index: int) -> None:
        annotation = self.annotations.iloc[index]
        image = Image.open(self.data_path.parent / annotation['Path'])
        classes = annotation[["Label"]].values.astype("float32")
        data = self.transforms(image)
        return data.repeat(3, 1, 1), torch.from_numpy(classes)
    
    def assign_labels(self):
        classes = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion']
        df = self.annotations[classes].copy()
        self.annotations["Label"] = df.apply(convert_label, axis=1, result_type="reduce")

    def get_img_num_per_cls(self, cls_num, imb_type, imb_factor):
        img_max = len(self.labeled_annotations) / cls_num
        img_num_per_cls = []
        if imb_type == 'exp':
            for cls_idx in range(cls_num):
                num = img_max * (imb_factor**(cls_idx / (cls_num - 1.0)))
                img_num_per_cls.append(int(num))
        elif imb_type == 'step':
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max))
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max * imb_factor))
        else:
            img_num_per_cls.extend([int(img_max)] * cls_num)
        return img_num_per_cls

    def get_img_num_per_cls_unlabeled(self, cls_num, labeled_img_num_list, imb_factor):
        img_unlabeled_total = np.sum(labeled_img_num_list) * self.unlabel_size_factor
        img_first_min = img_unlabeled_total // cls_num
        img_num_per_cls_unlabel = []
        for cls_idx in range(cls_num):
            num = img_first_min * (imb_factor**(cls_idx / (cls_num - 1.0)))
            img_num_per_cls_unlabel.append(int(num))
        factor = img_unlabeled_total / np.sum(img_num_per_cls_unlabel)
        img_num_per_cls_unlabel = [int(num * factor) for num in img_num_per_cls_unlabel]
        print(f"Unlabeled est total:\t{img_unlabeled_total}\n"
              f"After processing:\t{np.sum(img_num_per_cls_unlabel)},\t{img_num_per_cls_unlabel}")
        return img_num_per_cls_unlabel

    def gen_imbalanced_data(self, img_num_per_cls, img_num_per_cls_unlabeled):
        new_data = []
        targets_np = np.array(self.annotations["Label"], dtype=np.int64)
        classes = np.unique(targets_np)
        for the_class, the_img_num in zip(classes, img_num_per_cls):
            self.num_per_cls_dict[the_class] = the_img_num
            idx = np.where(targets_np == the_class)[0]
            np.random.shuffle(idx)
            selec_idx = idx[:the_img_num]
            new_data.append(self.annotations.iloc[selec_idx])

        # unlabeled data
        print("Loading pseudo labels from %s" % self.unlabeled_pseudo)
        with open(self.unlabeled_pseudo, 'rb') as f:
            aux_targets = pickle.load(f)
        aux_targets = aux_targets['extrapolated_targets']

        for the_class, the_img_num in zip(classes, img_num_per_cls_unlabeled):
            # ground truth is only used to select samples
            idx = np.where(self.unlabeled_annotations["Label"] == the_class)[0]
            np.random.shuffle(idx)
            selec_idx = idx[:the_img_num]
            new_data.append(self.unlabeled_annotations.iloc[selec_idx])
            new_data[-1]["Label"].replace(aux_targets[selec_idx], inplace=True)
            for pseudo_class in aux_targets[selec_idx]:
                self.num_per_cls_dict[pseudo_class] += 1
        self.annotations = pd.concat(new_data)

    def get_cls_num_list(self):
        cls_num_list = []
        for i in range(self.cls_num):
            cls_num_list.append(self.num_per_cls_dict[i])
        return cls_num_list

if __name__ == "__main__":
    ds_path = Path(cfg.DATA.PATH)
    dataset = SemiSupervisedImbalanceChexpert(ds_path / "train.csv", "train")
    print(dataset[0])