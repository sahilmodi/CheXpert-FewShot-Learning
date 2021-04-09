import os
import pickle
import numpy as np
import torch
from dataset.dataloader import build_dataloader

def convert_label(label):
    str_repr = "".join(map(str, label.numpy()))
    num_repr = int(str_repr, 2)
    return num_repr

def get_data(cfg, phase):
    if phase == "trainval":
        dl = chain(build_dataloader(cfg, "train"), build_dataloader(cfg, "valid"))
    elif phase == "val":
        dl = build_dataloader(cfg, "valid")        
    else:
        dl = build_dataloader(cfg, phase)

    data = {"data": [], "labels": []}

    for ims, labels in dl:
        labels = list(map(convert_label, labels))
        data["data"].extend(ims.data.cpu().numpy())
        data["labels"].extend(labels)

    data["data"] = np.array(data["data"])
    return data

