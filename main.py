from utils.config import _C as cfg
from model.dataloader import build_dataloader 

cfg.merge_from_file("config/sample.yaml")
cfg.freeze()

val_dl = build_dataloader(cfg.DATA, "valid", data_augmentation=False)
for data, labels in val_dl:
    print(data.shape)
    print(labels.shape)
    print(labels[0])
    exit()
