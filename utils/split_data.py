import argparse
import numpy as np
import pandas as pd
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--path", "-p", type=Path, required=True, help="Path to CheXpert dataset.")
ap.add_argument("--val_ratio", "-r", default=0.2, type=float, help="Percentage of train set to hold for validation.")
args = ap.parse_args()

# old_csv_path = args.path / "train.csv"
train_csv_path = args.path / "train_full.csv"
# old_csv_path.replace(train_csv_path)

train_csv = pd.read_csv(train_csv_path)
train_csv["Patient"] = train_csv["Path"].apply(int(lambda x: x.split("/")[-3].split("patient")[-1]))
train_csv["Patient"] -= train_csv["Patient"].iloc[0]
patients = np.bincount(train_csv["Patient"])
print(patients)
target_val_amount = int(train_csv.shape[0] * args.val_ratio)
print("Looking for approximately", target_val_amount, "images.")
while True:
    permutation = np.random.permutation(patients)
    count = 0
    for patient in patients:
        print(count)
        count += train_csv[train_csv["Patient"] == patient].shape[0]
        if target_val_amount*0.95 <= count <= target_val_amount*1.05:
            break
    print("Found a valid split containing", count, "validation images.") 
print(train_csv.head())
