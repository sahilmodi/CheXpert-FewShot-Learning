import argparse
import numpy as np
import pandas as pd
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--path", "-p", type=Path, required=True, help="Path to CheXpert dataset.")
ap.add_argument("--val_ratio", "-r", default=0.15, type=float, help="Percentage of train set to hold for validation.")
args = ap.parse_args()

np.random.seed(543)

print("Splitting into train, val, test sets.")

old_csv_path = args.path / "train.csv"
train_csv_path = args.path / "train_full.csv"

if not old_csv_path.exists():
    print("- [ERROR] Invalid dataset format. Abort")
    exit()

if train_csv_path.exists():
    print("- [ERROR] Dataset is already split. Abort.")
    exit()
    
old_csv_path.replace(train_csv_path)
(args.path / "valid.csv").replace(args.path / "test.csv")

train_df = pd.read_csv(train_csv_path)
train_df["Patient"] = train_df["Path"].apply(lambda x: int(x.split("/")[-3].split("patient")[-1]))
train_df["Patient"] -= train_df["Patient"].iloc[0]
patients_bc = np.bincount(train_df["Patient"])
assert np.count_nonzero(patients_bc) == patients_bc.shape[0], "Patient IDs are not consecutive. Abort."
permutation = np.random.permutation(patients_bc.shape[0])
patients = patients_bc[permutation]
patients_cs = np.cumsum(patients)

target_val_amount = int(train_df.shape[0] * args.val_ratio)
print("- Looking for approximately", target_val_amount, "images.")
idx = np.abs(patients_cs - target_val_amount).argmin()
count = patients_cs[idx]
if target_val_amount*0.95 <= count <= target_val_amount*1.05:
    print("- Found a valid split containing", count, "validation images.")

valid_patients_ids = permutation[:idx+1]
val_df = train_df[train_df["Patient"].isin(valid_patients_ids)]
train_df = train_df[~train_df["Patient"].isin(valid_patients_ids)]
del train_df["Patient"]
del val_df["Patient"]

print("- Filtering out rows which have uncertain labels (-1)")
with np.errstate(invalid='ignore'):
    train_df = train_df.iloc[~np.any(train_df.values[:, 5:] < 0, axis=1)]
    val_df = val_df.iloc[~np.any(val_df.values[:, 5:] < 0, axis=1)]

train_df.to_csv(old_csv_path, index=False)
val_df.to_csv(args.path / "valid.csv", index=False)

print("* Train Images:\t", train_df.shape[0])
print("* Val Images:\t", val_df.shape[0])
