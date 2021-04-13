import argparse
import numpy as np
import pandas as pd
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--path", "-p", type=Path, required=True, help="Path to CheXpert dataset.")
args = ap.parse_args()

classes = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion']
train, val, test = [pd.read_csv(args.path/p) for p in ["train.csv", "valid.csv", "test.csv"]]

counts = []
for spl in ["train", "val", "test"]:
    df = eval(spl)
    c = [spl]
    df["Patient"] = df["Path"].apply(lambda x: int(x.split("/")[-3].split("patient")[-1]))
    c.append(np.unique(df["Patient"]).shape[0])
    c.append(df.shape[0])
    for cls in classes:
        c.append((df[cls] == 1).sum())
    counts.append(c)

df = pd.DataFrame(counts, columns=["Dataset", "Patients", "Images"] + classes)
print(df)
