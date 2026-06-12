import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from pathlib import Path


class BangliaDataset(Dataset):
    def __init__(self, records: list, snp_features: dict | None = None,
                 augment: bool = False):
        self.records = records
        self.snp_features = snp_features  # {subject_id: [f1, f2, ...]}
        self.augment = augment
        self.n_snps = len(next(iter(snp_features.values()))) if snp_features else 0

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        sid = rec["subject_id"]

        x = torch.from_numpy(np.load(rec["path"]).astype(np.float32))
        if self.augment:
            x = self._augment(x)

        label = torch.tensor(rec["label"], dtype=torch.float32)

        if self.snp_features and sid in self.snp_features:
            snp = torch.tensor(self.snp_features[sid], dtype=torch.float32)
        else:
            snp = torch.zeros(self.n_snps, dtype=torch.float32)

        return x, snp, label, sid

    def _augment(self, x: torch.Tensor) -> torch.Tensor:
        if torch.rand(1) > 0.5:
            x = torch.flip(x, dims=[1])
        if torch.rand(1) > 0.5:
            x = torch.flip(x, dims=[2])
        x = x + torch.randn_like(x) * 0.01
        return x


def load_snp_features(snp_path: str) -> dict:
    with open(snp_path) as f:
        data = json.load(f)
    return data["subjects"]


def make_splits(manifest_path: str, n_folds: int = 5, seed: int = 42):
    with open(manifest_path) as f:
        records = json.load(f)

    labels = [r["label"] for r in records]
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds = []
    for train_idx, val_idx in skf.split(records, labels):
        folds.append({
            "train": [records[i] for i in train_idx],
            "val":   [records[i] for i in val_idx],
        })

    splits_path = Path(manifest_path).parent / "splits.json"
    with open(splits_path, "w") as f:
        json.dump(folds, f, indent=2)

    print(f"Splits salvos: {splits_path}")
    print(f"Fold 0 — train: {len(folds[0]['train'])}, val: {len(folds[0]['val'])}")
    return folds


def get_dataloaders(fold: dict, snp_features: dict | None = None,
                    batch_size: int = 8,
                    num_workers: int = 4) -> tuple[DataLoader, DataLoader]:
    train_ds = BangliaDataset(fold["train"], snp_features, augment=True)
    val_ds   = BangliaDataset(fold["val"],   snp_features, augment=False)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          num_workers=num_workers, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                          num_workers=num_workers, pin_memory=True)
    return train_dl, val_dl
