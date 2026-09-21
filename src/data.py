"""Torch dataset and datamodule for one leave-one-subject-out fold."""
import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset

from src.augment import channel_rotation


class WindowDataset(Dataset):
    def __init__(self, emg, imu, label, rotation=None, seed=0):
        self.emg, self.imu, self.label = emg, imu, label
        self.rotation = rotation
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, i):
        emg = self.emg[i]
        if self.rotation is not None:
            emg = channel_rotation(emg, self.label[i], self.rng, **self.rotation)
        return {"emg": torch.from_numpy(np.ascontiguousarray(emg)),
                "imu": torch.from_numpy(self.imu[i]),
                "label": int(self.label[i])}


def stratified_split(y, val_ratio, seed):
    rng = np.random.default_rng(seed)
    val = []
    for c in np.unique(y):
        idx = np.flatnonzero(y == c)
        rng.shuffle(idx)
        val.append(idx[:max(1, int(round(val_ratio * len(idx))))])
    val = np.concatenate(val)
    train = np.setdiff1d(np.arange(len(y)), val)
    return train, val


class FoldDataModule(pl.LightningDataModule):
    def __init__(self, arrays, batch_size=128, val_ratio=0.2, num_workers=4, seed=42,
                 rotation=None):
        super().__init__()
        self.a, self.batch_size, self.val_ratio = arrays, batch_size, val_ratio
        self.num_workers, self.seed, self.rotation = num_workers, seed, rotation

    def setup(self, stage=None):
        tr, va = stratified_split(self.a.y_src, self.val_ratio, self.seed)
        self.train_set = WindowDataset(self.a.emg_src[tr], self.a.imu_src[tr], self.a.y_src[tr],
                                       self.rotation, self.seed)
        self.val_set = WindowDataset(self.a.emg_src[va], self.a.imu_src[va], self.a.y_src[va])

    def train_dataloader(self):
        return DataLoader(self.train_set, self.batch_size, shuffle=True,
                          num_workers=self.num_workers, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.val_set, self.batch_size, num_workers=self.num_workers)
