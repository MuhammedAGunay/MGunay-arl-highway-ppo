from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class GridPolicyDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray):
        if x.ndim != 4 or x.shape[1] != 2:
            raise ValueError("x must be (N,2,H,W)")
        if y.ndim != 1 or y.shape[0] != x.shape[0]:
            raise ValueError("y must be (N,) and match x")

        self.x = torch.from_numpy(x.astype(np.float32, copy=False)).float()
        self.y = torch.from_numpy(y.astype(np.int64, copy=False)).long()

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]


def stratified_split_indices(
    y: np.ndarray,
    density_bin: np.ndarray | None,
    val_frac: float,
    test_frac: float,
    seed: int,
    stratify_by_density: bool = True,
) -> Dict[str, np.ndarray]:
    if not (0.0 <= val_frac < 1.0):
        raise ValueError("val_frac must be in [0,1).")
    if not (0.0 <= test_frac < 1.0):
        raise ValueError("test_frac must be in [0,1).")
    if (val_frac + test_frac) >= 1.0:
        raise ValueError("val_frac + test_frac must be < 1.0")

    rng = np.random.default_rng(seed)
    y = np.asarray(y, dtype=np.int64)

    if stratify_by_density and density_bin is not None:
        db = np.asarray(density_bin, dtype=np.int64)
        keys = y * 1000 + db
    else:
        keys = y

    train_parts = []
    val_parts = []
    test_parts = []

    for k in np.unique(keys):
        idx = np.where(keys == k)[0]
        rng.shuffle(idx)

        n = len(idx)
        n_test = int(round(n * test_frac))
        n_val = int(round(n * val_frac))

        if n >= 3 and test_frac > 0 and n_test == 0:
            n_test = 1
        if n >= 3 and val_frac > 0 and n_val == 0:
            n_val = 1

        if n_test + n_val > n:
            n_val = max(0, n - n_test)

        test_parts.append(idx[:n_test])
        val_parts.append(idx[n_test : n_test + n_val])
        train_parts.append(idx[n_test + n_val :])

    train_idx = np.concatenate(train_parts) if train_parts else np.array([], dtype=np.int64)
    val_idx = np.concatenate(val_parts) if val_parts else np.array([], dtype=np.int64)
    test_idx = np.concatenate(test_parts) if test_parts else np.array([], dtype=np.int64)

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)

    return {"train": train_idx, "val": val_idx, "test": test_idx}


def save_dataset_npz(
    path: str | Path,
    x: np.ndarray,
    y: np.ndarray,
    density_bin: np.ndarray,
    splits: Dict[str, np.ndarray],
) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(p),
        x=x.astype(np.float32),
        y=y.astype(np.int64),
        density_bin=density_bin.astype(np.int64),
        train_idx=splits["train"].astype(np.int64),
        val_idx=splits["val"].astype(np.int64),
        test_idx=splits["test"].astype(np.int64),
    )


def load_dataset_npz(path: str | Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    d = np.load(str(path))
    x = d["x"]
    y = d["y"]
    density_bin = d["density_bin"]
    splits = {"train": d["train_idx"], "val": d["val_idx"], "test": d["test_idx"]}
    return x, y, density_bin, splits
