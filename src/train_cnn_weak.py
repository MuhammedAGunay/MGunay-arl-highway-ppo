from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from .cnn_policy import SmallGridCNN
from .config import ConfigBundle
from .dataset import GridPolicyDataset, load_dataset_npz


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _take_fraction(idxs: np.ndarray, frac: float, seed: int) -> np.ndarray:
    frac = float(np.clip(frac, 0.0, 1.0))
    rng = np.random.default_rng(seed)
    idxs = idxs.copy()
    rng.shuffle(idxs)
    n = max(1, int(round(len(idxs) * frac)))
    return idxs[:n]


def train_weak(
    frac_train: float = 0.10,
    epochs: int = 3,
    batch_size: int = 256,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    out_name: str = "cnn_weak.pt",
) -> None:
    cfg = ConfigBundle()
    _set_seed(cfg.train.seed)

    x, y, _density_bin, splits = load_dataset_npz(cfg.data.dataset_npz_path)
    ds = GridPolicyDataset(x, y)

    train_idx_full = splits["train"]
    val_idx = splits["val"]

    train_idx = _take_fraction(train_idx_full, frac_train, seed=cfg.train.seed + 999)

    train_ds = Subset(ds, train_idx.tolist())
    val_ds = Subset(ds, val_idx.tolist())

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    device = torch.device(cfg.train.device if torch.cuda.is_available() else "cpu")

    model = SmallGridCNN(
        n_actions=4,
        features_dim=cfg.cnn.hidden_dim,
        kernel_size=int(cfg.cnn.kernel_size),
        stride=1,
        padding=int(cfg.cnn.padding),
        padding_mode=str(cfg.cnn.padding_mode),
        wall_padding=bool(cfg.cnn.wall_padding),
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val = -1.0
    out_dir = Path(cfg.train.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / out_name

    print(f"[weak] dataset={cfg.data.dataset_npz_path}")
    print(f"[weak] train_full={len(train_idx_full)} train_used={len(train_idx)} val={len(val_idx)}")
    print(f"[weak] epochs={epochs} frac_train={frac_train}")

    for ep in range(1, epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            logits = model(xb)
            loss = F.cross_entropy(logits, yb)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                pred = torch.argmax(model(xb), dim=1)
                correct += int((pred == yb).sum().item())
                total += int(yb.numel())

        val_acc = correct / max(total, 1)
        print(f"[weak epoch {ep}/{epochs}] val_acc={val_acc:.4f}")

        if val_acc > best_val:
            best_val = val_acc
            torch.save(
                {
                    "model": model.state_dict(),
                    "cfg": cfg.cnn.__dict__,
                    "meta": {"type": "weak", "frac_train": frac_train, "epochs": epochs},
                },
                str(out_path),
            )

    print(f"Saved WEAK model: {out_path} (best_val={best_val:.4f})")


if __name__ == "__main__":
    train_weak(frac_train=0.10, epochs=3, out_name="cnn_weak.pt")
