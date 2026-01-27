from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

from .cnn_policy import SmallGridCNN
from .config import ConfigBundle
from .dataset import GridPolicyDataset, save_dataset_npz, stratified_split_indices
from .solve_generate import build_dataset


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_cnn(cfg: ConfigBundle) -> None:
    _set_seed(cfg.train.seed)

    print("[1/3] Building dataset...")
    x, y, density_bin = build_dataset(cfg)
    print(f"Dataset built: N={len(y)}")

    print("[2/3] Splitting train/val/test...")
    splits = stratified_split_indices(
        y=y,
        density_bin=density_bin,
        val_frac=cfg.data.val_frac,
        test_frac=cfg.data.test_frac,
        seed=cfg.data.split_seed,
        stratify_by_density=cfg.data.stratify_by_density,
    )

    save_dataset_npz(cfg.data.dataset_npz_path, x, y, density_bin, splits)

    ds = GridPolicyDataset(x, y)
    train_ds = Subset(ds, splits["train"].tolist())
    val_ds = Subset(ds, splits["val"].tolist())

    train_loader = DataLoader(train_ds, batch_size=cfg.train.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.train.batch_size, shuffle=False)

    device = torch.device(cfg.train.device if torch.cuda.is_available() else "cpu")

    model = SmallGridCNN(
        n_actions=4,
        features_dim=cfg.cnn.hidden_dim,
        kernel_size=int(getattr(cfg.cnn, "kernel_size", 5)),
        stride=1,
        padding=int(getattr(cfg.cnn, "padding", 2)),
        padding_mode=str(getattr(cfg.cnn, "padding_mode", "zeros")),
        wall_padding=bool(getattr(cfg.cnn, "wall_padding", False)),
    ).to(device)

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.train.lr),
        weight_decay=float(cfg.train.weight_decay),
    )

    out_dir = Path(cfg.train.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / cfg.train.out_name

    best_val = -1.0

    print("[3/3] Training...")
    for epoch in range(1, cfg.train.epochs + 1):
        model.train()
        train_pbar = tqdm(
            train_loader,
            desc=f"train ep{epoch}/{cfg.train.epochs}",
            leave=False,
            dynamic_ncols=True,
        )

        run_loss = 0.0
        n_batches = 0
        for xb, yb in train_pbar:
            xb = xb.to(device)
            yb = yb.to(device)

            logits = model(xb)
            loss = F.cross_entropy(logits, yb)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            run_loss += float(loss.item())
            n_batches += 1
            train_pbar.set_postfix(loss=f"{run_loss / max(n_batches, 1):.4f}")

        model.eval()
        correct = 0
        total = 0
        val_pbar = tqdm(
            val_loader,
            desc=f"val ep{epoch}/{cfg.train.epochs}",
            leave=False,
            dynamic_ncols=True,
        )

        with torch.no_grad():
            for xb, yb in val_pbar:
                xb = xb.to(device)
                yb = yb.to(device)

                pred = torch.argmax(model(xb), dim=1)
                correct += int((pred == yb).sum().item())
                total += int(yb.numel())

                val_pbar.set_postfix(acc=f"{correct / max(total, 1):.4f}")

        val_acc = correct / max(total, 1)
        print(f"[epoch {epoch}/{cfg.train.epochs}] val_acc={val_acc:.4f}")

        if val_acc > best_val:
            best_val = val_acc
            torch.save({"model": model.state_dict(), "cfg": cfg.cnn.__dict__}, str(out_path))

    print(f"Saved: {out_path} (best_val={best_val:.4f})")
    print(f"Dataset saved: {cfg.data.dataset_npz_path} | N={len(ds)}")


if __name__ == "__main__":
    train_cnn(ConfigBundle())
