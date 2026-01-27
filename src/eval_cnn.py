from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .cnn_policy import SmallGridCNN
from .config import ConfigBundle
from .dataset import load_dataset_npz


def _load_state_dict(ckpt: object) -> dict:
    if isinstance(ckpt, dict) and "model" in ckpt:
        return ckpt["model"]
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        return ckpt["state_dict"]
    if isinstance(ckpt, dict):
        # fallback: maybe the dict itself is a state_dict
        return ckpt
    raise ValueError("Unsupported checkpoint format.")


def main(cfg: ConfigBundle) -> None:
    device = torch.device(cfg.train.device if torch.cuda.is_available() else "cpu")

    ckpt_path = Path(cfg.train.out_dir) / cfg.train.out_name
    ckpt = torch.load(str(ckpt_path), map_location=device)
    state_dict = _load_state_dict(ckpt)

    model = SmallGridCNN(
        n_actions=4,
        features_dim=cfg.cnn.hidden_dim,
        kernel_size=int(cfg.cnn.kernel_size),
        stride=1,
        padding=int(cfg.cnn.padding),
        padding_mode=str(cfg.cnn.padding_mode),
        wall_padding=bool(cfg.cnn.wall_padding),
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    x, y, _density_bin, splits = load_dataset_npz(cfg.data.dataset_npz_path)
    test_idx = splits["test"].astype(np.int64)

    x_test = torch.from_numpy(x[test_idx]).to(device)
    y_test = torch.from_numpy(y[test_idx]).to(device)

    with torch.no_grad():
        pred = torch.argmax(model(x_test), dim=1)

    acc = float((pred == y_test).float().mean().item())
    print(f"test_acc: {acc:.4f}")

    conf = np.zeros((4, 4), dtype=np.int64)
    yt = y_test.detach().cpu().numpy()
    yp = pred.detach().cpu().numpy()

    for t, p in zip(yt, yp):
        conf[int(t), int(p)] += 1

    print("confusion (true rows, pred cols) actions: 0FWD 1BWD 2LEFT 3RIGHT")
    print(conf)
    print("true action counts:", conf.sum(axis=1))
    print("pred action counts:", conf.sum(axis=0))


if __name__ == "__main__":
    main(ConfigBundle())
