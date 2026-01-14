from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.config import TrainConfig


def main() -> None:
    cfg = TrainConfig()
    project_root = Path(__file__).resolve().parents[1]

    monitor_path = project_root / cfg.log_dir / "monitor.csv"
    out_path = project_root / cfg.assets_dir / "reward_curve.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(monitor_path, comment="#")
    df["episode"] = range(1, len(df) + 1)

    plt.figure()
    plt.plot(df["episode"], df["r"])
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.title("Return vs Episode")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

    print(f"Saved plot to: {out_path}")


if __name__ == "__main__":
    main()
