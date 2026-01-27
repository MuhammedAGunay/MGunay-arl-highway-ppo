from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np
import torch

from .cnn_policy import SmallGridCNN
from .gridworld import Act


@dataclass(frozen=True)
class MacroConfig:
    macro_len: int = 4

    a_lane_left: int = 0
    a_idle: int = 1
    a_lane_right: int = 2
    a_faster: int = 3
    a_slower: int = 4

    fwd_seq: Tuple[int, ...] = (3, 1, 1, 4)
    bwd_seq: Tuple[int, ...] = (4, 1, 1, 3)


def _safe_torch_load(path: str, device: str) -> Any:
    """Try torch.load with weights_only=True (newer PyTorch), fallback otherwise."""
    try:
        return torch.load(path, map_location=device, weights_only=True)  # type: ignore[call-arg]
    except TypeError:
        return torch.load(path, map_location=device)


def _extract_state_dict(ckpt: Any) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Return (state_dict, cfg_dict). cfg_dict may be empty."""
    if isinstance(ckpt, dict) and "model" in ckpt:
        state_dict = ckpt["model"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    elif isinstance(ckpt, dict):
        state_dict = ckpt
    else:
        raise ValueError("Unsupported checkpoint format.")

    cfg_dict: Dict[str, Any] = {}
    if isinstance(ckpt, dict) and isinstance(ckpt.get("cfg"), dict):
        cfg_dict = ckpt["cfg"]

    return state_dict, cfg_dict


class SystemPolicy:
    def __init__(self, model_path: str, device: str = "cpu") -> None:
        device_str = device if (device == "cpu" or torch.cuda.is_available()) else "cpu"
        self.device = torch.device(device_str)

        ckpt = _safe_torch_load(model_path, device=device_str)
        state_dict, cnn_cfg = _extract_state_dict(ckpt)

        self.model = SmallGridCNN(
            n_actions=4,
            features_dim=int(cnn_cfg.get("hidden_dim", 128)),
            kernel_size=int(cnn_cfg.get("kernel_size", 5)),
            stride=1,
            padding=int(cnn_cfg.get("padding", 2)),
            padding_mode=str(cnn_cfg.get("padding_mode", "zeros")),
            wall_padding=bool(cnn_cfg.get("wall_padding", False)),
        ).to(self.device)

        self.model.load_state_dict(state_dict)
        self.model.eval()

    def predict_from_obs_2hw(self, obs_2hw: np.ndarray) -> int:
        """obs_2hw: (2,4,7) float32 -> action (0..3)."""
        x = torch.from_numpy(obs_2hw[None]).to(self.device)
        with torch.no_grad():
            logits = self.model(x)
            return int(torch.argmax(logits, dim=1).item())

    def predict(self, obstacle_grid_4x7: np.ndarray, ego_row: int, ego_col: int = 3) -> int:
        """(4,7) obstacles + ego pos -> action (0..3)."""
        obs = np.zeros((2, 4, 7), dtype=np.float32)
        obs[0] = obstacle_grid_4x7.astype(np.float32)
        obs[1, int(ego_row), int(ego_col)] = 1.0
        return self.predict_from_obs_2hw(obs)


def obs_to_obstacles_4x7(obs: np.ndarray, ego_row: int, ego_col: int = 3) -> np.ndarray:
    """OccupancyGrid obs -> obstacles (4,7) uint8, with ego cell cleared."""
    if not isinstance(obs, np.ndarray) or obs.ndim != 3:
        raise ValueError(f"Expected obs ndarray with ndim=3, got {type(obs)} shape={getattr(obs,'shape',None)}")

    presence = obs[0]

    if presence.shape != (4, 7):
        if presence.T.shape == (4, 7):
            presence = presence.T
        else:
            raise ValueError(f"presence expected (4,7), got {presence.shape}")

    obstacles = (presence > 0.1).astype(np.uint8)
    obstacles[int(ego_row), int(ego_col)] = 0
    return obstacles


def act_to_meta_sequence(a: int, mc: MacroConfig) -> Tuple[int, ...]:
    aa = Act(int(a))
    if aa == Act.LEFT:
        return (mc.a_lane_left,)
    if aa == Act.RIGHT:
        return (mc.a_lane_right,)
    if aa == Act.FORWARD:
        seq = mc.fwd_seq
    elif aa == Act.BACKWARD:
        seq = mc.bwd_seq
    else:
        return (mc.a_idle,)

    if len(seq) < mc.macro_len:
        seq = seq + (mc.a_idle,) * (mc.macro_len - len(seq))
    return seq[: mc.macro_len]


def step_macro(
    env: Any,
    meta_seq: Tuple[int, ...],
) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
    total_r = 0.0
    terminated = False
    truncated = False
    last_obs: np.ndarray | None = None
    last_info: Dict[str, Any] = {}

    for ma in meta_seq:
        obs, r, terminated, truncated, info = env.step(int(ma))
        total_r += float(r)
        last_obs = obs
        last_info = info
        if terminated or truncated:
            break

    if last_obs is None:
        raise RuntimeError("env.step did not return an observation.")

    return last_obs, total_r, terminated, truncated, last_info
