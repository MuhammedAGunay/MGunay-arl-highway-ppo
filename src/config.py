from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

# =========================
# Simulation (highway-env)
# =========================

@dataclass(frozen=True)
class SimConfig:
    env_id: str = "highway-v0"
    seed: int = 256

    duration: int = 100
    lanes_count: int = 4
    vehicles_count: int = 50
    simulation_frequency: int = 15
    policy_frequency: int = 4

    # OccupancyGrid (C, X, Y) typically; swap_xy makes it (C, Y, X)
    x_min: float = -52.5
    x_max: float = 52.5
    x_step: float = 15.0

    lane_width: float = 4.0
    y_min_m: float = -14.0
    y_max_m: float = 14.0

    features: Sequence[str] = ("presence", "vx", "vy", "on_road")
    align_to_vehicle_axes: bool = True
    absolute: bool = False
    clip: bool = True
    as_image: bool = False
    swap_xy: bool = True

    # DiscreteMetaAction speeds (low / normal / high)
    target_speeds: Sequence[float] = (18.0, 23.0, 28.0)

    # Macro controls
    macro_len_fwd_bwd: int = 4
    macro_len_lane: int = 8
    lane_hold: int = 1  # IDLE action id

    fwd_seq: Sequence[int] = (3, 1, 1, 4)
    bwd_seq: Sequence[int] = (4, 1, 1, 3)

    # System2 expects ego at fixed center column in 4x7
    ego_center_col: int = 3


@dataclass(frozen=True)
class VideoConfig:
    fps: int = 15
    out_dir: str = "videos"
    name_prefix: str = "eval"
    max_macro_steps: int = 250

    cell_px: int = 14
    pad: int = 8
    show_vx: bool = False


@dataclass(frozen=True)
class ModelConfig:
    model_path: str = "models/cnn.pt"
    device: str = "cuda"


@dataclass(frozen=True)
class SimBundle:
    sim: SimConfig = SimConfig()
    video: VideoConfig = VideoConfig()
    model: ModelConfig = ModelConfig()


# =========================
# Gridworld (dataset MDP)
# =========================

@dataclass(frozen=True)
class GridworldConfig:
    h: int = 4
    w: int = 7

    ego_start_col: int = 3
    goal_col: int = 6

    gamma: float = 0.90
    living_reward: float = -0.10
    noise: float = 0.0
    max_steps: int = 30

    crash_reward: float = -1.0
    goal_rewards_by_row: Sequence[float] = (1.00, 1.05, 1.10, 1.15)

    col_progress_weight: float = 0.00


@dataclass(frozen=True)
class MapGenConfig:
    obstacle_prob: float = 0.18
    obstacle_prob_choices: Sequence[float] = (0.10, 0.18, 0.28, 0.38)
    bwd_obstacle_prob_choices: Sequence[float] = (0.28, 0.38, 0.50)
    ensure_start_free: bool = True


@dataclass(frozen=True)
class DatasetConfig:
    n_maps: int = 5_000
    sample_all_rows: bool = True

    extra_random_states_per_map: int = 0

    balance_actions: bool = True
    target_per_action: int | None = 20_000

    stratify_by_density: bool = True
    density_bin_edges: Sequence[int] = (0, 5, 10, 15, 28)

    val_frac: float = 0.10
    test_frac: float = 0.10
    split_seed: int = 123

    drop_trapped: bool = True
    max_mining_maps: int = 200_000

    dataset_npz_path: str = "data/80Kdataset.npz"


@dataclass(frozen=True)
class CNNConfig:
    in_channels: int = 2
    channels: Sequence[int] = (32, 64)
    kernel_size: int = 3
    padding: int = 1

    hidden_dim: int = 128
    padding_mode: str = "zeros"
    wall_padding: bool = False


@dataclass(frozen=True)
class TrainConfig:
    seed: int = 42
    epochs: int = 10
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-4

    out_dir: str = "models"
    out_name: str = "cnn.pt"

    device: str = "cuda"


@dataclass(frozen=True)
class ConfigBundle:
    gw: GridworldConfig = GridworldConfig()
    gen: MapGenConfig = MapGenConfig()
    data: DatasetConfig = DatasetConfig()
    cnn: CNNConfig = CNNConfig()
    train: TrainConfig = TrainConfig()
