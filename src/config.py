from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TrainConfig:
    env_id: str = "highway-v0"
    seed: int = 42

    total_timesteps: int = 20_000
    n_envs: int = 4

    learning_rate: float = 3e-4
    n_steps: int = 1024
    batch_size: int = 256
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    device: str = "cpu"

    checkpoint_every_timesteps: int = 5_000

    log_dir: str = "logs"
    checkpoint_dir: str = "checkpoints"
    model_dir: str = "models"
    assets_dir: str = "assets"
    videos_dir: str = "videos"


@dataclass(frozen=True)
class VideoConfig:
    fps: int = 15
    duration: int = 200
    seed_untrained: int = 123
    seed_trained: int = 456
