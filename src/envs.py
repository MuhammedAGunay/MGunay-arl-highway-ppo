from __future__ import annotations

from typing import Optional

import gymnasium as gym
import highway_env  # noqa: F401
import numpy as np

from .config import SimConfig


class SwapXYObs(gym.ObservationWrapper):
    """Swap (C, X, Y) -> (C, Y, X)."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        if not isinstance(self.observation_space, gym.spaces.Box):
            raise TypeError("SwapXYObs expects Box observation space.")

        old = self.observation_space
        if len(old.shape) != 3:
            raise ValueError(f"Expected 3D obs (C,X,Y), got: {old.shape}")

        c, x, y = old.shape
        self.observation_space = gym.spaces.Box(
            low=float(np.min(old.low)),
            high=float(np.max(old.high)),
            shape=(c, y, x),
            dtype=old.dtype,
        )

    def observation(self, observation: np.ndarray) -> np.ndarray:
        return np.transpose(observation, (0, 2, 1))


def build_env(cfg: SimConfig, render_mode: Optional[str] = None) -> gym.Env:
    env = gym.make(cfg.env_id, render_mode=render_mode)
    base = env.unwrapped.config

    base.update(
        {
            "observation": {
                "type": "OccupancyGrid",
                "features": list(cfg.features),
                "grid_size": [[float(cfg.x_min), float(cfg.x_max)], [float(cfg.y_min_m), float(cfg.y_max_m)]],
                "grid_step": [float(cfg.x_step), float(cfg.lane_width)],
                "align_to_vehicle_axes": bool(cfg.align_to_vehicle_axes),
                "absolute": bool(cfg.absolute),
                "clip": bool(cfg.clip),
                "as_image": bool(cfg.as_image),
            },
            "action": {
                "type": "DiscreteMetaAction",
                "target_speeds": list(cfg.target_speeds),
            },
            "duration": int(cfg.duration),
            "lanes_count": int(cfg.lanes_count),
            "vehicles_count": int(cfg.vehicles_count),
            "simulation_frequency": int(cfg.simulation_frequency),
            "policy_frequency": int(cfg.policy_frequency),
        }
    )

    env.unwrapped.configure(base)
    env.reset(seed=int(cfg.seed))

    if cfg.swap_xy:
        env = SwapXYObs(env)

    return env
