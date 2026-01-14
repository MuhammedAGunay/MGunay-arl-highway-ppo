from __future__ import annotations

from typing import Callable, Optional

import gymnasium as gym
import highway_env  # noqa: F401
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv


def make_single_env(
    env_id: str,
    seed: int,
    render_mode: Optional[str] = None,
) -> Callable[[], gym.Env]:
    def _init() -> gym.Env:
        env = gym.make(env_id, render_mode=render_mode)
        env.reset(seed=seed)
        return env

    return _init


def make_vec_env(env_id: str, seed: int, n_envs: int) -> DummyVecEnv:
    env_fns = [make_single_env(env_id, seed + i) for i in range(n_envs)]
    try:
        return SubprocVecEnv(env_fns)
    except Exception:
        return DummyVecEnv(env_fns)
