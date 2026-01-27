from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Tuple

import numpy as np

from .config import GridworldConfig


class Act(IntEnum):
    FORWARD = 0
    BACKWARD = 1
    LEFT = 2
    RIGHT = 3


@dataclass(frozen=True)
class StepResult:
    next_pos: Tuple[int, int]
    reward: float
    done: bool
    done_reason: str  # "goal" | "crash" | "timeout" | "none"


def make_obs(obstacles: np.ndarray, ego_pos: Tuple[int, int]) -> np.ndarray:
    h, w = obstacles.shape
    obs = np.zeros((2, h, w), dtype=np.float32)
    obs[0] = obstacles.astype(np.float32)
    r, c = ego_pos
    obs[1, r, c] = 1.0
    return obs


def step_deterministic(
    cfg: GridworldConfig,
    obstacles: np.ndarray,
    pos: Tuple[int, int],
    action: int,
    t: int,
) -> StepResult:
    h, w = cfg.h, cfg.w
    r, c = pos

    a = Act(int(action))
    nr, nc = r, c

    if a == Act.FORWARD:
        nc = min(w - 1, c + 1)
    elif a == Act.BACKWARD:
        nc = max(0, c - 1)
    elif a == Act.LEFT:
        nr = max(0, r - 1)
    elif a == Act.RIGHT:
        nr = min(h - 1, r + 1)

    rew = float(cfg.living_reward)
    done = False
    reason = "none"

    if obstacles[nr, nc] == 1:
        rew += float(cfg.crash_reward)
        return StepResult((nr, nc), rew, True, "crash")

    if cfg.col_progress_weight != 0.0:
        rew += float(cfg.col_progress_weight) * float(nc - c)

    if nc == cfg.goal_col:
        rew += float(cfg.goal_rewards_by_row[nr])
        return StepResult((nr, nc), rew, True, "goal")

    if (t + 1) >= cfg.max_steps:
        done = True
        reason = "timeout"

    return StepResult((nr, nc), rew, done, reason)


class FiniteHorizonSolver:
    """Finite-horizon dynamic programming with timeout terminal."""

    def __init__(self, cfg: GridworldConfig, obstacles: np.ndarray):
        self.cfg = cfg
        self.obstacles = obstacles.astype(np.int8)
        self.h, self.w = cfg.h, cfg.w
        self.t_max = int(cfg.max_steps)

        self.v = np.zeros((self.t_max + 1, self.h, self.w), dtype=np.float32)
        self.pi = np.zeros((self.t_max, self.h, self.w), dtype=np.int64)

    def solve(self) -> None:
        cfg = self.cfg

        for t in range(self.t_max - 1, -1, -1):
            for r in range(self.h):
                for c in range(self.w):
                    if self.obstacles[r, c] == 1:
                        self.v[t, r, c] = float(cfg.crash_reward)
                        self.pi[t, r, c] = int(Act.FORWARD)
                        continue

                    best_q = -1e9
                    best_a = 0

                    for a in range(4):
                        sr = step_deterministic(cfg, self.obstacles, (r, c), a, t)
                        q = float(sr.reward)

                        if not sr.done:
                            nr, nc = sr.next_pos
                            q += float(cfg.gamma) * float(self.v[t + 1, nr, nc])

                        if q > best_q:
                            best_q = q
                            best_a = a

                    self.v[t, r, c] = float(best_q)
                    self.pi[t, r, c] = int(best_a)

    def best_action(self, pos: Tuple[int, int], t: int = 0) -> int:
        r, c = pos
        tt = int(np.clip(t, 0, self.t_max - 1))
        return int(self.pi[tt, r, c])
