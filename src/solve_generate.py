from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
from tqdm.auto import tqdm

from .config import ConfigBundle
from .gridworld import FiniteHorizonSolver


def _make_rewards(cfg: ConfigBundle) -> np.ndarray:
    rewards = np.zeros((cfg.gw.h, cfg.gw.w), dtype=np.float32)
    for r, rew in enumerate(cfg.gw.goal_rewards_by_row):
        rewards[r, cfg.gw.goal_col] = float(rew)
    return rewards


def _obstacle_count(obstacles: np.ndarray) -> int:
    return int(np.sum(obstacles.astype(np.int32)))


def _density_bin(count: int, edges: Sequence[int]) -> int:
    for i in range(len(edges) - 1):
        lo = int(edges[i])
        hi = int(edges[i + 1])
        if lo <= count < hi:
            return i
    return max(0, len(edges) - 2)


def _is_trapped(obstacles: np.ndarray, ego_pos: Tuple[int, int]) -> bool:
    h, w = obstacles.shape
    r, c = ego_pos
    neigh = [(r, c + 1), (r, c - 1), (r - 1, c), (r + 1, c)]
    blocked = 0

    for rr, cc in neigh:
        if rr < 0 or rr >= h or cc < 0 or cc >= w:
            blocked += 1
        elif bool(obstacles[rr, cc]):
            blocked += 1

    return blocked == 4


def _encode_obs(obstacles: np.ndarray, ego_pos: Tuple[int, int]) -> np.ndarray:
    h, w = obstacles.shape
    x = np.zeros((2, h, w), dtype=np.float32)
    x[0] = obstacles.astype(np.float32)
    r, c = ego_pos
    x[1, r, c] = 1.0
    return x


def _best_action(cfg: ConfigBundle, obstacles: np.ndarray, ego_pos: Tuple[int, int]) -> int:
    solver = FiniteHorizonSolver(cfg.gw, obstacles)
    solver.solve()
    return solver.best_action(ego_pos, t=0)


def build_dataset(cfg: ConfigBundle) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(cfg.train.seed)

    h, w = cfg.gw.h, cfg.gw.w
    ego_col = cfg.gw.ego_start_col

    buckets: Dict[int, List[Tuple[np.ndarray, int, int]]] = {0: [], 1: [], 2: [], 3: []}

    def add_sample(obstacles_in: np.ndarray, ego_pos: Tuple[int, int]) -> None:
        obstacles = obstacles_in
        if cfg.gen.ensure_start_free:
            obstacles = obstacles.copy()
            obstacles[ego_pos[0], ego_pos[1]] = False

        if cfg.data.drop_trapped and _is_trapped(obstacles, ego_pos):
            return

        a = _best_action(cfg, obstacles, ego_pos)
        dens = _obstacle_count(obstacles)
        dens_bin = _density_bin(dens, cfg.data.density_bin_edges)

        obs = _encode_obs(obstacles, ego_pos)
        buckets[a].append((obs, a, dens_bin))

    base_p = float(cfg.gen.obstacle_prob)

    base_iter = tqdm(range(int(cfg.data.n_maps)), desc="dataset/base", dynamic_ncols=True)
    for _ in base_iter:
        obstacles = (rng.random((h, w)) < base_p)

        rows = range(h) if cfg.data.sample_all_rows else [int(rng.integers(0, h))]
        for r in rows:
            add_sample(obstacles, (r, ego_col))

        for _ in range(int(cfg.data.extra_random_states_per_map)):
            rr = int(rng.integers(0, h))
            cc = int(rng.integers(0, w))
            add_sample(obstacles, (rr, cc))

        if cfg.data.balance_actions:
            base_iter.set_postfix(
                {
                    "FWD": len(buckets[0]),
                    "BWD": len(buckets[1]),
                    "LEFT": len(buckets[2]),
                    "RIGHT": len(buckets[3]),
                }
            )

    if not cfg.data.balance_actions:
        x = np.stack([t[0] for a in buckets for t in buckets[a]], axis=0)
        y = np.asarray([t[1] for a in buckets for t in buckets[a]], dtype=np.int64)
        d = np.asarray([t[2] for a in buckets for t in buckets[a]], dtype=np.int64)
        perm = rng.permutation(len(y))
        return x[perm], y[perm], d[perm]

    target = (
        min(len(buckets[a]) for a in buckets)
        if cfg.data.target_per_action is None
        else int(cfg.data.target_per_action)
    )

    def postfix() -> Dict[str, str]:
        return {
            "FWD": f"{len(buckets[0])}/{target}",
            "BWD": f"{len(buckets[1])}/{target}",
            "LEFT": f"{len(buckets[2])}/{target}",
            "RIGHT": f"{len(buckets[3])}/{target}",
        }

    mined = 0
    if any(len(buckets[a]) < target for a in buckets):
        mine_iter = tqdm(total=int(cfg.data.max_mining_maps), desc="dataset/mining", dynamic_ncols=True)

        while mined < int(cfg.data.max_mining_maps) and any(len(buckets[a]) < target for a in buckets):
            mined += 1
            mine_iter.update(1)

            if len(buckets[1]) < target:
                p = float(rng.choice(cfg.gen.bwd_obstacle_prob_choices))
            else:
                p = float(rng.choice(cfg.gen.obstacle_prob_choices))

            obstacles = (rng.random((h, w)) < p)

            rows = range(h) if cfg.data.sample_all_rows else [int(rng.integers(0, h))]
            for r in rows:
                add_sample(obstacles, (r, ego_col))

            for _ in range(int(cfg.data.extra_random_states_per_map)):
                rr = int(rng.integers(0, h))
                cc = int(rng.integers(0, w))
                add_sample(obstacles, (rr, cc))

            mine_iter.set_postfix(postfix())

            if all(len(buckets[a]) >= target for a in buckets):
                break

        mine_iter.close()

    n_bins = len(cfg.data.density_bin_edges) - 1
    x_list: List[np.ndarray] = []
    y_list: List[int] = []
    d_list: List[int] = []

    for a in (0, 1, 2, 3):
        items = buckets[a]
        if not items:
            continue

        rng.shuffle(items)

        if cfg.data.stratify_by_density and n_bins > 1:
            per_bin = int(np.ceil(target / n_bins))
            by_bin: Dict[int, List[Tuple[np.ndarray, int, int]]] = {b: [] for b in range(n_bins)}

            for it in items:
                by_bin[it[2]].append(it)

            chosen: List[Tuple[np.ndarray, int, int]] = []
            for b in range(n_bins):
                lst = by_bin[b]
                if not lst:
                    continue
                rng.shuffle(lst)
                chosen.extend(lst[:per_bin])

            if len(chosen) < target:
                rng.shuffle(items)
                chosen.extend(items[: (target - len(chosen))])

            chosen = chosen[:target]
        else:
            chosen = items[:target]

        for obs, yy, densb in chosen:
            x_list.append(obs)
            y_list.append(yy)
            d_list.append(densb)

    x = np.stack(x_list, axis=0)
    y = np.asarray(y_list, dtype=np.int64)
    d = np.asarray(d_list, dtype=np.int64)

    perm = rng.permutation(len(y))
    return x[perm], y[perm], d[perm]
