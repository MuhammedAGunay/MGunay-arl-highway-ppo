from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium.wrappers import RecordVideo
from PIL import Image, ImageDraw, ImageFont

from .config import SimBundle
from .envs import build_env
from .sim_bridge import SystemPolicy

A_LANE_LEFT = 0
A_IDLE = 1
A_LANE_RIGHT = 2
A_FASTER = 3
A_SLOWER = 4


def _find_overlay_wrapper(env: gym.Env) -> Optional[gym.Env]:
    cur: Optional[gym.Env] = env
    while cur is not None:
        if hasattr(cur, "set_overlay"):
            return cur
        cur = getattr(cur, "env", None)
    return None


def _ego_lane_index(env: gym.Env) -> int:
    v = getattr(env.unwrapped, "vehicle", None)
    if v is None:
        return 0

    li = getattr(v, "lane_index", None)
    if li is None or len(li) < 3:
        return 0

    return int(li[2])


def obs7x7_to_bool_2x4x7(
    obs_cyx: np.ndarray,
    ego_lane: int,
    ego_col: int = 3,
) -> np.ndarray:
    if not isinstance(obs_cyx, np.ndarray) or obs_cyx.ndim != 3:
        raise ValueError(
            f"Expected obs as (C,Y,X) ndarray, got {type(obs_cyx)} shape={getattr(obs_cyx,'shape',None)}"
        )

    c, y, x = obs_cyx.shape
    if (y, x) != (7, 7):
        raise ValueError(
            f"Expected grid 7x7, got (Y,X)=({y},{x}). Check SimConfig x/y bounds/steps."
        )

    presence = obs_cyx[0]
    on_road = obs_cyx[3] if c >= 4 else None
    if on_road is None:
        raise ValueError("on_road feature missing. Ensure SimConfig.features includes 'on_road'.")

    road_rows = [i for i in range(y) if float(np.mean(on_road[i])) > 0.5]

    if len(road_rows) != 4:
        means = [(i, float(np.mean(on_road[i]))) for i in range(y)]
        means.sort(key=lambda t: t[1], reverse=True)
        road_rows = sorted([t[0] for t in means[:4]])

    obstacles_4x7 = (presence[road_rows, :] > 0.1).astype(np.float32)

    ego_4x7 = np.zeros((4, 7), dtype=np.float32)
    ego_lane = int(np.clip(ego_lane, 0, 3))
    ego_col = int(np.clip(ego_col, 0, 6))

    ego_4x7[ego_lane, ego_col] = 1.0
    obstacles_4x7[ego_lane, ego_col] = 0.0

    return np.stack([obstacles_4x7, ego_4x7], axis=0).astype(np.float32)


def system_action_to_meta_seq(
    action: int,
    macro_len_fwd_bwd: int,
    macro_len_lane: int,
    fwd_seq: Tuple[int, ...],
    bwd_seq: Tuple[int, ...],
    lane_hold_action: int,
) -> Tuple[int, ...]:
    if action == 2:
        return (A_LANE_LEFT,) + (lane_hold_action,) * (macro_len_lane - 1)
    if action == 3:
        return (A_LANE_RIGHT,) + (lane_hold_action,) * (macro_len_lane - 1)

    if action == 0:
        seq = tuple(int(x) for x in fwd_seq)
        L = macro_len_fwd_bwd
    elif action == 1:
        seq = tuple(int(x) for x in bwd_seq)
        L = macro_len_fwd_bwd
    else:
        return (A_IDLE,) * macro_len_fwd_bwd

    if len(seq) < L:
        seq = seq + (A_IDLE,) * (L - len(seq))
    return seq[:L]


def step_macro(env: gym.Env, seq: Tuple[int, ...]) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
    total_r = 0.0
    terminated = False
    truncated = False
    info: Dict[str, Any] = {}
    obs: Optional[np.ndarray] = None

    for aa in seq:
        obs, r, terminated, truncated, info = env.step(int(aa))
        total_r += float(r)
        if terminated or truncated:
            break

    try:
        _ = env.render()
    except Exception:
        pass

    if obs is None:
        raise RuntimeError("env.step did not return an observation.")

    return obs, total_r, terminated, truncated, info


class OverlayWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, cell_px: int = 14, pad: int = 8):
        super().__init__(env)
        self.last_grid: Optional[np.ndarray] = None
        self.last_action: Optional[int] = None
        self.cell_px = int(cell_px)
        self.pad = int(pad)
        try:
            self.font = ImageFont.load_default()
        except Exception:
            self.font = None

    def set_overlay(self, grid_2x4x7: np.ndarray, action: int) -> None:
        self.last_grid = grid_2x4x7
        self.last_action = int(action)

    def render(self):
        frame = self.env.render()
        if frame is None or not isinstance(frame, np.ndarray):
            return frame

        img = Image.fromarray(frame)
        draw = ImageDraw.Draw(img, "RGBA")

        if self.last_grid is None:
            return np.array(img)

        g = self.last_grid
        obstacles = g[0]
        ego = g[1]

        inset_w = 7 * self.cell_px
        inset_h = 4 * self.cell_px
        x0 = 10
        y0 = img.size[1] - inset_h - 10

        draw.rectangle(
            [x0 - self.pad, y0 - self.pad, x0 + inset_w + self.pad, y0 + inset_h + self.pad],
            fill=(0, 0, 0, 140),
        )

        for i in range(4):
            for j in range(7):
                cx0 = x0 + j * self.cell_px
                cy0 = y0 + i * self.cell_px
                cx1 = cx0 + self.cell_px - 1
                cy1 = cy0 + self.cell_px - 1

                if obstacles[i, j] > 0.5:
                    draw.rectangle([cx0, cy0, cx1, cy1], fill=(255, 0, 0, 180))
                else:
                    draw.rectangle([cx0, cy0, cx1, cy1], outline=(255, 255, 255, 35))

                if ego[i, j] > 0.5:
                    draw.rectangle([cx0, cy0, cx1, cy1], outline=(0, 255, 0, 255), width=2)

        if self.font is not None and self.last_action is not None:
            name = {0: "FWD", 1: "BWD", 2: "LEFT", 3: "RIGHT"}.get(self.last_action, str(self.last_action))
            draw.rectangle([5, 5, 160, 28], fill=(0, 0, 0, 140))
            draw.text((10, 9), f"S2={name}", fill=(255, 255, 255, 255), font=self.font)

        return np.array(img)


def main() -> None:
    cfg = SimBundle()

    env = build_env(cfg.sim, render_mode="rgb_array")
    env = OverlayWrapper(env, cell_px=cfg.video.cell_px, pad=cfg.video.pad)

    out_dir = Path(cfg.video.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    env = RecordVideo(
        env,
        video_folder=str(out_dir),
        name_prefix=cfg.video.name_prefix,
        episode_trigger=lambda ep: True,
        fps=int(cfg.video.fps),
    )
    try:
        env.unwrapped.set_record_video_wrapper(env)
    except Exception:
        pass

    policy = SystemPolicy(cfg.model.model_path, device=cfg.model.device)

    try:
        obs, _ = env.reset(seed=int(cfg.sim.seed))
        for _ in range(2):
            _ = env.render()

        terminated = False
        truncated = False
        macro_t = 0

        while not (terminated or truncated) and macro_t < int(cfg.video.max_macro_steps):
            ego_lane = _ego_lane_index(env)
            s2_grid = obs7x7_to_bool_2x4x7(
                obs,
                ego_lane=ego_lane,
                ego_col=cfg.sim.ego_center_col,
            )

            action = policy.predict_from_obs_2hw(s2_grid)

            ow = _find_overlay_wrapper(env)
            if ow is None:
                raise RuntimeError("OverlaySystem2Wrapper not found in wrapper chain.")
            ow.set_overlay(s2_grid, action)

            seq = system_action_to_meta_seq(
                action=action,
                macro_len_fwd_bwd=int(cfg.sim.macro_len_fwd_bwd),
                macro_len_lane=int(cfg.sim.macro_len_lane),
                fwd_seq=tuple(cfg.sim.fwd_seq),
                bwd_seq=tuple(cfg.sim.bwd_seq),
                lane_hold_action=int(cfg.sim.lane_hold),
            )

            obs, _r, terminated, truncated, _info = step_macro(env, seq)
            macro_t += 1

        for _ in range(2):
            _ = env.render()

        print(
            f"Saved video(s) to: {out_dir} | macro_steps={macro_t} "
            f"term={terminated} trunc={truncated}"
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
