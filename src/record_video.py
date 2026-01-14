from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import highway_env  # noqa: F401
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import PPO

from src.config import TrainConfig, VideoConfig


def make_env(env_id: str, duration: int) -> gym.Env:
    env = gym.make(env_id, render_mode="rgb_array")
    cfg = env.unwrapped.config
    cfg["duration"] = duration
    env.unwrapped.configure(cfg)
    return env


def wrap_video(env: gym.Env, video_dir: Path, name_prefix: str, fps: int) -> gym.Env:
    wrapped = RecordVideo(
        env,
        video_folder=str(video_dir),
        name_prefix=name_prefix,
        episode_trigger=lambda e: True,
        fps=fps,
    )
    wrapped.unwrapped.set_record_video_wrapper(wrapped)
    return wrapped


def record_random(env_id: str, video_dir: Path, name_prefix: str, vcfg: VideoConfig) -> Path:
    env = make_env(env_id, duration=vcfg.duration)
    env = wrap_video(env, video_dir, name_prefix, fps=vcfg.fps)

    obs, info = env.reset(seed=vcfg.seed_untrained)
    terminated = truncated = False
    steps = 0
    while not (terminated or truncated) and steps < vcfg.duration:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        steps += 1

    env.close()
    mp4s = sorted(video_dir.glob(f"{name_prefix}*.mp4"))
    if not mp4s:
        raise FileNotFoundError("Untrained stage video not recorded.")
    return mp4s[-1]


def record_model(
    env_id: str,
    model_path: Path,
    video_dir: Path,
    name_prefix: str,
    vcfg: VideoConfig,
    device: str,
) -> Path:
    env = make_env(env_id, duration=vcfg.duration)
    env = wrap_video(env, video_dir, name_prefix, fps=vcfg.fps)

    model = PPO.load(str(model_path), device=device)

    obs, info = env.reset(seed=vcfg.seed_trained)
    terminated = truncated = False
    steps = 0
    while not (terminated or truncated) and steps < vcfg.duration:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(int(action))
        steps += 1

    env.close()
    mp4s = sorted(video_dir.glob(f"{name_prefix}*.mp4"))
    if not mp4s:
        raise FileNotFoundError(f"Trained stage video not recorded: {model_path}")
    return mp4s[-1]


def stitch_sequential(a: Path, b: Path, c: Path, out_mp4: Path, fps: int) -> None:
    from moviepy import VideoFileClip, concatenate_videoclips

    clips = [VideoFileClip(str(p)) for p in [a, b, c]]
    final = concatenate_videoclips(clips, method="compose")

    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    final.write_videofile(str(out_mp4), fps=fps, audio=False, codec="libx264")

    for clip in clips:
        clip.close()
    final.close()


def main() -> None:
    cfg = TrainConfig()
    vcfg = VideoConfig()

    project_root = Path(__file__).resolve().parents[1]
    video_dir = project_root / cfg.videos_dir
    checkpoints_dir = project_root / cfg.checkpoint_dir
    out_mp4 = project_root / cfg.assets_dir / "evolution.mp4"

    video_dir.mkdir(parents=True, exist_ok=True)
    for p in video_dir.glob("*.mp4"):
        p.unlink()

    half = checkpoints_dir / "ppo_highway_10000_steps.zip"
    full = checkpoints_dir / "ppo_highway_20000_steps.zip"

    v1 = record_random(cfg.env_id, video_dir, "stage1_untrained", vcfg)
    v2 = record_model(cfg.env_id, half, video_dir, "stage2_half", vcfg, device=cfg.device)
    v3 = record_model(cfg.env_id, full, video_dir, "stage3_full", vcfg, device=cfg.device)

    stitch_sequential(v1, v2, v3, out_mp4, fps=vcfg.fps)
    print(f"Saved evolution video to: {out_mp4}")


if __name__ == "__main__":
    main()
