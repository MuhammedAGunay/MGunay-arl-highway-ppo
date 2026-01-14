from __future__ import annotations

from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import VecMonitor

from src.config import TrainConfig
from src.envs import make_vec_env


def train(cfg: TrainConfig) -> None:
    project_root = Path(__file__).resolve().parents[1]

    log_dir = project_root / cfg.log_dir
    checkpoint_dir = project_root / cfg.checkpoint_dir
    model_dir = project_root / cfg.model_dir

    log_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    env = make_vec_env(cfg.env_id, seed=cfg.seed, n_envs=cfg.n_envs)
    env = VecMonitor(env, filename=str(log_dir / "monitor.csv"))

    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=cfg.learning_rate,
        n_steps=cfg.n_steps,
        batch_size=cfg.batch_size,
        n_epochs=cfg.n_epochs,
        gamma=cfg.gamma,
        gae_lambda=cfg.gae_lambda,
        clip_range=cfg.clip_range,
        ent_coef=cfg.ent_coef,
        vf_coef=cfg.vf_coef,
        max_grad_norm=cfg.max_grad_norm,
        verbose=1,
        tensorboard_log=str(log_dir),
        seed=cfg.seed,
        device=cfg.device,
    )

    save_freq_steps = max(cfg.checkpoint_every_timesteps // cfg.n_envs, 1)

    checkpoint_cb = CheckpointCallback(
        save_freq=save_freq_steps,
        save_path=str(checkpoint_dir),
        name_prefix="ppo_highway",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    model.learn(
        total_timesteps=cfg.total_timesteps,
        callback=checkpoint_cb,
        progress_bar=True,
    )

    model.save(str(model_dir / "ppo_highway_final"))
    env.close()


if __name__ == "__main__":
    train(TrainConfig())
