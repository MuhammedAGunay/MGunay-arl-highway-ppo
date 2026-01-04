from __future__ import annotations

import gymnasium as gym
import highway_env


def run_random(episodes: int = 3, seed: int = 42) -> None:
    env = gym.make("highway-v0", render_mode="rgb_array")
    obs, info = env.reset(seed=seed)

    for ep in range(episodes):
        terminated = False
        truncated = False
        ep_return = 0.0

        while not (terminated or truncated):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            ep_return += float(reward)

        print(f"Episode {ep + 1}: return={ep_return:.2f}")
        obs, info = env.reset()

    env.close()


if __name__ == "__main__":
    run_random()
