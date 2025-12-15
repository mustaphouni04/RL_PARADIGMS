"""
Evaluation of trained Pacman Atari agents (SB3)
"""

import os
import numpy as np
import torch
import gymnasium as gym

from stable_baselines3 import A2C, PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.evaluation import evaluate_policy

from config import Config
from wrappers import create_pacman_env


MODEL_PATH = "/fhome/pmlai09/miguel/env/pacman/models/ppo/best/best_model.zip"
typem = "ppo"          #a2c or ppo
N_EVAL_EPISODES = 100
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"



def load_model(path, algo, device):
    if typem == "a2c":
        return A2C.load(path, device=device)
    elif typem == "ppo":
        return PPO.load(path, device=device)
    else:
        raise ValueError(f"Algorithm not supported: {typem}")


def main():

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    cfg = Config()

    def make_env():
        env = create_pacman_env(cfg, test_mode=False)
        return env

    env = DummyVecEnv([make_env])
    env = VecFrameStack(env, cfg.FRAME_STACK)

    print(f"  Obs space: {env.observation_space}")
    print(f"  Action space: {env.action_space}")

    model = load_model(MODEL_PATH, typem, DEVICE)
    print(f"Model ({typem.upper()}) in {DEVICE}")

    mean_reward, std_reward = evaluate_policy(
        model,
        env,
        n_eval_episodes=N_EVAL_EPISODES,
        deterministic=True,
        render=False,
        return_episode_rewards=False
    )

    print("Results:")
    print(f"episodes : {N_EVAL_EPISODES}")
    print(f"Mean reward:        {mean_reward:.2f}")
    print(f"Std reward:   {std_reward:.2f}")

    print("Detailed per-episode results:")
    episode_rewards, episode_lengths = evaluate_policy(
        model,
        env,
        n_eval_episodes=N_EVAL_EPISODES,
        deterministic=True,
        render=False,
        return_episode_rewards=True
    )

    for i, (r, l) in enumerate(zip(episode_rewards, episode_lengths), 1):
        print(f"  Ep {i:02d} | Reward: {r:7.2f} | Steps: {l:5d}")

    # Save results to a file
    os.makedirs("results", exist_ok=True)
    results_path = "results/evaluation.txt"

    with open(results_path, "w") as f:
        f.write("Pacman Evaluation Results\n")
        f.write(f"Model: {MODEL_PATH}\n")
        f.write(f"Algorithm: {typem}\n")
        f.write(f"Episodes: {N_EVAL_EPISODES}\n")
        f.write(f"Mean reward: {mean_reward:.2f}\n")
        f.write(f"Std reward: {std_reward:.2f}\n\n")
        for i, (r, l) in enumerate(zip(episode_rewards, episode_lengths), 1):
            f.write(f"Episode {i}: reward={r:.2f}, steps={l}\n")

    print(results_path)

    env.close()


if __name__ == "__main__":
    main()
