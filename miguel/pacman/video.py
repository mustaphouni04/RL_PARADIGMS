"""
"""
#record_video.py
"""

import os
import cv2
import imageio
import torch
import gymnasium as gym

from stable_baselines3 import A2C, PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

from config import Config
from wrappers import create_pacman_env


MODEL_PATH = "/fhome/pmlai09/miguel/env/pacman/models/ppo/final.zip"
typem = "ppo"

MAX_STEPS = 600000
FPS = 60

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

VIDEO_DIR = "videos"
os.makedirs(VIDEO_DIR, exist_ok=True)

TARGET_SIZE = (320, 240)



# LOAD MODEL
def load_model(path, algo, device):
    if typem == "a2c":
        return A2C.load(path, device=device)
    elif typem == "ppo":
        return PPO.load(path, device=device)
    else:
        raise ValueError(f"Algorithm not supported: {typem}")


# MAIN
def main():
    print("Pacman Gameplay Recording")

    cfg = Config()

    def make_model_env():
        return create_pacman_env(cfg, test_mode=True)

    model_env = DummyVecEnv([make_model_env])
    model_env = VecFrameStack(model_env, cfg.FRAME_STACK)

    render_env = gym.make(
        cfg.ENV_ID,
        render_mode="rgb_array",
        obs_type="rgb",
        full_action_space=False,
        frameskip=1,
        repeat_action_probability=0.0,
    )

    model = load_model(MODEL_PATH, typem, DEVICE)
    print(f"Model loaded ({typem.upper()})")

    obs = model_env.reset()
    render_env.reset()

    frames = []
    total_reward = 0.0

    print("\ngameplay started...")

    for step in range(MAX_STEPS):
        action, _ = model.predict(obs, deterministic=True)

        obs, reward, done, info = model_env.step(action)

        action_scalar = int(action[0])
        _, _, term_r, trunc_r, _ = render_env.step(action_scalar)

        total_reward += reward[0]

        frame = render_env.render()
        if frame is not None:
            frame = cv2.resize(frame, TARGET_SIZE, interpolation=cv2.INTER_AREA)
            frames.append(frame)

        if done[0] or term_r or trunc_r:
            print(f"Episode terminated at step {step}")
            break

    model_env.close()
    render_env.close()

    print(f"\n Reward: {total_reward:.1f}")
    print(f"Frames captured: {len(frames)}")

    if not frames:
        print("No frames captured")
        return

    gif_path = os.path.join(VIDEO_DIR, "pacman_gameplay.gif")
    imageio.mimsave(gif_path, frames, fps=FPS)

    print("\nVideo saved:")


if __name__ == "__main__":
    main()
"""

"""
record_video.py
"""

import os
import cv2
import imageio
import torch
import gymnasium as gym

from stable_baselines3 import A2C, PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

from config import Config
from wrappers import create_pacman_env


MODEL_PATH = "/fhome/pmlai09/miguel/env/pacman/models/ppo/final.zip"
typem = "ppo"

MAX_STEPS = 600000
FPS = 60
N_EPISODES = 60  # número de episodios a probar

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

VIDEO_DIR = "videos"
os.makedirs(VIDEO_DIR, exist_ok=True)

TARGET_SIZE = (320, 240)


def load_model(path, algo, device):
    if typem == "a2c":
        return A2C.load(path, device=device)
    elif typem == "ppo":
        return PPO.load(path, device=device)
    else:
        raise ValueError(f"Algorithm not supported: {typem}")


def main():
    cfg = Config()

    def make_model_env():
        return create_pacman_env(cfg, test_mode=True)

    model_env = DummyVecEnv([make_model_env])
    model_env = VecFrameStack(model_env, cfg.FRAME_STACK)

    render_env = gym.make(
        cfg.ENV_ID,
        render_mode="rgb_array",
        obs_type="rgb",
        full_action_space=False,
        frameskip=1,
        repeat_action_probability=0.0,
    )

    model = load_model(MODEL_PATH, typem, DEVICE)

    best_reward = -float("inf")
    best_frames = []

    for ep in range(N_EPISODES):
        obs = model_env.reset()
        render_env.reset()

        frames = []
        total_reward = 0.0

        for step in range(MAX_STEPS):
            action, _ = model.predict(obs, deterministic=True)

            obs, reward, done, info = model_env.step(action)

            action_scalar = int(action[0])
            _, _, term_r, trunc_r, _ = render_env.step(action_scalar)

            total_reward += reward[0]

            frame = render_env.render()
            if frame is not None:
                frame = cv2.resize(frame, TARGET_SIZE, interpolation=cv2.INTER_AREA)
                frames.append(frame)

            if done[0] or term_r or trunc_r:
                break

        if total_reward > best_reward:
            best_reward = total_reward
            best_frames = frames

    model_env.close()
    render_env.close()

    if not best_frames:
        return

    gif_path = os.path.join(VIDEO_DIR, "pacman_best_episode.gif")
    imageio.mimsave(gif_path, best_frames, fps=FPS)


if __name__ == "__main__":
    main()
