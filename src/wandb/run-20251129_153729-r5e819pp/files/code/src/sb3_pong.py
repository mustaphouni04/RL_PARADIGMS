import wandb
import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing, RecordVideo
from wrap import FrameStack
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecTransposeImage
from stable_baselines3.common.callbacks import CallbackList 
from wandb.integration.sb3 import WandbCallback
import os
from ale_py import ALEInterface
ale = ALEInterface()
import ale_py

gym.register_envs(ale_py)

# --- CONFIGURATION ---
CONFIG = {
    "project_name": "Step2-RLProject",
    "env_id": "PongNoFrameskip-v4", 
    "total_timesteps": 1_000_000,
    "model_name": "ppo_pong_standard",
    "export_path": "./models/"
}

def make_env(render_mode="rgb_array"):
    print(f"Creating environment: {CONFIG['env_id']}")
    
    env = gym.make(CONFIG['env_id'], render_mode=render_mode)
    
    env = AtariPreprocessing(
        env, 
        noop_max=30, 
        frame_skip=4, 
        screen_size=84, 
        grayscale_obs=True, 
        scale_obs=False)

    env = FrameStack(env, 4)

    return env

def train_model():
    run = wandb.init(
        project=CONFIG["project_name"],
        config=CONFIG,
        name=CONFIG["model_name"],
        sync_tensorboard=True,
        monitor_gym=True,
        save_code=True
    )

    env = DummyVecEnv([lambda: make_env(render_mode="rgb_array")])
    
    env = VecTransposeImage(env)
    
    env = VecMonitor(env)

    video_folder = f"runs/{run.id}/videos"
    os.makedirs(video_folder, exist_ok=True)
    
    from stable_baselines3.common.vec_env import VecVideoRecorder
    env = VecVideoRecorder(
        env,
        video_folder=video_folder,
        record_video_trigger=lambda x: x % 100_000 == 0,
        video_length=2000,
        name_prefix=CONFIG["model_name"]
    )

    print(f"\n>>> Training Agent (Right Paddle) vs Atari AI (Left Paddle)...")
    
    model = PPO(
        "CnnPolicy", 
        env, 
        verbose=1, 
        tensorboard_log=f"runs/{run.id}", 
        device="cuda",
        # Specific PPO Hyperparameters for Atari (optional but recommended)
        learning_rate=2.5e-4,
        n_steps=128,
        batch_size=256,
        n_epochs=4,
        ent_coef=0.01,
    )

    wandb_callback = WandbCallback(verbose=2)
    
    model.learn(
        total_timesteps=CONFIG["total_timesteps"], 
        callback=wandb_callback
    )

    os.makedirs(CONFIG["export_path"], exist_ok=True)
    save_path = f"{CONFIG['export_path']}{CONFIG['model_name']}"
    model.save(save_path)
    print(f"Model exported at '{save_path}'")

    env.close()
    run.finish()

if __name__ == "__main__":
    train_model()
