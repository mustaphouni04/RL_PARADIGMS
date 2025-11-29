import wandb
import gymnasium as gym
from stable_baselines3 import PPO
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecVideoRecorder
from stable_baselines3.common.callbacks import BaseCallback 
from utils.utils import make_env
from wandb.integration.sb3 import WandbCallback
import os
from ale_py import ALEInterface
import ale_py

ale = ALEInterface()
gym.register_envs(ale_py)

# --- CONFIGURATION ---
CONFIG = {
    "project_name": "Step2-RLProject",
    "env_id": "PongNoFrameskip-v4", 
    "total_timesteps": 10_000_000,
    "model_name": "ppo_pong_costum_settings",
    "export_path": "./models/",
    "learning_rate": 2.5e-4,
    "gamma": 0.99,
    "batch_size": 256,
    "n_steps": 128,
    "mean_reward_bound": 19.0,
    "reward_window": 10
}

class StopOnRewardCallback(BaseCallback):
    def __init__(self, reward_threshold: float, verbose: int = 1):
        super().__init__(verbose)
        self.reward_threshold = reward_threshold

    def _on_step(self) -> bool:
        ep_info_buffer = self.model.ep_info_buffer
        
        if len(ep_info_buffer) > 0:
            mean_reward = np.mean([ep_info["r"] for ep_info in ep_info_buffer])
            
            if mean_reward >= self.reward_threshold:
                if self.verbose > 0:
                    print(f"\nSOLVED! Mean reward {mean_reward:.2f} >= {self.reward_threshold}")
                    print(f"Stopping training at step {self.num_timesteps}")
                return False # Returning False stops the training
        
        return True

def train_model():
    run = wandb.init(
        project=CONFIG["project_name"],
        config=CONFIG,
        name=CONFIG["model_name"],
        sync_tensorboard=True,
        monitor_gym=False,
        save_code=True
    )

    env = DummyVecEnv([lambda: make_env(CONFIG["env_id"], render_mode="rgb_array")])
    
    env = VecMonitor(env)
    print(f"\n>>> Training Agent (Right Paddle) vs Atari AI (Left Paddle)...")

    video_folder = f"runs/{run.id}/videos"
    os.makedirs(video_folder, exist_ok=True)
    
    env = VecVideoRecorder(
        env,
        video_folder=video_folder,
        record_video_trigger=lambda x: x % 100_000 == 0,
        video_length=2000,
        name_prefix=CONFIG["model_name"]
    )

    
    model = PPO(
        "CnnPolicy", 
        env, 
        verbose=1, 
        tensorboard_log=f"runs/{run.id}", 
        device="cuda",
        learning_rate=CONFIG["learning_rate"],
        gamma=CONFIG["gamma"],
        n_steps=CONFIG["n_steps"],
        batch_size=CONFIG["batch_size"],
        n_epochs=4,
        ent_coef=0.01,
        clip_range=0.1,
        policy_kwargs={"normalize_images": False}
    )
    
    stop_callback = StopOnRewardCallback(reward_threshold=CONFIG["mean_reward_bound"])
    wandb_callback = WandbCallback(verbose=2)
    
    model.learn(
        total_timesteps=CONFIG["total_timesteps"], 
        callback=[wandb_callback, stop_callback]
    )

    os.makedirs(CONFIG["export_path"], exist_ok=True)
    save_path = f"{CONFIG['export_path']}{CONFIG['model_name']}"
    model.save(save_path)
    print(f"Model exported at '{save_path}'")

    env.close()

    wandb.log({"video": wandb.Video(os.path.join(video_folder, f"{CONFIG['model_name']}-step-0-to-step-2000.mp4"), fps=30, format="mp4")})

    run.finish()

if __name__ == "__main__":
    train_model()
