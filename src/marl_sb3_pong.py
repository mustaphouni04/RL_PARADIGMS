import wandb
import gymnasium as gym
from stable_baselines3 import PPO, A2C
import numpy as np
from stable_baselines3.common.vec_env import VecMonitor, VecVideoRecorder
from stable_baselines3.common.callbacks import CheckpointCallback 
from wandb.integration.sb3 import WandbCallback
import os
import supersuit as ss
from pettingzoo.atari import pong_v3

# --- CONFIGURATION ---
CONFIG = {
    "project_name": "Step2-RLProject-MARL",
    "env_id": "pong_v3", # PettingZoo ID
    "total_timesteps": 10_000_000,
    "model_name": "a2c_pong_marl_selfplay",
    "export_path": "./models/",
    "batch_size": 256,
    "learning_rate": 2.5e-4,
    "gamma": 0.99,
    "n_steps": 32, 
    "num_envs": 4,
}

# --- MARL ENVIRONMENT FACTORY ---
def make_marl_env(num_envs):
    env = pong_v3.parallel_env(render_mode="rgb_array")

    env = ss.color_reduction_v0(env, mode="B")
    env = ss.resize_v1(env, x_size=84, y_size=84)
    env = ss.frame_stack_v1(env, 4)

    def flip_obs(obs):
        return np.flip(obs, axis=2)

    env = ss.observation_lambda_v0(env, 
                                   lambda obs, obs_space, agent: flip_obs(obs) if agent == 'second_0' else obs,
                                   lambda obs_space: obs_space)


    env = ss.pettingzoo_env_to_vec_env_v1(env)

    env = ss.concat_vec_envs_v1(env, num_vec_envs=num_envs, num_cpus=4, base_class='stable_baselines3')

    return env

def resume_training():
    run = wandb.init(
        project=CONFIG["project_name"],
        config=CONFIG,
        name=CONFIG["model_name"] + "_continued",
        sync_tensorboard=True,
        monitor_gym=False,
        save_code=True
    )

    env = make_marl_env(CONFIG["num_envs"])
    env.render_mode = "rgb_array"
    env = VecMonitor(env)
    env.render_mode = "rgb_array"

    video_folder = f"runs/{run.id}/videos"
    os.makedirs(video_folder, exist_ok=True)

    
    env = VecVideoRecorder(
        env,
        video_folder=video_folder,
        record_video_trigger=lambda x: x % 100 == 0,
        video_length=2000,
        name_prefix=CONFIG["model_name"]
    )

    model_path = f"{CONFIG['export_path']}{CONFIG['model_name']}_final"

    print(f"Loading model from: {model_path}")

    model = A2C.load(model_path,
                     env=env,
                     tensorboard_log=f"runs/{run.id}")


    print(f"\n>>> Resuming training from step {model.num_timesteps}...")

    wandb_callback = WandbCallback(verbose=2)

    checkpoint_callback = CheckpointCallback(
        save_freq=500_000,
        save_path=CONFIG["export_path"],
        name_prefix=CONFIG["model_name"] + "_ckpt"
    )

    model.learn(
            total_timesteps=10_000_000,
            callback=[wandb_callback, checkpoint_callback],
            reset_num_timesteps=False)

    save_path = f"{CONFIG['export_path']}{CONFIG['model_name']}_continued"
    model.save(save_path)

    env.close()
    run.finish()


def train_model():
    run = wandb.init(
        project=CONFIG["project_name"],
        config=CONFIG,
        name=CONFIG["model_name"],
        sync_tensorboard=True,
        monitor_gym=False,
        save_code=True
    )

    # Create MARL Env
    env = make_marl_env(CONFIG["num_envs"])
    env.render_mode = "rgb_array"
    
    # Monitor Wrapper (SB3)
    env = VecMonitor(env)
    env.render_mode = "rgb_array"
    
    print(f"\n>>> Training MARL Self-Play (Both Paddles controlled by A2C)...")

    video_folder = f"runs/{run.id}/videos"
    os.makedirs(video_folder, exist_ok=True)

    
    env = VecVideoRecorder(
        env,
        video_folder=video_folder,
        record_video_trigger=lambda x: x % 100 == 0,
        video_length=2000,
        name_prefix=CONFIG["model_name"]
    )

    model = A2C(
        "CnnPolicy",
        env,
        verbose=1,
        tensorboard_log=f"runs/{run.id}",
        device="cuda",
        learning_rate=CONFIG["learning_rate"],
        gamma=CONFIG["gamma"],
        n_steps=CONFIG["n_steps"],
        #batch_size= CONFIG["batch_size"],
        #n_epochs=4,
        ent_coef=0.01,
        #clip_range=0.1,
        policy_kwargs={"normalize_images": True} # SB3 handles int->float auto if True
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=500_000,
        save_path=CONFIG["export_path"],
        name_prefix=CONFIG["model_name"]
    )

    wandb_callback = WandbCallback(verbose=2)

    model.learn(
        total_timesteps=CONFIG["total_timesteps"],
        callback=[wandb_callback, checkpoint_callback]
    )

    os.makedirs(CONFIG["export_path"], exist_ok=True)
    save_path = f"{CONFIG['export_path']}{CONFIG['model_name']}_final"
    model.save(save_path)
    print(f"Model exported at '{save_path}'")

    env.close()
    
    # Log video
    video_path = os.path.join(video_folder, f"{CONFIG['model_name']}-step-0-to-step-2000.mp4")
    if os.path.exists(video_path):
        wandb.log({"video": wandb.Video(video_path, fps=30, format="mp4")})

    run.finish()

if __name__ == "__main__":
    resume_training()
