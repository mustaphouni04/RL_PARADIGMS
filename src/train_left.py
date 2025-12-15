import wandb
import gymnasium as gym
from stable_baselines3 import A2C
from stable_baselines3.common.policies import ActorCriticCnnPolicy
from stable_baselines3.common.vec_env import VecMonitor, VecVideoRecorder, VecEnvWrapper
from stable_baselines3.common.callbacks import CheckpointCallback
from wandb.integration.sb3 import WandbCallback
import numpy as np
import torch
import os
import supersuit as ss
from pettingzoo.atari import pong_v3

# --- CONFIGURATION ---
CONFIG = {
    "project_name": "Step2-RLProject-MARL",
    "env_id": "pong_v3",
    "total_timesteps": 10_000_000,
    "model_name": "left_specialist", # Changed name to avoid overwriting self-play model
    "opponent_path": "./models/oldv2/a2c_pong_marl_selfplay_continuedv2", # Path to the FROZEN model
    "export_path": "./models/specialists/",
    "learning_rate": 2.5e-4,
    "gamma": 0.99,
    "n_steps": 32,
    "num_envs": 6, # Increased slightly for speed, adjust based on CPU cores
}

class SymmetricActorCriticPolicy(ActorCriticCnnPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.force_flip = False

    def extract_features(self, obs: torch.Tensor, features_extractor=None) -> torch.Tensor:
        if self.force_flip:
            obs = torch.flip(obs, dims=[-1])
        return super().extract_features(obs, features_extractor)

class LeftSpecialistVecWrapper(VecEnvWrapper):
    """
    Wraps the 12-agent VecEnv (6 matches x 2 players).
    - Hides the Right Agents (indices 0, 2, 4...) from the Trainer.
    - Uses 'opponent_model' to predict actions for Right Agents.
    - Only exposes Left Agents (indices 1, 3, 5...) to the Trainer.
    """
    def __init__(self, venv, opponent_model):
        # We assume the env structure is [Right0, Left0, Right1, Left1, ...]
        self.total_envs = venv.num_envs
        self.num_learners = self.total_envs // 2
        
        # Indices
        self.right_indices = np.arange(0, self.total_envs, 2) # Opponents
        self.left_indices = np.arange(1, self.total_envs, 2)  # Learners (Students)
        
        # Initialize Wrapper
        # We pass the observation/action space of the underlying venv (which is correct for single agents)
        super().__init__(venv, venv.observation_space, venv.action_space)
        
        # --- CRITICAL FIX ---
        # We must overwrite num_envs so SB3 knows we are only exposing half the agents
        self.num_envs = self.num_learners 
        
        self.opponent_model = opponent_model
        self.last_obs_all = None

    def reset(self):
        # SB3 VecEnv reset() returns only 'obs'
        obs = self.venv.reset()
        self.last_obs_all = obs
        # Return only Learner (Left) observations
        return obs[self.left_indices]

    def step_async(self, actions):
        # 'actions' comes from the Trainer (Learning Agent) -> shape (6,)
        
        # 1. Get Opponent Observations (indices 0, 2, 4...)
        opp_obs = self.last_obs_all[self.right_indices]
        
        # 2. Predict Opponent Actions (Batched on GPU)
        opp_actions, _ = self.opponent_model.predict(opp_obs, deterministic=True)
        
        # 3. Interleave Actions: [Opp0, Learn0, Opp1, Learn1, ...]
        # We need an array of size 12 to step the real environment
        full_actions = np.empty((self.total_envs,), dtype=actions.dtype)
        full_actions[self.right_indices] = opp_actions
        full_actions[self.left_indices] = actions
        
        self.venv.step_async(full_actions)

    def step_wait(self):
        # FIX 1: SB3 VecEnv returns 4 values (obs, rews, dones, infos)
        obs, rewards, dones, infos = self.venv.step_wait()
        
        self.last_obs_all = obs
        
        # FIX 2: Filter 'infos' (list) using list comprehension
        filtered_infos = [infos[i] for i in self.left_indices]
        
        # Return only Learner (Left) data (shape 6)
        return (obs[self.left_indices], 
                rewards[self.left_indices], 
                dones[self.left_indices], 
                filtered_infos)

def make_marl_env(num_envs):
    env = pong_v3.parallel_env(render_mode="rgb_array")
    env = ss.sticky_actions_v0(env, repeat_action_probability=0.25)
    env = ss.color_reduction_v0(env, mode="B") # Training on Hard Mode (Blue channel)
    env = ss.resize_v1(env, x_size=84, y_size=84)
    env = ss.frame_stack_v1(env, 4, stack_dim=0)

    # Flip Logic: Only flip the Left Agent ('second_0') so it sees the game from the Right perspective
    def flip_obs(obs):
        return np.flip(obs, axis=2)

    env = ss.observation_lambda_v0(env,
                                   lambda obs, obs_space, agent: flip_obs(obs) if agent == 'second_0' else obs,
                                   lambda obs_space: obs_space)

    env = ss.dtype_v0(env, dtype=np.float32)
    env = ss.normalize_obs_v0(env, env_min=0, env_max=1)
    env = ss.reshape_v0(env, (4, 84, 84))

    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, num_vec_envs=num_envs, num_cpus=4, base_class='stable_baselines3')

    return env

def resume_training():
    run = wandb.init(
        project=CONFIG["project_name"],
        config=CONFIG,
        name=CONFIG["model_name"],
        sync_tensorboard=True,
        monitor_gym=False,
        save_code=True
    )

    # 1. Create the Full Environment (Both Agents)
    env = make_marl_env(CONFIG["num_envs"])
    
    # 2. Load the OPPONENT Model (Frozen)
    print(f"Loading Opponent from: {CONFIG['opponent_path']}")
    opponent_model = A2C.load(CONFIG["opponent_path"], device="cuda", custom_objects={'policy_class': SymmetricActorCriticPolicy})
    opponent_model.policy.force_flip = False # Opponent sees raw view (Right side)
    
    # 3. Wrap the Env to freeze the opponent
    env = LeftSpecialistVecWrapper(env, opponent_model)
    env = VecMonitor(env)
    env.render_mode = "rgb_array"
    
    # Optional: Recorder (Only records the Left agent's view)
    video_folder = f"runs/{run.id}/videos"
    os.makedirs(video_folder, exist_ok=True)
    env = VecVideoRecorder(env, video_folder=video_folder,
                           record_video_trigger=lambda x: x % 50000 == 0,
                           video_length=2000, name_prefix=CONFIG["model_name"])
    env.render_mode = "rgb_array"

    # 4. Load the STUDENT Model (To be trained)
    # We load the same weights as a starting point
    print(f"Loading Student from: {CONFIG['opponent_path']}")
    model = A2C.load(CONFIG["opponent_path"],
                     env=env,
                     tensorboard_log=f"runs/{run.id}",
                     custom_objects={'policy_class': SymmetricActorCriticPolicy})
    
    # IMPORTANT: The STUDENT does NOT need force_flip=True here.
    # Why? Because 'make_marl_env' already flipped the pixels for 'second_0'.
    model.policy.force_flip = False

    print(f"\n>>> Starting Specialist Training...")

    wandb_callback = WandbCallback(verbose=2)
    checkpoint_callback = CheckpointCallback(
        save_freq=100_000,
        save_path=CONFIG["export_path"],
        name_prefix=CONFIG["model_name"]
    )

    model.learn(
        total_timesteps=CONFIG["total_timesteps"],
        callback=[wandb_callback, checkpoint_callback],
        reset_num_timesteps=True # Reset stats because this is a new "Specialist" run
    )

    model.save(f"{CONFIG['export_path']}{CONFIG['model_name']}_final")
    env.close()
    run.finish()

if __name__ == "__main__":
    resume_training()
