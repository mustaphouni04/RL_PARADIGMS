"""
Configuration file for PacMan RL project - FIXED VERSION
"""
import torch

class Config:
    # Environment
    ENV_ID = "ALE/Pacman-v5"
    RENDER_MODE = "rgb_array"
    
    # Training
    TOTAL_TIMESTEPS = 20_000_000  
    EVAL_FREQ = 500_000
    N_EVAL_EPISODES = 10
    SAVE_FREQ = 500_000
    
    # A2C Hyperparameters
    """A2C_PARAMS = {
        'learning_rate': 7e-4,
        'n_steps': 5,
        'gamma': 0.99,
        'gae_lambda': 1,
        'ent_coef': 0.03,
        'vf_coef': 0.25,
        'max_grad_norm': 0.5,
        'rms_prop_eps': 1e-5,
        'use_rms_prop': True,
        'normalize_advantage': True,
        'use_sde': False,
        'policy_kwargs': dict(
            normalize_images=True,  
        ),
        'device': 'cuda',
    }"""
    #optimized params
    A2C_PARAMS = {
        "learning_rate": 5.261006440824724e-05,
        "n_steps": 512,
        "gamma": 0.9860582530007191,
        "gae_lambda": 0.9683443509628686,
        "ent_coef": 0.00014080392595234795,
        "vf_coef": 0.47918921558902916,
        "max_grad_norm": 1.0,
        "normalize_advantage": True,
    }
    #optimized params
    PPO_PARAMS = {
        "learning_rate": 2.08e-05,
        "n_steps": 2048,
        "batch_size": 128,
        "n_epochs": 10,

        "gamma": 0.9867,
        "gae_lambda": 0.9222,

        "clip_range": 0.29,
        "ent_coef": 1.27e-4,
        "vf_coef": 0.88,
        "max_grad_norm": 0.82,

        # PPO Atari defaults
        "normalize_advantage": True,
    }
    
    # Wrappers
    FRAME_SKIP = 4
    FRAME_STACK = 4
    RESIZE_DIMS = (84, 84)
    
    # Paths
    MODEL_DIR = "models"
    LOG_DIR = "results/logs"
    VIDEO_DIR = "results/videos"
    PLOT_DIR = "results/plots"
    
    # Wandb - added for your code
    ENV_NAME = "ALE/Pacman-v5"  # Add this line for compatibility