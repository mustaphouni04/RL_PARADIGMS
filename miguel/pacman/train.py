"""
A2C / PPO ATARI TRAINING + OPTUNA
"""

import argparse
import copy
import os
import sys
from typing import Dict, Any, Optional
import warnings

import optuna
from optuna.pruners import MedianPruner
import numpy as np
import torch
import wandb

from stable_baselines3 import A2C, PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecFrameStack
from stable_baselines3.common.callbacks import (
    EvalCallback, CheckpointCallback, CallbackList, BaseCallback
)
from stable_baselines3.common.evaluation import evaluate_policy

from config import Config
from wrappers import create_pacman_env

def setup_device():
    if torch.cuda.is_available():
        device = "cuda"
        print(f"GPU avalaible: {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        print("CPU only.")
    return device


# WANDB CALLBACKS
class WandbEpisodeCallback(BaseCallback):
    """callback to log episode rewards and lengths to WandB."""
    
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_reward = 0
        self.episode_length = 0

    def _on_step(self) -> bool:
        reward = float(self.locals["rewards"][0])
        done = bool(self.locals["dones"][0])

        self.episode_reward += reward
        self.episode_length += 1

        if done:
            wandb.log({
                "train/episode_reward": self.episode_reward,
                "train/episode_length": self.episode_length,
                "train/timesteps": self.num_timesteps
            })
            self.episode_reward = 0
            self.episode_length = 0
        
        return True


class OptunaReportCallback(BaseCallback):
    """to report intermediate results to Optuna and handle pruning."""
    
    def __init__(self, trial: optuna.Trial, eval_freq: int, n_eval_episodes: int = 5):
        super().__init__()
        self.trial = trial
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.eval_idx = 0
        self.is_pruned = False

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            #evalaute
            eval_env = self.training_env
            mean_reward, _ = evaluate_policy(
                self.model, eval_env, n_eval_episodes=self.n_eval_episodes
            )
            
            #report to optuna
            self.trial.report(mean_reward, self.eval_idx)
            self.eval_idx += 1
            
            #verify pruning
            if self.trial.should_prune():
                self.is_pruned = True
                return False
        
        return True


# ENV FACTORIES
def make_single_env(cfg: Config, test: bool = False, seed: Optional[int] = None):
    """creates a single environment instance."""
    def _init():
        env = create_pacman_env(cfg, test_mode=test)
        if seed is not None:
            env.seed(seed)
        return Monitor(env)
    return _init


def make_vec_env(cfg: Config, test: bool = False, n_envs: int = 1, use_subproc: bool = True):
    """
    to create a vectorized environment with frame stacking. 
    """
    if n_envs == 1 or not use_subproc:
        env = DummyVecEnv([make_single_env(cfg, test, seed=i) for i in range(n_envs)])
    else:
        env = SubprocVecEnv([make_single_env(cfg, test, seed=i) for i in range(n_envs)])
    
    # Frame stacking (verify FRAME_STACK exists in cfg)
    env = VecFrameStack(env, cfg.FRAME_STACK)
    return env


# PARAMETER HANDLING
def fix_params_for_images(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    to ensure that image observations are properly normalized.
    Adds 'normalize_images': True to policy_kwargs.
    """
    p = copy.deepcopy(params)
    
    if "policy_kwargs" not in p:
        p["policy_kwargs"] = {}
    
    p["policy_kwargs"]["normalize_images"] = True
    
    return p


def validate_ppo_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """IT validates PPO params, especially batch_size vs n_steps."""
    if "n_steps" in params and "batch_size" in params:
        if params["n_steps"] % params["batch_size"] != 0:
            #adjust batch_size
            n_steps = params["n_steps"]
            batch_size = params["batch_size"]
            valid_batch_sizes = [b for b in [64, 128, 256, 512, 1024] if n_steps % b == 0]
            
            if valid_batch_sizes:
                old_bs = batch_size
                params["batch_size"] = min(valid_batch_sizes, key=lambda x: abs(x - batch_size))
                print(f"batch_size of {old_bs} to {params['batch_size']} "
                      f"(must divide n_steps={n_steps})")
            else:
                raise ValueError(f"No valid batch_size found for n_steps={n_steps}")
    
    return params


# MODEL FACTORY
def make_model(typem: str, env, params: Dict[str, Any], cfg: Config, device: str = "auto"):
    params = copy.deepcopy(params)
    params.pop("device", None)

    common_kwargs = {
        "policy": "CnnPolicy",
        "env": env,
        "tensorboard_log": cfg.LOG_DIR,
        "verbose": 1,
        "device": device,
    }

    if typem == "a2c":
        return A2C(**common_kwargs, **params)
    elif typem == "ppo":
        params = validate_ppo_params(params)
        return PPO(**common_kwargs, **params)



# OPTUNA PARAM SPACES
def sample_a2c_params(trial: optuna.Trial) -> Dict[str, Any]:
    """to define the hyperparameter search space for A2C."""
    return {
        "learning_rate": trial.suggest_float("lr", 5e-5, 5e-3, log=True),
        "n_steps": trial.suggest_categorical("n_steps", [128, 256, 512, 1024]),
        "gamma": trial.suggest_float("gamma", 0.98, 0.9995),
        "gae_lambda": trial.suggest_float("gae_lambda", 0.90, 1.0),
        "ent_coef": trial.suggest_float("ent_coef", 1e-4, 5e-2, log=True),
        "vf_coef": trial.suggest_float("vf_coef", 0.3, 1.0),
        "max_grad_norm": trial.suggest_categorical("max_grad_norm", [0.5, 1.0, 2.0]),
        "normalize_advantage": trial.suggest_categorical("normalize_advantage", [True, False]),
    }


def sample_ppo_params(trial: optuna.Trial) -> Dict[str, Any]:
    """to define the hyperparameter search space for PPO."""
    n_steps = trial.suggest_categorical("n_steps", [128, 256, 512, 1024, 2048])
    
    batch_size = 128
    
    return {
        "learning_rate": trial.suggest_float("learning_rate", 5e-6, 5e-4, log=True),
        "n_steps": n_steps,
        "batch_size": batch_size,
        "n_epochs": trial.suggest_int("n_epochs", 3, 10),
        "gamma": trial.suggest_float("gamma", 0.98, 0.9995),
        "gae_lambda": trial.suggest_float("gae_lambda", 0.90, 1.0),
        "clip_range": trial.suggest_float("clip_range", 0.1, 0.3),
        "ent_coef": trial.suggest_float("ent_coef", 1e-4, 5e-2, log=True),
        "vf_coef": trial.suggest_float("vf_coef", 0.3, 1.0),
        "max_grad_norm": trial.suggest_float("max_grad_norm", 0.3, 2.0),
    }


def sample_params(trial: optuna.Trial, typem: str) -> Dict[str, Any]:
    """to sample hyperparameters based on the algorithm type."""
    if typem == "a2c":
        return sample_a2c_params(trial)
    elif typem == "ppo":
        return sample_ppo_params(trial)
    else:
        raise ValueError(f"{typem} not supported.")


# TRAIN FUNCTION
def train(
    cfg: Config,
    typem: str,
    params_override: Optional[Dict[str, Any]] = None,
    device: str = "auto",
    n_train_envs: int = 4,
    run_name: Optional[str] = None
):
    """
    it trains an RL agent using the specified algorithm and configuration.
    """
    if typem == "a2c":
        params = copy.deepcopy(cfg.A2C_PARAMS)
    elif typem == "ppo":
        params = copy.deepcopy(cfg.PPO_PARAMS)
    else:
        raise ValueError(f"{typem} not supported.")
    
    if params_override:
        params.update(params_override)
    
    params = fix_params_for_images(params)
    
    # Initialize WandB
    wandb_run_name = run_name or f"{typem}_final_train"
    wandb.init(
        project=cfg.WANDB_PROJECT,
        name=wandb_run_name,
        config=params,
        sync_tensorboard=True,
        reinit=True
    )
    
    try:
        #create envs
        print(f"\n{'='*50}")
        print(f"Training {typem.upper()} with {n_train_envs} environments in parallel")
        print(f"{'='*50}\n")
        
        env = make_vec_env(cfg, test=False, n_envs=n_train_envs, use_subproc=(n_train_envs > 1))
        eval_env = make_vec_env(cfg, test=True, n_envs=1, use_subproc=False)
        
        #create model
        model = make_model(typem, env, params, cfg, device)
        
        # Callbacks
        callbacks = CallbackList([
            EvalCallback(
                eval_env,
                best_model_save_path=os.path.join(cfg.MODEL_DIR, typem, "best"),
                eval_freq=max(cfg.EVAL_FREQ // n_train_envs, 1),
                n_eval_episodes=cfg.N_EVAL_EPISODES,
                deterministic=True,
                render=False,
                verbose=1
            ),
            CheckpointCallback(
                save_freq=max(cfg.SAVE_FREQ // n_train_envs, 1),
                save_path=os.path.join(cfg.MODEL_DIR, typem, "checkpoints"),
                name_prefix=f"{typem}_ckpt",
                verbose=1
            ),
            WandbEpisodeCallback()
        ])
        
        #train
        model.learn(
            total_timesteps=cfg.TOTAL_TIMESTEPS,
            callback=callbacks,
            progress_bar=True
        )
        
        #save final model
        final_path = os.path.join(cfg.MODEL_DIR, typem, "final")
        model.save(final_path)
        print(f"\nFinal model saved to: {final_path}")
        
        #final evaluation
        print("\nEvaluating final model...")
        mean_reward, std_reward = evaluate_policy(
            model, eval_env, n_eval_episodes=20, deterministic=True
        )
        print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
        
        wandb.log({
            "final/mean_reward": mean_reward,
            "final/std_reward": std_reward
        })
        
        return model
        
    except Exception as e:
        print(f"\n Error during training: {e}")
        raise
    
    finally:
        #cleanup
        wandb.finish()
        try:
            env.close()
            eval_env.close()
        except:
            pass


# OPTUNA OBJECTIVE
def objective(trial: optuna.Trial, cfg: Config, typem: str, hpo_steps: int, device: str, n_train_envs: int = 4, enable_pruning: bool = True) -> float:
    """
    to define the objective function for Optuna hyperparameter optimization.
    """
    sampled_params = sample_params(trial, typem)
    
    #initialize wandb
    wandb.init(
        project=cfg.WANDB_PROJECT,
        name=f"{typem}_trial_{trial.number}",
        config=sampled_params,
        reinit=True,
        tags=["optuna", typem]
    )
    
    env = None
    eval_env = None
    
    try:
        #process params
        params = fix_params_for_images(sampled_params)
        
        #create envs
        env = make_vec_env(cfg, test=False, n_envs=n_train_envs, use_subproc=(n_train_envs > 1))
        eval_env = make_vec_env(cfg, test=True, n_envs=1, use_subproc=False)
        
        #create model
        model = make_model(typem, env, params, cfg, device)
        
        # Callbacks
        callbacks = [WandbEpisodeCallback()]
        
        if enable_pruning:
            #create callback for pruning
            prune_callback = OptunaReportCallback(
                trial, 
                eval_freq=max(hpo_steps // 10, 5000),  #evaluate 10 times
                n_eval_episodes=3
            )
            callbacks.append(prune_callback)
        
        callbacks = CallbackList(callbacks)
        
        #train
        model.learn(total_timesteps=hpo_steps, callback=callbacks, progress_bar=False)
        
        #if pruning is enabled, check if trial was pruned
        if enable_pruning and prune_callback.is_pruned:
            raise optuna.TrialPruned()
        
        #evaluate
        mean_reward, std_reward = evaluate_policy(
            model, eval_env, n_eval_episodes=10, deterministic=True
        )
        
        wandb.log({
            "hpo/mean_reward": mean_reward,
            "hpo/std_reward": std_reward,
            "hpo/trial": trial.number
        })
        
        print(f"Trial {trial.number}: mean_reward = {mean_reward:.2f} +/- {std_reward:.2f}")
        
        return mean_reward  #the objective to maximize
        
    except optuna.TrialPruned:
        print(f"Trial {trial.number}: Pruned")
        raise
        
    except Exception as e:
        print(f"Trial {trial.number} failed: {e}")
        return float('-inf')  # Penalize failed trials
        
    finally:
        wandb.finish()
        if env is not None:
            try:
                env.close()
            except:
                pass
        if eval_env is not None:
            try:
                eval_env.close()
            except:
                pass


# MAIN
def main():
    parser = argparse.ArgumentParser(description="train RL models in Pacman-v5")
    
    #basic arguments
    parser.add_argument("--typem", type=str, default="ppo", choices=["a2c", "ppo"],
                       help=" type of RL algorithm to use")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"],
                       help="Computing device")
    parser.add_argument("--n-envs", type=int, default=4,
                       help="Number of parallel environments for training")
    parser.add_argument("--wandb-project", type=str, default="pacman-rl",
                       help="name of the WandB project")
    
    #arguments for HPO
    parser.add_argument("--optimize", action="store_true",
                       help="execute hyperparameter optimization with Optuna")
    parser.add_argument("--trials", type=int, default=100,
                       help="Number of trials for Optuna")
    parser.add_argument("--hpo-timesteps", type=int, default=1_000_000,
                       help="Timesteps per trial in HPO")
    parser.add_argument("--no-pruning", action="store_true",
                       help="desactivate pruning in Optuna trials")
    
    #arguments for final training
    parser.add_argument("--final-train", action="store_true",
                       help="train final model with best hyperparameters after HPO")
    parser.add_argument("--run-name", type=str, default=None,
                       help="name for the WandB run (final training)")
    
    args = parser.parse_args()
    
    #config
    cfg = Config()
    cfg.WANDB_PROJECT = args.wandb_project
    
    #setup device
    if args.device == "auto":
        device = setup_device()
    else:
        device = args.device
    
    print(f"Configuration:")
    print(args.typem.upper())
    print(f"  Device: {device}")
    print(f"  Parallel environments: {args.n_envs}")
    print(f"  WandB project: {args.wandb_project}")
    
    if args.optimize:
        print(f"Initiating hyperparameter optimization with Optuna...")
        print(f"   Trials: {args.trials}")
        print(f"   Timesteps per trial: {args.hpo_timesteps:,}")
        print(f"   Pruning: {'Desactivate' if args.no_pruning else 'Activate'}\n")
        
        #create study
        pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=3) if not args.no_pruning else None
        
        study = optuna.create_study(
            direction="maximize",
            pruner=pruner,
            study_name=f"{args.typem}_pacman_optimization"
        )
        
        #optimize
        study.optimize(
            lambda trial: objective(
                trial, cfg, args.typem, args.hpo_timesteps, 
                device, args.n_envs, enable_pruning=not args.no_pruning
            ),
            n_trials=args.trials,
            show_progress_bar=True,
            catch=(Exception,)
        )
        
        #report best trial
        print(f"\nBest trial: {study.best_trial.number}")
        print(f"Best reward: {study.best_value:.2f}")
        print(f"\nBest hyperparameters:")
        for key, value in study.best_trial.params.items():
            print(f"  {key}: {value}")
        
        if args.final_train:
            print("\nTraining final model with best hyperparameters...\n")
            train(
                cfg, 
                args.typem, 
                params_override=study.best_trial.params,
                device=device,
                n_train_envs=args.n_envs,
                run_name=args.run_name or f"{args.typem}_final_optimized"
            )
    
    else:
        #final training without HPO
        print("\nTraining final model with default hyperparameters...\n")
        train(
            cfg, 
            args.typem,
            device=device,
            n_train_envs=args.n_envs,
            run_name=args.run_name
        )
    
    print("\nend of script.")


if __name__ == "__main__":
    main()