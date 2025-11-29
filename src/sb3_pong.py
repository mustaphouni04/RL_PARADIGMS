import wandb
import numpy as np 
import supersuit as ss 
from pettingzoo.atari import pong_v3
from datetime import datetime
import toml
import gymnasium as gym 
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecTransposeImage
from stable_baselines3 import A2C, PPO
from stable_baselines3.ppo.policies import MlpPolicy
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, StopTrainingOnRewardThreshold, CallbackList
from wandb.integration.sb3 import WandbCallback

class SeparateAgentLogger(BaseCallback):
    def __init__(self, verbose=0):
        super(SeparateAgentLogger, self).__init__(verbose)

    def _on_step(self) -> bool:
        dones = self.locals['dones']
        infos = self.locals['infos']

        for idx, info in enumerate(infos):
            if 'episode' in info:
                reward = info['episode']['r']
                length = info['episode']['l']
                
                agent_name = "Right_Paddle" if idx == 0 else "Left_Paddle"
                
                wandb.log({
                    f"{agent_name}/reward": reward,
                    f"{agent_name}/ep_length": length,
                    "global_step": self.num_timesteps
                })
        return True

print("Using Gymnasium version {}".format(gym.__version__))

def make_env(render_mode="rgb_array"):
    env = pong_v3.parallel_env(num_players=2, render_mode=render_mode)

    env = ss.color_reduction_v0(env, mode="B")
    env = ss.resize_v1(env, x_size=84, y_size=84)
    env = ss.frame_stack_v1(env, 4)
    #env = ss.dtype_v0(env, dtype=np.float32)
    #env = ss.normalize_obs_v0(env, env_min=0, env_max=1)

    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, 1, num_cpus=0, base_class='stable_baselines3')

    env = VecTransposeImage(env)
    env = VecMonitor(env)

    return env
try:
    config = toml.load("../configs/sb3_config.toml")
except toml.decoder.TomlDecodeError:
    config = {"env": {"policy_type": "CnnPolicy", "total_timesteps": 2000000, "export_path": "./models/"}}

def train_model(env, model_name, config):
    run = wandb.init(
        project="Step2-RLProject",
        config=config,
        name = model_name,
        sync_tensorboard=True, 
        save_code=True,       
        monitor_gym=True
    )

    print("\n>>> Creating and training model '{}'...".format(model_name))

    tensorboard_log = f"runs/{run.id}"
    if model_name == "a2c":
        model = A2C(config["env"]["policy_type"], env, verbose=0, tensorboard_log=tensorboard_log)
    elif model_name == "ppo":
        model = PPO(config["env"]["policy_type"], env, verbose=0, tensorboard_log=tensorboard_log)
    else:
        print("Error, unknown model ({})".format(model_name))
        return None
    
    wandb_callback = WandbCallback(verbose=2)
    
    agent_logger = SeparateAgentLogger()
    
    callback_list = CallbackList([wandb_callback, agent_logger])

    t0 = datetime.now()
    
    model.learn(total_timesteps=config["env"]["total_timesteps"], callback=callback_list)
    
    t1 = datetime.now()
    print(f'>>> Training time: {t1-t0}')

    save_path = f"{config['env']['export_path']}{model_name}"
    model.save(save_path)
    print(f"Model exported at '{save_path}'")
    
    run.finish()

if __name__ == "__main__":
    env = make_env()
    train_model(env, "ppo", config)
