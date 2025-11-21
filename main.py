import time
import numpy as np
import collections
import datetime
import tomllib
import gymnasium as gym
from ale_py import ALEInterface
from torch import nn
import ale_py
import torch

from utils.utils import make_env
from models.conv_dqn import make_DQN, dueling_DQN
from src.experience import ExperienceReplay
from src.agent import Agent

def main():
    ale = ALEInterface()
    gym.register_envs(ale_py)

    print("Using Gym version {}".format(gym.__version__))

    with open("configs/dqn_config.toml", "rb") as f:
        cfg = tomllib.load(f)

    env = make_env(cfg["env"]["name"])
    device = cfg["env"]["device"]

    is_dueling = cfg["model"]["dueling"]
    is_double = cfg["model"]["double"]
    is_n_step = cfg["model"]["n_step"]
    is_basic = cfg["model"]["basic"]
    n_step_horizon = cfg["model"]["n_step_horizon"]

    replay_buffer = cfg["replay_buffer"]["capacity"]

    buffer = ExperienceReplay(replay_buffer)
    agent = Agent(env, buffer)

    lr = cfg["training"]["learning_rate"]
    batch_size = cfg["training"]["batch_size"]
    gamma = cfg["training"]["gamma"]
    eps_start = cfg["training"]["eps_start"]
    eps_decay = cfg["training"]["eps_decay"]
    eps_min = cfg["training"]["eps_min"]

    number_of_rewards_to_average = cfg["training"]["number_of_rewards_to_average"]
    mean_reward_bound = cfg["training"]["mean_reward_bound"]
    sync_target_network = cfg["training"]["sync_target_network"]

    epsilon = eps_start 
    total_rewards = []
    frame_number = 0

    if is_n_step:
        variant = "n-step_DQN"

        net = make_DQN(env.observation_space.shape, env.action_space.n).to(device)
        target_net = net

    elif is_dueling:
        variant = "Dueling_DQN"

        net = dueling_DQN(env.observation_space.shape, env.action_space.n).to(device)
        target_net = dueling_DQN(env.observation_space.shape, env.action_space.n).to(device)

    elif is_double:
        variant = "Double_DQN"
        net = make_DQN(env.observation_space.shape, env.action_space.n).to(device)
        target_net = make_DQN(env.observation_space.shape, env.action_space.n).to(device)

    elif is_basic:
        variant = "Vanilla_DQN"
        net = make_DQN(env.observation_space.shape, env.action_space.n).to(device)
        target_net = net

    print("Using variant:", variant) 

    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    while True:
        frame_number += 1
        epsilon = max(epsilon * eps_decay, eps_min)

        reward = agent.step(net, epsilon, device=device)
        if reward is not None:
            total_rewards.append(reward)

            mean_reward = np.mean(total_rewards[-number_of_rewards_to_average:])
            print(f"Frame:{frame_number} | Total games:{len(total_rewards)} | Mean reward: {mean_reward:.3f}  (epsilon used: {epsilon:.2f})")

            if mean_reward > mean_reward_bound:
                print(f"SOLVED in {frame_number} frames and {len(total_rewards)} games")
                break

        if len(buffer) < replay_buffer:
            continue
        
        if is_n_step:
            batch = buffer.sample_n_step(batch_size, n_step_horizon)
        else:
            batch = buffer.sample(batch_size)

        states_, actions_, rewards_, dones_, next_states_ = batch

        states = torch.tensor(states_).to(device)
        next_states = torch.tensor(next_states_).to(device)
        actions = torch.tensor(actions_).to(device)
        rewards = torch.tensor(rewards_).to(device)
        dones = torch.BoolTensor(dones_).to(device)

        Q_values = net(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)

        with torch.no_grad():
            next_state_values = target_net(next_states).max(1)[0]
            next_state_values[dones] = 0.0

        expected_Q_values = next_state_values * gamma + rewards
        loss = nn.MSELoss()(Q_values, expected_Q_values)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if frame_number % sync_target_network == 0:
            target_net.load_state_dict(net.state_dict())
        
    torch.save(net.state_dict(), cfg["env"]["name"] + f"{variant}.dat")

if __name__ == "__main__":
    main()
