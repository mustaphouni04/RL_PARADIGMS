import numpy as np
import gymnasium as gym
from ale_py import ALEInterface
import ale_py
import torch

from utils.utils import make_env
from models.conv_dqn import make_DQN, dueling_DQN 

import torch
from PIL import Image

ENV_NAME = "PongNoFrameskip-v4"
model_path = "checkpoints/dueling_dqn/PongNoFrameskip-v4_dueling.dat"
visualize = False 
num_episodes = 50

ale = ALEInterface()
gym.register_envs(ale_py)

env = make_env(ENV_NAME, render_mode="rgb_array")  
net = dueling_DQN(env.observation_space.shape, env.action_space.n)
net.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
net.eval() 

episode_rewards = []
wins = 0

seed = 0
for ep in range(num_episodes):
    state, info = env.reset(seed=seed)
    total_reward = 0.0
    images = []

    while True:
        if visualize:
            img = env.render()
            if img is not None:
                images.append(Image.fromarray(img))

        state_tensor = torch.tensor(np.asarray([state]), dtype=torch.float32)
        with torch.no_grad():
            q_vals = net(state_tensor).numpy()[0]
        action = int(np.argmax(q_vals))

        state, reward, done, truncated, info = env.step(action)
        total_reward += reward
        if done or truncated:
            break

    print(f"Episode {ep+1}: Total reward = {total_reward}")
    episode_rewards.append(total_reward)

    if total_reward > 0:
        wins += 1

    if visualize and ep == 0 and images:
        images[0].save("video_double.gif", save_all=True, append_images=images[1:], duration=60, loop=0)
    seed += 1


avg_reward = np.mean(episode_rewards)
success_rate = (wins / num_episodes) * 100
print(f"\nAverage reward over {num_episodes} episodes: {avg_reward:.2f}")
print(f"Success rate: {success_rate:.2f}%")

env.close()
