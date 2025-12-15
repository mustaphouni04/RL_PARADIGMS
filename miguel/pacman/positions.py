#not important, this is just to test position resets on death

import gymnasium as gym

env = gym.make("ALE/MsPacman-v5", full_action_space=True, render_mode="rgb_array")
obs, info = env.reset()

#save position to detect reset
initial_pos = None
if "ale.lives" in info:
    prev_lives = info["ale.lives"]
else:
    prev_lives = 3  # fallback
print(prev_lives)

done = False
step = 0

while not done and step < 500:
    action = env.action_space.sample()  #random action 
    obs, reward, terminated, truncated, info = env.step(action)

    #save position to detect reset
    pacman_pos = info.get("pacman_pos", None)

    #detect death
    if initial_pos is None and pacman_pos is not None:
        initial_pos = pacman_pos
    elif pacman_pos is not None and pacman_pos == initial_pos and step > 0:
        print(f"pacman dead: {step}")

    step += 1
    if terminated or truncated:
        done = True

env.close()