"""
Atari wrappers for Pacman-v5 (Gymnasium + SB3)
"""

import gymnasium as gym
import numpy as np
import cv2
from collections import deque
from gymnasium import spaces


# 1. No-op reset

class NoopResetEnv(gym.Wrapper):
    def __init__(self, env, noop_max=30):
        super().__init__(env)
        self.noop_max = noop_max
        self.noop_action = 0  # NOOP Atari

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        noops = np.random.randint(1, self.noop_max + 1)

        for _ in range(noops):
            obs, _, terminated, truncated, info = self.env.step(self.noop_action)
            if terminated or truncated:
                obs, info = self.env.reset(**kwargs)

        return obs, info


# 2. GhostLifeEnv HYBRID 

class GhostLifeEnvHybrid(gym.Wrapper):
    """
    only end episode on real death (not timeout)
    """
    def __init__(self, env, death_reward_threshold=-200, timeout_steps=150):
        super().__init__(env)
        self.lives = 0
        self.was_real_done = True
        self.death_reward_threshold = death_reward_threshold
        self.timeout_steps_threshold = timeout_steps
        self.steps_without_pellet = 0

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        real_done = terminated or truncated
        current_lives = info.get("lives", 0)

        if reward > 0:
            self.steps_without_pellet = 0
        else:
            self.steps_without_pellet += 1

        #detect life lost
        lost_life = (current_lives < self.lives) and (current_lives > 0)

        if lost_life:
            #hybrid logic
            is_timeout_by_reward = reward > self.death_reward_threshold
            is_timeout_by_steps = (
                self.steps_without_pellet >= self.timeout_steps_threshold
            )

            if is_timeout_by_reward or is_timeout_by_steps:
                #instead of death, it's a timeout
                terminated = False
            else:
                # real death
                terminated = True

            self.steps_without_pellet = 0
        else:
            terminated = terminated or truncated

        self.was_real_done = real_done
        self.lives = current_lives

        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        if self.was_real_done:
            obs, info = self.env.reset(**kwargs)
        else:
            # Soft reset
            obs, _, _, _, info = self.env.step(0)

        self.lives = info.get("lives", 0)
        self.steps_without_pellet = 0

        return obs, info


# 3. MaxAndSkipEnv
class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env, skip=4):
        super().__init__(env)
        self._skip = skip
        self._obs_buffer = deque(maxlen=2)

    def step(self, action):
        total_reward = 0.0
        terminated = False
        truncated = False

        for _ in range(self._skip):
            obs, reward, term, trun, info = self.env.step(action)
            self._obs_buffer.append(obs)
            total_reward += reward
            terminated |= term
            truncated |= trun
            if terminated or truncated:
                break

        max_frame = np.maximum(self._obs_buffer[0], self._obs_buffer[-1])
        return max_frame, total_reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._obs_buffer.clear()
        self._obs_buffer.append(obs)
        return obs, info


# 4. Resize + Grayscale
class ResizeAndGrayScale(gym.ObservationWrapper):
    def __init__(self, env, shape=(84, 84)):
        super().__init__(env)
        self.shape = tuple(shape)

        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(1,) + self.shape,
            dtype=np.uint8,
        )

    def observation(self, obs):
        resized = cv2.resize(obs, self.shape, interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
        return gray[np.newaxis, ...]


# 5. ClipRewardEnv
class ClipRewardEnv(gym.RewardWrapper):
    def reward(self, reward):
        return np.sign(reward)

# 6. MaxEpisodeSteps
class MaxEpisodeSteps(gym.Wrapper):
    def __init__(self, env, max_steps=6000):
        super().__init__(env)
        self.max_steps = max_steps
        self.steps = 0

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.steps += 1

        if self.steps >= self.max_steps:
            truncated = True

        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        self.steps = 0
        return self.env.reset(**kwargs)


# Pacman environment with wrappers
def create_pacman_env(config, test_mode=False):
    env = gym.make(
        config.ENV_ID,
        obs_type="rgb",
        frameskip=1,
        full_action_space=False,  
        repeat_action_probability=0.0,
        render_mode=config.RENDER_MODE if test_mode else None,
    )

    if not test_mode:
        env = NoopResetEnv(env, noop_max=30)
        env = GhostLifeEnvHybrid(
            env,
            death_reward_threshold=-200,
            timeout_steps=150,
        )

    env = MaxAndSkipEnv(env, skip=config.FRAME_SKIP)
    env = ResizeAndGrayScale(env, config.RESIZE_DIMS)

    if not test_mode:
        env = ClipRewardEnv(env)

    env = MaxEpisodeSteps(env, max_steps=6000)

    print(f"[wrappers] final obs space: {env.observation_space}")

    return env
