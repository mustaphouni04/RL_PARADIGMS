import torch
import numpy as np
import collections

Experience = collections.namedtuple('Experience', field_names=['state', 'action', 'reward', 'done', 'new_state'])

class Agent:
    def __init__(self, env, exp_replay_buffer):
        self.env = env
        self.exp_replay_buffer = exp_replay_buffer
        self._reset()

    def _reset(self):
        self.current_state = self.env.reset()
        self.total_reward = 0.0

    def step(self, net, epsilon=0.0, device="cpu"):
        done_reward = None
        if np.random.random() < epsilon:
            action = self.env.action_space.sample()
        else:
            if isinstance(self.current_state, tuple):
                self.current_state = self.current_state[0]

            state_ = np.array([self.current_state])
            state = torch.tensor(state_).to(device)
            q_vals = net(state)
            _, act_ = torch.max(q_vals, dim=1)
            action = int(act_.item())

        new_state, reward, is_done, _, _ = self.env.step(action)
        self.total_reward += reward

        exp = Experience(self.current_state, action, reward, is_done, new_state)
        self.exp_replay_buffer.append(exp)
        self.current_state = new_state
        if is_done:
            done_reward = self.total_reward
            self._reset()

        return done_reward
