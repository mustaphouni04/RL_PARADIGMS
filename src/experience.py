import collections
import numpy as np

class ExperienceReplay:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        self.buffer.append(experience)

    def sample(self, BATCH_SIZE):
        indices = np.random.choice(len(self.buffer), BATCH_SIZE, replace=False)
        states, actions, rewards, dones, next_states = zip(*[self.buffer[idx] for idx in indices])

        states = [state[0] if isinstance(state, tuple) else state for state in states]
        actions = [action for action in actions]
        rewards = [reward for reward in rewards]
        dones = [done for done in dones]
        next_states = [new_state[0] if isinstance(new_state, tuple) else new_state for new_state in next_states]

        return np.array(states), np.array(actions), np.array(rewards, dtype=np.float32), \
               np.array(dones, dtype=np.uint8), np.array(next_states)

    def sample_n_step(self, batch_size, n_step=3, gamma=0.99):
        """采样并计算 n 步回报"""
        if len(self.buffer) < batch_size + n_step:
            return None

        # 随机选择起始索引
        start_indices = np.random.choice(len(self.buffer) - n_step, batch_size, replace=False)

        states, actions, n_step_rewards, dones, next_states = [], [], [], [], []

        for start_idx in start_indices:
            # 计算 n 步回报
            n_step_return = 0
            for i in range(n_step):
                exp = self.buffer[start_idx + i]
                n_step_return += exp.reward * (gamma ** i)

                # 如果遇到终止状态，提前结束
                if exp.done:
                    break

            # 获取起始经验和最终经验
            start_exp = self.buffer[start_idx]
            final_exp = self.buffer[min(start_idx + n_step - 1, len(self.buffer) - 1)]

            states.append(start_exp.state[0] if isinstance(start_exp.state, tuple) else start_exp.state)
            actions.append(start_exp.action)
            n_step_rewards.append(n_step_return)
            dones.append(final_exp.done)
            next_states.append(final_exp.new_state[0] if isinstance(final_exp.new_state, tuple) else final_exp.new_state)

        return (np.array(states), np.array(actions), np.array(n_step_rewards, dtype=np.float32),
                np.array(dones, dtype=np.uint8), np.array(next_states))
