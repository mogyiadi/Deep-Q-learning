import random
from collections import deque
import torch


class ReplayBuffer:
    def __init__(self, capacity=50000):
        self.buffer = deque(maxlen=capacity)

    def store(self, state, action, reward, next_state, done):
        self.buffer.append((state.cpu(), action, reward, next_state.cpu(), done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            torch.stack(states),
            torch.tensor(actions,  dtype=torch.long),
            torch.tensor(rewards,  dtype=torch.float32),
            torch.stack(next_states),
            torch.tensor(dones,    dtype=torch.float32),
        )

    def __len__(self):
        return len(self.buffer)
