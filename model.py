import torch.nn as nn


class QAgent(nn.Module):
    def __init__(self, hidden_size=128):
        super().__init__()

        # Simple Sequential model
        self.network = nn.Sequential(
            nn.Linear(4, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 2)
        )

    def forward(self, state):
        return self.network(state)
