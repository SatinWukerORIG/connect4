import torch
import torch.nn as nn

class Connect4Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Conv2d(
                in_channels=2,
                out_channels=64,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),

            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),

            nn.Flatten(),

            nn.Linear(64 * 6 * 7, 128),
            nn.ReLU(),

            nn.Linear(128, 7)
        )

    def forward(self, x):
        return self.network(x)

