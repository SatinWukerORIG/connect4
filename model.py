import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.relu(out + residual)
        return out

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

            ResidualBlock(64),
            ResidualBlock(64),
            ResidualBlock(64),

            nn.Flatten(),

            nn.Linear(64 * 6 * 7, 128),
            nn.ReLU(),

            nn.Linear(128, 7)
        )

    def forward(self, x):
        return self.network(x)

