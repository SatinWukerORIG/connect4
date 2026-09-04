import os

import torch


def _default_device():
    """CUDA on Colab, MPS on an Apple GPU, else CPU. Override with DEVICE=cpu."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


DEVICE = os.environ.get("DEVICE") or _default_device()

EPISODES = 600
BATCH_SIZE = 64
MIN_REPLAY_SIZE = 500
MAX_REPLAY_SIZE = 200_000
TARGET_UPDATE_FREQ = 500

GAMMA = 0.99
EPSILON_DECAY_STEPS = 20_000
EPSILON_MIN = 0.005
LR = 1e-4
CHECKPOINT_DIR = "checkpoint"
CHECKPOINT_PATH = None
MAX_CHECKPOINTS = 10

OPPONENT_POOL_SIZE = 8          # past-version opponents kept resident in memory
OPPONENT_POOL_REFRESH_EVERY = 10  # re-scan the checkpoint dir every N draws
