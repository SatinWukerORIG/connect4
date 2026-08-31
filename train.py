import random
from collections import deque
from pathlib import Path
import time

import torch
import numpy as np

import config
import environment


memory_buffer1 = deque(maxlen=config.MEMORY_SIZE)
memory_buffer2 = deque(maxlen=config.MEMORY_SIZE)

env = environment.Connect4Env()

total_steps = 0
for episode in range(config.EPISODES):
    state, _ = env.reset()
    episode_step = 0
    total_reward = 0

    done = False
    while not done:
        pass

