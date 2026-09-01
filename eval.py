import random
from pathlib import Path

import torch

import config
import environment
from agent import Agent

eval_path = "checkpoint/best_model.pth"

eval_env = environment.Connect4Env()
agent = Agent()
agent.load(eval_path)


