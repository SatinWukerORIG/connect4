import random
from pathlib import Path

import torch

import config
import environment
from agent import Agent


def main():
    eval_path = "checkpoint/best_model.pth"

    eval_env = environment.Connect4Env()
    agent = Agent()
    if Path(eval_path).exists():
        agent.load(eval_path)
        print(f"Loaded model from {eval_path}")

if __name__ == "__main__":
    main()
