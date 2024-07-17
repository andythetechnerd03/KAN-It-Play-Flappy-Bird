"""
 This is the main file of the project.
"""

import argparse
import os
import yaml
import torch

from src.agent import Agent

def parse_args():
    """ Parse command line arguments
    Returns:
        argparse.Namespace: command line arguments
    """
    parser = argparse.ArgumentParser(description="Run the agent")
    parser.add_argument("--env", type=str, default="cartpole", help="environment to run the agent")
    parser.add_argument("--train", action="store_true", help="train the agent")
    parser.add_argument("--render", action="store_true", help="whether to render the environment (a.k.a. display the gameplay),\
                        use it for testing only.")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    agent = Agent(args.env)
    if args.train:
        agent.run(train=args.train, render=False)
    else:
        agent.test(num_episodes=1000, render=args.render)