"""
This file contains the implementation of the experience replay buffer.
"""
from collections import deque
import numpy as np
import random

class ExperienceReplay:
    def __init__(self, max_size: int, seed: int=None):
        """ Initialize ExperienceReplay class
        Args:
            max_size (int): maximum size of buffer
            seed (int): random seed
        """
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
        self.position = 0
        if seed is not None:
            random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        # Add experience tuple (state, action, reward, next_state, done) to buffer
        if len(self.buffer) < self.max_size:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.max_size

    def sample(self, batch_size):
        # Sample batch_size experiences from buffer
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        # Return number of experiences in buffer
        return len(self.buffer)