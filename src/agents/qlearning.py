import numpy as np
import gymnasium as gym
import time
import math
import pandas as pd

env = gym.make("LunarLander-v2")

LEARNING_RATE = 0.1
DISCOUNT = 0.95

n_bins = [10, 10, 10, 10, 10, 10, 2, 2]

max_epsilon = 1.0
min_epsilon = 0.01
decay_rate = 0.001
print(f"low observation state = {env.observation_space.low}")
print(f"high observation state = {env.observation_space.high}")

q_table = np.random.uniform(low=0, high=1, size=(n_bins + [env.action_space.n]))

def get_discrete_state(state, bins=n_bins):
    #discretize them here.
    state = 0

bins = [
    np.linspace(-1.0, 1.0, n_bins[0]),
    np.linspace(-1.0, 1.0, n_bins[1]),
    np.linspace(-3.5, 3.5, n_bins[2]),
    np.linspace(-3.5, 3.5, n_bins[3]),
    np.linspace(-2.0, 2.0, n_bins[4]),
    np.linspace(-3.5, 3.5, n_bins[5]),
]

]



