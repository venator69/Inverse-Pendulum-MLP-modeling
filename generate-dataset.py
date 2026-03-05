import gymnasium as gym
import numpy as np
import pandas as pd

env = gym.make("CartPole-v1")

samples = 50000

data = []

obs, info = env.reset()

for _ in range(samples):

    # random control input
    action = env.action_space.sample()

    next_obs, reward, terminated, truncated, info = env.step(action)

    # record transition
    row = np.concatenate((obs, [action], next_obs))
    data.append(row)

    obs = next_obs

    # reset if episode finished
    if terminated or truncated:
        obs, info = env.reset()

env.close()

# convert to dataframe
columns = [
    "x", "x_dot", "theta", "theta_dot",
    "action",
    "x_next", "x_dot_next", "theta_next", "theta_dot_next"
]

df = pd.DataFrame(data, columns=columns)

df.to_csv("cartpole_dataset.csv", index=False)

print("Dataset saved:", df.shape)