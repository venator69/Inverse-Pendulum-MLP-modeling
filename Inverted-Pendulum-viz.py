import gymnasium as gym
import time
import numpy as np

env = gym.make("CartPole-v1", render_mode="human")

obs, info = env.reset()

while True:

    action = env.action_space.sample()

    obs, reward, terminated, truncated, info = env.step(action)

    # pole angle
    theta = obs[2]

    time.sleep(1/60)

    # stop when pole falls past 90 degrees
    if abs(theta) > np.pi/2:
        print("Pole fell past 90°")
        time.sleep(2)
        obs, info = env.reset()

env.close()