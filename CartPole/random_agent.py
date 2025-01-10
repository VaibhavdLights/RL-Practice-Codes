import gym
# import tensorflow as tf
import torch
import torch.nn as nn

env = gym.make("CartPole-v1", render_mode="human")

for episode in range(5):
    done = False
    state, info = env.reset()
    while not done:
        env.render()
        action = torch.randint(0, 2, size=(1,)).numpy()[0]
        state, reward, done, _, __ = env.step(action)

env.close()
