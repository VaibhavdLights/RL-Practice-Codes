import gym
import torch
import torch.nn as nn
from gym import spaces
import numpy as np

env = gym.make("CarRacing-v2", continuous=False, render_mode="human")

class QNetwork(nn.Module):
    def __init__(self, observation_space: spaces.Box, feature_dim):
        super(QNetwork, self).__init__()
        n_channels = observation_space.shape[0]
        self.conv_layers = nn.Sequential(
            nn.Conv2d(n_channels, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
        )

        with torch.no_grad():
            dummy_input = torch.zeros(1, *observation_space.shape)
            n_flatten = self.conv_layers(dummy_input).shape[1]


        self.fc_layers = nn.Sequential(
            nn.Linear(n_flatten, 128),
            nn.ReLU(),
            nn.Linear(128, feature_dim),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x

observation_space = gym.spaces.Box(low=0, high=255, shape=(3, 96, 96), dtype=np.float32)
feature_dim = 5

q_net = QNetwork(observation_space=observation_space, feature_dim=feature_dim)
q_net.load_state_dict(torch.load("dqn_q_net.pth"))
q_net.eval()

def policy(state, explore=0.0):
    if np.random.rand() <= explore:
        return np.random.randint(5)

    with torch.no_grad():
        state_tensor = torch.tensor([state], dtype=torch.float32)
        state_tensor = state_tensor.permute(0,3,1,2)

        return q_net(state_tensor).argmax().item()

for episode in range(5):
    done = False
    state = env.reset()[0]

    while not done:
        action = policy(state)
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

env.close()