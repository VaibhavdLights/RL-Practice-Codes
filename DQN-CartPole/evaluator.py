import gym
import torch
import torch.nn as nn
import numpy as np

env = gym.make("CartPole-v1", render_mode="human")

# Q Network using Sequential
class QNetwork(nn.Module):
    def __init__(self, num_features):
        super(QNetwork, self).__init__()

        # Sequential model definition
        self.model = nn.Sequential(
            nn.Linear(num_features, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 2)
        )

    def forward(self, x):
        return self.model(x)

q_net = QNetwork(num_features=4)
q_net.load_state_dict(torch.load("dqn_q_net.pth"))
q_net.eval()

# Epsilon-Greedy Policy
def policy(state, explore=0.0):
    if np.random.rand() <= explore:
        return np.random.randint(2)
    with torch.no_grad():
        return q_net(torch.tensor(state, dtype=torch.float32)).argmax().item()

for episode in range(5):
    done = False
    state, _ = env.reset()
    state = torch.tensor([state], dtype=torch.float32)

    while not done:
        action = policy(state)
        next_state, reward, done, _, _ = env.step(action)
        state = torch.tensor([next_state], dtype=torch.float32)

env.close()