import gym
import torch
import torch.nn as nn
import numpy as np

env = gym.make("LunarLander-v2", render_mode="human")

# Q Network using Sequential
class QNetwork(nn.Module):
    def __init__(self, num_features):
        super(QNetwork, self).__init__()

        # Sequential model definition
        self.model = nn.Sequential(
            nn.Linear(num_features, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 4)
        )

    def forward(self, x):
        return self.model(x)

q_net = QNetwork(num_features=8)
q_net.load_state_dict(torch.load("dqn_q_net.pth"))
q_net.eval()

# Epsilon-Greedy Policy
def policy(state, explore=0.0):
    if np.random.rand() <= explore:
        return np.random.randint(4)
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