import gym
import torch
import torch.nn as nn
import numpy as np

env = gym.make("CartPole-v1", render_mode="human")

# Q Network using Sequential
class MyNN(nn.Module):
    def __init__(self, num_features):
        super(MyNN, self).__init__()

        # Sequential model definition
        self.model = nn.Sequential(
            nn.Linear(num_features, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )

    def forward(self, x):
        return self.model(x)

q_net = MyNN(num_features=4)
q_net.load_state_dict(torch.load("sarsa_q_net_sequential.pth"))
q_net.eval()

# Policy function
def policy(state, explore=0.0):
    if np.random.uniform() <= explore:
        # Random action (exploration)
        return np.random.randint(0, 2)
    else:
        # Greedy action (exploitation)
        with torch.no_grad():
            return torch.argmax(q_net(state)).item()

for episode in range(5):
    done = False
    state, _ = env.reset()
    state = torch.tensor([state], dtype=torch.float32)

    while not done:
        action = policy(state)
        next_state, reward, done, _, _ = env.step(action)
        state = torch.tensor([next_state], dtype=torch.float32)

env.close()