import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Define the environment
env = gym.make("CartPole-v1")

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


# Initialize Q-network and optimizer
q_net = MyNN(num_features=4)
optimizer = optim.Adam(q_net.parameters(), lr=0.001)

# Parameters
ALPHA = 0.001
EPSILON = 1.0
EPSILON_DECAY = 1.001
GAMMA = 0.99
NUM_EPISODES = 500


# Policy function
def policy(state, explore=0.0):
    if np.random.uniform() <= explore:
        # Random action (exploration)
        return np.random.randint(0, 2)
    else:
        # Greedy action (exploitation)
        with torch.no_grad():
            return torch.argmax(q_net(state)).item()


# Training loop
for episode in range(NUM_EPISODES):
    done = False
    total_reward = 0
    episode_length = 0

    # Reset environment and preprocess state
    state = env.reset()[0]
    state = torch.tensor([state], dtype=torch.float32)

    while not done:
        # Get action using policy
        action = policy(state, EPSILON)

        # Take step in the environment
        next_state, reward, done, _, _ = env.step(action)
        next_state = torch.tensor([next_state], dtype=torch.float32)

        # Compute target
        with torch.no_grad():
            next_q_values = q_net(next_state)
            next_action = policy(next_state, EPSILON)
            target = reward + (GAMMA * next_q_values[0][next_action] if not done else reward)

        # Compute loss
        current_q_values = q_net(state)
        delta = target - current_q_values[0][action]

        # Backpropagation
        optimizer.zero_grad()
        # current_q_values[0][action].backward(torch.tensor(delta).detach())
        loss = 0.5 * (delta ** 2)
        loss.backward()
        optimizer.step()

        # Update state and total reward
        state = next_state
        total_reward += reward
        episode_length += 1

    print(
        f"Episode: {episode}, Length: {episode_length}, Rewards: {total_reward:.2f}, Epsilon: {EPSILON:.4f}"
    )
    EPSILON /= EPSILON_DECAY

# Save the Q-network
torch.save(q_net.state_dict(), "q_learning_net_sequential.pth")
env.close()