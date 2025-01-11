import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import pandas as pd

# Check for GPU availability and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

env = gym.make("CartPole-v1")

# Q Network using Sequential
class QNetwork(nn.Module):
    def __init__(self, num_features):
        super(QNetwork, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(num_features, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 2)
        )

    def forward(self, x):
        return self.model(x)

# Instantiate networks and move to device
q_net = QNetwork(num_features=4).to(device)
target_net = QNetwork(num_features=4).to(device)
target_net.load_state_dict(q_net.state_dict())
target_net.eval()

# Loss and optimizer
loss_fn = nn.MSELoss()  # Equivalent to Huber loss
optimizer = optim.Adam(q_net.parameters())

# Parameters
EPSILON = 1.0
EPSILON_DECAY = 1.005
GAMMA = 0.99
MAX_TRANSITIONS = 100_000
NUM_EPISODES = 400
BATCH_SIZE = 64
TARGET_UPDATE_AFTER = 4
LEARN_AFTER_STEPS = 3

REPLAY_BUFFER = deque(maxlen=MAX_TRANSITIONS)

# Inserts transition into Replay Buffer
def insert_transition(transition):
    REPLAY_BUFFER.append(transition)

# Samples a batch of transitions from Replay Buffer randomly
def sample_transitions(batch_size=16):
    sampled = random.sample(REPLAY_BUFFER, batch_size)
    states, actions, rewards, next_states, dones = zip(*sampled)
    return (
        torch.tensor(states, dtype=torch.float32, device=device),
        torch.tensor(actions, dtype=torch.int64, device=device),
        torch.tensor(rewards, dtype=torch.float32, device=device),
        torch.tensor(next_states, dtype=torch.float32, device=device),
        torch.tensor(dones, dtype=torch.bool, device=device),
    )

# Epsilon-Greedy Policy
def policy(state, explore=0.0):
    if np.random.rand() <= explore:
        return np.random.randint(2)
    with torch.no_grad():
        state_tensor = torch.tensor(state, dtype=torch.float32, device=device)
        return q_net(state_tensor).argmax().item()

# Reward function
def calculate_reward(state):
    reward = -1.0
    if -0.5 <= state[0] <= 0.5 and -1 <= state[1] <= 1 and -0.07 <= state[2] <= 0.07 and -0.525 <= state[3] <= 0.525:
        reward = 1.0
    return reward

# Gathering Random initial states for Average Q Metric
random_states = []
done = False
state = env.reset()[0]
for _ in range(20):
    if not done:
        random_states.append(state)
        state, _, done, _, _ = env.step(policy(state))
random_states = torch.tensor(random_states, dtype=torch.float32, device=device)

# Get Q values for states
def get_q_values(states):
    with torch.no_grad():
        return q_net(states).max(dim=1)[0]

# Initializations before training
step_counter = 0
metric = {"episode": [], "length": [], "total_reward": [], "avg_q": [], "exploration": []}

for episode in range(NUM_EPISODES):
    state = env.reset()[0]
    done = False
    trunc = False
    total_rewards = 0
    episode_length = 0

    while not (done or trunc):
        action = policy(state, EPSILON)
        next_state, _, done, _, _ = env.step(action)
        reward = calculate_reward(next_state)
        insert_transition((state, action, reward, next_state, done))
        state = next_state
        step_counter += 1

        if step_counter >= BATCH_SIZE and step_counter % LEARN_AFTER_STEPS == 0:
            states, actions, rewards, next_states, dones = sample_transitions(BATCH_SIZE)

            # Compute targets
            with torch.no_grad():
                next_action_values = target_net(next_states).max(dim=1)[0]
                targets = rewards + GAMMA * next_action_values * (~dones)

            # Compute Q values for the selected actions
            preds = q_net(states)
            current_values = preds.gather(1, actions.unsqueeze(1)).squeeze()

            # Compute loss and backpropagate
            loss = loss_fn(current_values, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if step_counter % TARGET_UPDATE_AFTER == 0:
            target_net.load_state_dict(q_net.state_dict())

        if episode_length >=500: trunc=True

        total_rewards += reward
        episode_length += 1

    # Save metrics
    avg_q = get_q_values(random_states).mean().item()
    metric["episode"].append(episode)
    metric["length"].append(episode_length)
    metric["total_reward"].append(total_rewards)
    metric["avg_q"].append(avg_q)
    metric["exploration"].append(EPSILON)
    EPSILON /= EPSILON_DECAY

    print(f"episode: {episode}, episode_length: {episode_length}, total_reward: {total_rewards}, avg_q: {avg_q}")
    pd.DataFrame(metric).to_csv("metric.csv", index=False)

env.close()
torch.save(q_net.state_dict(), "dqn_q_net.pth")