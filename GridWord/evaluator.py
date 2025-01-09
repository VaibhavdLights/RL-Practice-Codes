import gym
import pickle as pkl
import numpy as np

cliffEnv = gym.make('CliffWalking-v0', render_mode="human")

q_table = pkl.load(open("q_learning_q_table.pkl", "rb"))

def policy(state, explore=0.0):
    action = int(np.argmax(q_table[state]))
    if np.random.random() <= explore:
        action = np.random.randint(low=0, high=4)
    return action

NUM_EPISODES = 5

for episode in range(NUM_EPISODES):
    total_reward = 0
    episode_len = 0
    done = False
    state, info = cliffEnv.reset()

    while not done:
        action = policy(state)
        state, reward, done, _, __ = cliffEnv.step(action)

        episode_len += 1
        total_reward += reward

    print(f"Epsiode: {episode+1}, Episode Length: {episode_len}, Total Reward: {total_reward}")

cliffEnv.close()

print(q_table)