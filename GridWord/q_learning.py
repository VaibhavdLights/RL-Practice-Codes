import gym
import numpy as np
import pickle as pkl

cliffEnv = gym.make('CliffWalking-v0', render_mode="ansi")

q_table = np.zeros(shape=(48, 4))

def policy(state, explore=0.0):
    action = int(np.argmax(q_table[state]))
    if np.random.random() <= explore:
        action = np.random.randint(low=0, high=4)
    return action

# PARAMAETERS
EPSILON = 0.1
ALPHA = 0.1
GAMMA = 0.9
NUM_EPISODES=500

for episode in range(NUM_EPISODES):
    total_reward = 0
    episode_len = 0
    done = False
    state, info = cliffEnv.reset()

    while not done:
        action = policy(state, EPSILON)
        next_state, reward, done, _, __ = cliffEnv.step(action)
        next_action = policy(next_state)

        q_table[state][action] += ALPHA * (reward + GAMMA * q_table[next_state][next_action] - q_table[state][action])

        state = next_state
        total_reward += reward
        episode_len += 1
    print(f"Epsiode: {episode+1}, Episode Length: {episode_len}, Total Reward: {total_reward}")

cliffEnv.close()

pkl.dump(q_table, open("q_learning_q_table.pkl", "wb"))
print('Training Complete. Q-Table Saved.')