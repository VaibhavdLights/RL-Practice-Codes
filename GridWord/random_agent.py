import gym
import numpy as np

cliffEnv = gym.make('CliffWalking-v0', render_mode="human")

ACTION = ['up', 'right', 'down', 'left']
state, info = cliffEnv.reset()

for _ in range(1000):
    cliffEnv.render()
    action = np.random.randint(low=0, high=4)
    print(f"{state} ->> {ACTION[action]}")
    state, reward, terminated, truncated, info = cliffEnv.step(action)

    if terminated or truncated:
        state, info = cliffEnv.reset()


cliffEnv.close()