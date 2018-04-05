import numpy as np
import gym
import keras
from keras.models import Sequential
from keras.layers import Dense
from collections import deque

class QNetwork:
    def __init__(self, hidden_size = 16):
        self.model = model = Sequential()
        model.add(Dense(hidden_size, activation="relu", input_dim=4))
        model.add(Dense(hidden_size, activation="relu"))
        model.add(Dense(2, activation="linear"))
        model.compile(loss='mse', optimizer=keras.optimizers.Adam(lr=0.001))

    def replay(self, memory, batch_size = 32, epochs = 1):
        for i in np.arange(epochs):
            x = np.zeros((batch_size, 4))
            y = np.zeros((batch_size, 2))
            mini_batch = memory.sample(batch_size)

            for i, (current_state, current_action, current_reward, next_state) in enumerate(mini_batch):
                s = self.model.predict(current_state)[0]
                reward = current_reward

                if next_state is not None:
                    alpha = 0.2
                    gamma = 0.99

                    s_1 = self.model.predict(next_state)[0]
                    reward = (1-alpha) * s[current_action] + alpha * (reward + gamma * max(s_1))

                x[i] = current_state
                y[i] = s
                y[i][current_action] = reward

            self.model.fit(x, y, epochs=1, verbose=0)

    def get_action(self, state, episode):
        # epsiron = 1 - (0.9 / 1000000) * episode
        epsiron = 1 - (0.9 / 1000) * episode
        epsiron = max(epsiron, 0.1)
        if epsiron <= np.random.uniform(0, 1):
            pred = self.model.predict(state)[0]
            return np.argmax(pred)
        else:
            return np.random.choice([0, 1])

class Memory:
    def __init__(self, max_memory_size=10000):
        self.buffer = deque(maxlen=max_memory_size)

    def add(self, exp):
        self.buffer.append(exp)

    def sample(self, batch_size):
        idx = np.random.choice(np.arange(len(self.buffer)), size=batch_size, replace=False)
        return [self.buffer[i] for i in idx]

    def len(self):
        return len(self.buffer)

num_episodes = 5000
max_number_of_steps = 200
num_consecutive_iterations = 100
goal_average_steps = 195

last_time_steps = np.zeros(num_consecutive_iterations)
env = gym.make('CartPole-v0')
qn = QNetwork()
memory = Memory()

for episode in range(num_episodes):
    # 環境の初期化
    state = env.reset()
    state = np.reshape(state, [1, 4])

    for t in range(max_number_of_steps):
        # env.render()

        action = qn.get_action(state, episode)
        next_state, reward, done, info = env.step(action)
        next_state = np.reshape(next_state, [1, 4])

        if done:
            next_state = None
            reward = -1
        else:
            reward = 0

        memory.add((state, action, reward, next_state))
        state = next_state

        if done:
            print("{} episode finished after {} time steps / mean {}".format(episode + 1, t + 1, last_time_steps.mean()))
            last_time_steps = np.hstack((last_time_steps[1:], [t + 1]))
            break

    if memory.len() > 32:
        qn.replay(memory, batch_size=32)

    if (last_time_steps.mean() > goal_average_steps):
        print("episode {} train agent successfuly!".format(episode))
        break
env.close()
