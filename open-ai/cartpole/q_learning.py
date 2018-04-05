import numpy as np
import gym

num_episodes = 1000
max_number_of_steps = 200
num_consecutive_iterations = 100
goal_average_steps = 195

last_time_steps = np.zeros(num_consecutive_iterations)
env = gym.make('CartPole-v0')

q_table = np.random.uniform(-1, 1, [4 ** 4, 2])

def bins(min, max, num):
    return np.linspace(min, max, num+1)[1:-1]

def digitize(observation):
    cart_pos, cart_v, pole_angle, pole_v = observation
    digitized = [
        np.digitize(cart_pos, bins=bins(-2.4, 2.4, 4)),
        np.digitize(cart_v, bins=bins(-3, 3, 4)),
        np.digitize(pole_angle, bins=bins(-5, 5, 4)),
        np.digitize(pole_v, bins=bins(-2, 2, 4))
    ]
    return sum([x * (4 ** i) for i, x in enumerate(digitized)])

alpha = 0.2
gamma = 0.99
def get_action(current_state, current_action, next_observation, current_reward):
    next_state = digitize(next_observation)

    epsiron = 0.5 * (0.99 ** episode)
    if epsiron <= np.random.uniform(0, 1):
        next_action = np.argmax(q_table[next_state])
    else:
        next_action = np.random.choice([0,

        ])


    q_table[current_state, current_action] = (1-alpha) * q_table[current_state, current_action] +\
        alpha * (reward + gamma * q_table[next_state, next_action])

    return next_action, next_state

for episode in range(num_episodes):
    # 環境の初期化
    observation = env.reset()

    state = digitize(observation)
    action = np.argmax(q_table[state])

    score = 0
    for t in range(max_number_of_steps):
        env.render()


        observation, reward, done, info = env.step(action)

        if done:
            reward = -200

        action, state = get_action(state, action, observation, reward)

        score += reward

        if done:
            print("{} episode finished after {} time steps / mean {}".format(episode + 1, t + 1, last_time_steps.mean()))
            last_time_steps = np.hstack((last_time_steps[1:], [t + 1]))
            break

    if (last_time_steps.mean() > goal_average_steps):
        print("episode {} train agent successfuly!".format(episode))
        break
env.close()
