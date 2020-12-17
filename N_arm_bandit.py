# N-armed bandit testbed
import numpy as np
import matplotlib.pyplot as plt

N_ARM = 10
EPISODES = 2000
PLAYS = 1000

#numpy random generator.
np_rand = np.random.default_rng()

class EpsGreedy:
    def __init__(self, epsilon=0.1):
        self.epsilon = epsilon
        self.reward_history = []
        self.sampled_action_values = np.zeros(shape=(N_ARM,))
        self.play_count = 0
        self.action_selection_counter = np.zeros(shape=(N_ARM, ))

    def play(self, true_action_values):
        self.take_action(self.sampled_action_values)
        self.play_count += 1


    def take_action(self, estimated_rewards):
        if np_rand.uniform() >= self.epsilon:
            best_action_idx = np.argmax(estimated_rewards)
            best_action_reward = np_rand.normal(true_action_values[best_action_idx], 1)
            self.reward_history.append(best_action_reward)

            self.sampled_action_values[best_action_idx] = \
                (self.sampled_action_values[best_action_idx] * self.action_selection_counter[best_action_idx]
                 + best_action_reward) / (self.action_selection_counter[best_action_idx] + 1)

            self.action_selection_counter[best_action_idx] += 1

        else:
            random_action_idx = np_rand.integers(N_ARM)
            random_action_reward = np_rand.normal(true_action_values[random_action_idx], 1)
            self.reward_history.append(random_action_reward)

            self.sampled_action_values[random_action_idx] = \
                (self.sampled_action_values[random_action_idx] * self.action_selection_counter[random_action_idx] +
                 random_action_reward) / (self.action_selection_counter[random_action_idx] + 1)

            self.action_selection_counter[random_action_idx] += 1



reward_list = [[], [], []]

for i in range(EPISODES):
    true_action_values = np_rand.uniform(-2, 2, size=(N_ARM,))
    greedy = EpsGreedy(epsilon=0)
    low_eps = EpsGreedy(epsilon=0.01)
    high_eps = EpsGreedy(epsilon=0.1)

    for _ in range(PLAYS):
        greedy.play(true_action_values)
        low_eps.play(true_action_values)
        high_eps.play(true_action_values)

    reward_list[0].append(greedy.reward_history)
    reward_list[1].append(low_eps.reward_history)
    reward_list[2].append(high_eps.reward_history)


fig = plt.figure()
plt.plot(range(PLAYS), np.average(reward_list[0], axis=0), label='eps-0')
plt.plot(range(PLAYS), np.average(reward_list[1], axis=0), label='eps-0.01')
plt.plot(range(PLAYS), np.average(reward_list[2], axis=0), label='eps-0.1')
plt.legend(loc='lower right')
plt.show()


