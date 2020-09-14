# N-armed bandit testbed
import numpy as np
import matplotlib.pyplot as plt

N_ARM = 10
EPISODES = 2000
PLAYS = 1000

class EpsGreedy:
    def __init__(self, epsilon=0.1):
        self.epsilon = epsilon
        self.reward_history = []
        self.sampled_reward_values = np.zeros(shape=(N_ARM, ))
        self.play_count = 0
        self.action_selection_counter = np.zeros(shape=(N_ARM, ))

    def play(self, true_action_values):
        estimated_rewards = self.estimate_rewards(true_action_values)
        self.take_action(estimated_rewards)
        self.play_count += 1

    def estimate_rewards(self, true_action_values):
        random_estimated_rewards = [np.random.normal(q_value, 1) for q_value in true_action_values]
        estimated_rewards = random_estimated_rewards.copy()

        for index in range(N_ARM):
            if self.sampled_reward_values[index]:
                estimated_rewards[index] = self.sampled_reward_values[index]

        return estimated_rewards

    def take_action(self, estimated_rewards):
        if np.random.uniform() >= self.epsilon:
            best_action_idx = np.argmax(estimated_rewards)
            best_action_reward = np.max(estimated_rewards)
            self.reward_history.append(best_action_reward)

            self.sampled_reward_values[best_action_idx] = \
                (self.sampled_reward_values[best_action_idx] * self.action_selection_counter[best_action_idx] + best_action_reward) \
                / (self.action_selection_counter[best_action_idx] + 1)

            self.action_selection_counter[best_action_idx] += 1

        else:
            random_action_idx = np.random.randint(N_ARM)
            self.reward_history.append(estimated_rewards[random_action_idx])

            self.sampled_reward_values[random_action_idx] = \
                (self.sampled_reward_values[random_action_idx] * self.action_selection_counter[random_action_idx] + estimated_rewards[random_action_idx]) \
                / (self.action_selection_counter[random_action_idx] + 1)

            self.action_selection_counter[random_action_idx] += 1

    def get_accum_history(self):
        accumulated_reward_history = np.cumsum(self.reward_history)
        return (range(self.play_count), accumulated_reward_history)

    def get_avg_history(self, avg_len=5):
        avg_range = [x_value for x_value in range(EPISODES) if x_value % avg_len == 0]
        average_reward_history = [np.average(self.reward_history[i:i+avg_len]) for i in avg_range]
        return (avg_range, average_reward_history)


reward_list = [[], [], []]

for i in range(EPISODES):
    true_action_values = np.random.normal(0, 1, size=(N_ARM,))
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


