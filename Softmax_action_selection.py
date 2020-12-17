# N-armed bandit testbed
import numpy as np
import matplotlib.pyplot as plt

N_ARM = 10
EPISODES = 2000
PLAYS = 1000

#numpy random generator.
np_rand = np.random.default_rng()



class SoftmaxBandit:

    def __init__(self, temp=1.0):
        self.temp = temp
        self.reward_history = []
        self.sampled_action_values = np.zeros(shape=(N_ARM,))
        self.play_count = 0
        self.action_selection_counter = np.zeros(shape=(N_ARM,))

    def play(self):
        self.take_action(self.sampled_action_values)
        self.play_count += 1


    def take_action(self, estimated_rewards):

        action_idx = np_rand.choice(range(N_ARM), p=self.gibbs_pd(estimated_rewards))
        action_reward = np_rand.normal(true_action_values[action_idx], 1)
        self.reward_history.append(action_reward)

        self.sampled_action_values[action_idx] = \
            (self.sampled_action_values[action_idx] * self.action_selection_counter[action_idx]
             + action_reward) / (self.action_selection_counter[action_idx] + 1)

        self.action_selection_counter[action_idx] += 1

    def gibbs_pd(self, estimation):
        power = estimation / self.temp
        return np.exp(power) / np.sum(np.exp(power), axis=0)


reward_list = [[], [], [], []]
temps = []

for i in range(EPISODES):
    true_action_values = np_rand.uniform(-2, 2, size=(N_ARM,))
    bandit = SoftmaxBandit(temp=0.1)
    bandit1 = SoftmaxBandit(temp=0.2)
    bandit2 = SoftmaxBandit(temp=0.3)
    bandit3 = SoftmaxBandit(temp=0.4)
    temps.extend([bandit.temp, bandit1.temp, bandit2.temp,
                  bandit3.temp])

    for _ in range(PLAYS):
        bandit.play()
        bandit1.play()
        bandit2.play()
        bandit3.play()


    reward_list[0].append(bandit.reward_history)
    reward_list[1].append(bandit1.reward_history)
    reward_list[2].append(bandit2.reward_history)
    reward_list[3].append(bandit3.reward_history)


fig = plt.figure()
for reward_list, temp in zip(reward_list, temps):
    plt.plot(range(PLAYS), np.average(reward_list, axis=0), label=f'temp-{temp}')

plt.legend(loc='lower right')
plt.show()


