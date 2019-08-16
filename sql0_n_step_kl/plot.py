import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

smoothing_window = 20

# dqn7_rewards = np.load('SQL0-sql0-durations.npy')[:1000]
dqn8_rewards = np.load('SQL0-sql0-durations.npy')
# dqn8_rewards = np.load('res/env4-sql0-durations.npy')

# print(dqn7_rewards)

# dqn7_smooth = pd.Series(dqn7_rewards).rolling(smoothing_window,min_periods=1).mean()
dqn8_smooth = pd.Series(dqn8_rewards).rolling(smoothing_window,min_periods=1).mean()

plt.figure(figsize=(10,5))
plt.title('Benchmark Training Results DQN vs Distral', fontsize='20')
plt.xlabel('Episodes ', fontsize='16')
plt.ylabel('Reward', fontsize='16')

# plt.ylim(0,100)

# plt.plot(dqn7_smooth, label="Env 7 - SQL")
plt.plot(dqn8_smooth, label="Env 8 - DQN")

plt.legend(loc='best', fontsize='20')
# plt.savefig('Benchmark-dqn-vs-distral78-reward.eps', format='eps', dpi=1000)
plt.show()
# plt.close()
