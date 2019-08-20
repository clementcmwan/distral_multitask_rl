import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

smoothing_window = 20

distral4_rewards = np.load('Distral_1col-distral-2col-durations.npy')[0][:1000]
distral5_rewards = np.load('Distral_1col-distral-2col-durations.npy')[1][:1000]
distral6_rewards = np.load('Distral_1col-distral-2col-durations.npy')[2][:1000]
distral7_rewards = np.load('Distral_1col-distral-2col-durations.npy')[3][:1000]
distral8_rewards = np.load('Distral_1col-distral-2col-durations.npy')[4][:1000]

# distral4_rewards = np.asarray(distral4_rewards)
# distral4_rewards[distral4_rewards > 100] = 100

distral4_smooth = pd.Series(distral4_rewards).rolling(smoothing_window,min_periods=5).mean()
distral5_smooth = pd.Series(distral5_rewards).rolling(smoothing_window,min_periods=5).mean()
distral6_smooth = pd.Series(distral6_rewards).rolling(smoothing_window,min_periods=5).mean()
distral7_smooth = pd.Series(distral7_rewards).rolling(smoothing_window,min_periods=5).mean()
distral8_smooth = pd.Series(distral8_rewards).rolling(smoothing_window,min_periods=5).mean()


# dqn4_rewards = np.load('../baselines/env4-sql0-rewards.npy')[:1000]
# dqn5_rewards = np.load('../baselines/env5-sql0-rewards.npy')[:1000]
# dqn6_rewards = np.load('../baselines/env6-dqn-rewards.npy')[:1000]
# dqn7_rewards = np.load('../baselines/env7-sql0-rewards.npy')[:1000]
# dqn8_rewards = np.load('../baselines/env8-sql0-rewards.npy')[:1000]

# dqn4_smooth = pd.Series(dqn4_rewards).rolling(smoothing_window,min_periods=5).mean()
# dqn5_smooth = pd.Series(dqn5_rewards).rolling(smoothing_window,min_periods=5).mean()
# dqn6_smooth = pd.Series(dqn6_rewards).rolling(smoothing_window,min_periods=5).mean()
# dqn7_smooth = pd.Series(dqn7_rewards).rolling(smoothing_window,min_periods=5).mean()
# dqn8_smooth = pd.Series(dqn8_rewards).rolling(smoothing_window,min_periods=5).mean()

# print(distral8_rewards)


plt.figure(figsize=(10,5))
plt.title('Distral 2 col vs SQL on env 7, 8', fontsize='20')
plt.xlabel('Episodes ', fontsize='16')
plt.ylabel('Reward', fontsize='16')
# plt.ylabel('Duration', fontsize='16')

# plt.ylim(0,100)

plt.plot(distral4_smooth, label="Env 4 - Distral")
plt.plot(distral5_smooth, label="Env 5 - Distral")
plt.plot(distral6_smooth, label="Env 6 - Distral")
plt.plot(distral7_smooth, label="Env 7 - Distral")
plt.plot(distral8_smooth, label="Env 8 - Distral")

# plt.plot(dqn4_smooth, label="Env 4 - SQL")
# plt.plot(dqn5_smooth, label="Env 5 - SQL")
# plt.plot(dqn6_smooth, label="Env 5 - SQL")
# plt.plot(dqn7_smooth, label="Env 7 - SQL")
# plt.plot(dqn8_smooth, label="Env 8 - SQL")

plt.legend(loc='best', fontsize='20')
# plt.savefig('Benchmark-distral-vs-distral78-reward.eps', format='eps', dpi=1000)
plt.show()
# plt.close()
