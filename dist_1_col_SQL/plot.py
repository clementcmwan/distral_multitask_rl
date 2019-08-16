import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def clip(data, dur_bool):
	if dur_bool:
		data = pd.DataFrame(data)
		# nums = np.arange(30,50)
		# data[data == 1000] = int(np.random.choice(nums, 1))
		data[data > 100] = 100
		return data.to_numpy().squeeze()
	else:
		data = pd.DataFrame(data)
		data[data < -30] = -30
		return data.to_numpy().squeeze()


smoothing_window = 20

distral4_durations = np.load('Distral_1col-distral-2col-durations.npy')[0][:200]
distral5_durations = np.load('Distral_1col-distral-2col-durations.npy')[1][:200]
# distral6_durations = np.load('Distral_1col-distral-2col-durations.npy')[2][:200]
# distral7_durations = np.load('Distral_1col-distral-2col-durations.npy')[3][:200]
# distral8_durations = np.load('Distral_1col-distral-2col-durations.npy')[4][:200]

# distral4_durations = np.asarray(distral4_durations)
# distral4_durations[distral4_durations > 100] = 100

distral4_smooth = pd.Series(distral4_durations).rolling(smoothing_window,min_periods=5).mean()
distral5_smooth = pd.Series(distral5_durations).rolling(smoothing_window,min_periods=5).mean()
# distral6_smooth = pd.Series(distral6_durations).rolling(smoothing_window,min_periods=5).mean()
# distral7_smooth = pd.Series(distral7_durations).rolling(smoothing_window,min_periods=5).mean()
# distral8_smooth = pd.Series(distral8_durations).rolling(smoothing_window,min_periods=5).mean()

# distral_tot = (distral4_smooth+distral5_smooth+distral6_smooth+distral7_smooth+distral8_smooth)/5.

# dqn4_durations = np.load('../results/env4-sql0-durations.npy')[:200]
# dqn5_durations = np.load('../results/env5-sql0-durations.npy')[:200]
# dqn6_durations = np.load('../results/env6-sql0-durations.npy')[:200]
# dqn7_durations = np.load('../results/env7-sql0-durations.npy')[:200]
# dqn8_durations = np.load('../results/env8-sql0-durations.npy')[:200]

# dqn4_durations = np.concatenate((clip(dqn4_durations[:100], 1), dqn4_durations[100:]))
# dqn5_durations = np.concatenate((clip(dqn5_durations[:100], 1), dqn5_durations[100:]))
# dqn6_durations = np.concatenate((clip(dqn6_durations[:100], 1), dqn6_durations[100:]))
# dqn7_durations = np.concatenate((clip(dqn7_durations[:100], 1), dqn7_durations[100:]))
# dqn8_durations = np.concatenate((clip(dqn8_durations[:100], 1), dqn8_durations[100:]))

# dqn4_smooth = pd.Series(dqn4_durations).rolling(smoothing_window,min_periods=5).mean()
# dqn5_smooth = pd.Series(dqn5_durations).rolling(smoothing_window,min_periods=5).mean()
# dqn6_smooth = pd.Series(dqn6_durations).rolling(smoothing_window,min_periods=5).mean()
# dqn7_smooth = pd.Series(dqn7_durations).rolling(smoothing_window,min_periods=5).mean()
# dqn8_smooth = pd.Series(dqn8_durations).rolling(smoothing_window,min_periods=5).mean()

# print(distral8_durations)

# dqn_tot = (dqn4_smooth+dqn5_smooth+dqn6_smooth+dqn7_smooth+dqn8_smooth)/5.

plt.figure(figsize=(10,5))
plt.title('Distral 1 col vs SQL on env 4, 5, 7', fontsize='20')
plt.xlabel('Episodes ', fontsize='16')
# plt.ylabel('Reward', fontsize='16')
plt.ylabel('Duration', fontsize='16')

# plt.ylim(0,100)

# plt.plot(distral_tot, label="Average Reward - Distral")
# plt.plot(dqn_tot, label="Average Reward - SQL")

plt.plot(distral4_smooth, label="Env 4 - Distral")
plt.plot(distral5_smooth, label="Env 5 - Distral")
# plt.plot(distral6_smooth, label="Env 7 - Distral")
# plt.plot(distral7_smooth, label="Env 7 - Distral")
# plt.plot(distral8_smooth, label="Env 8 - Distral")

# plt.plot(dqn4_smooth, label="Env 4 - SQL")
# plt.plot(dqn5_smooth, label="Env 5 - SQL")
# plt.plot(dqn6_smooth, label="Env 5 - SQL")
# plt.plot(dqn7_smooth, label="Env 7 - SQL")
# plt.plot(dqn8_smooth, label="Env 8 - SQL")

plt.legend(loc='best', fontsize='16')
# plt.savefig('Benchmark-distral-vs-distral78-reward.eps', format='eps', dpi=1000)
plt.show()
# plt.close()
