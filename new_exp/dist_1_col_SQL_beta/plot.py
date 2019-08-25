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

distral4_rewards_b = np.load('res_beta_5/4_5_6_7_8/Distral_1col-distral-2col-rewards.npy')[0][:200]
distral5_rewards_b = np.load('res_beta_5/4_5_6_7_8/Distral_1col-distral-2col-rewards.npy')[1][:200]
distral6_rewards_b = np.load('res_beta_5/4_5_6_7_8/Distral_1col-distral-2col-rewards.npy')[2][:200]
distral7_rewards_b = np.load('res_beta_5/4_5_6_7_8/Distral_1col-distral-2col-rewards.npy')[3][:200]
distral8_rewards_b = np.load('res_beta_5/4_5_6_7_8/Distral_1col-distral-2col-rewards.npy')[4][:200]

# distral4_rewards = np.asarray(distral4_rewards)
# distral4_rewards[distral4_rewards > 100] = 100

distral4_smooth_b = pd.Series(distral4_rewards_b).rolling(smoothing_window,min_periods=5).mean()
distral5_smooth_b = pd.Series(distral5_rewards_b).rolling(smoothing_window,min_periods=5).mean()
distral6_smooth_b = pd.Series(distral6_rewards_b).rolling(smoothing_window,min_periods=5).mean()
distral7_smooth_b = pd.Series(distral7_rewards_b).rolling(smoothing_window,min_periods=5).mean()
distral8_smooth_b = pd.Series(distral8_rewards_b).rolling(smoothing_window,min_periods=5).mean()

distral_tot_b = (distral4_smooth_b+distral5_smooth_b+distral6_smooth_b+distral7_smooth_b+distral8_smooth_b)/5.



distral4_rewards = np.load('../../dist_2_col_SQL/res/4_5_6_7_8/Distral_1col-distral-2col-rewards.npy')[0][:200]
distral5_rewards = np.load('../../dist_2_col_SQL/res/4_5_6_7_8/Distral_1col-distral-2col-rewards.npy')[1][:200]
distral6_rewards = np.load('../../dist_2_col_SQL/res/4_5_6_7_8/Distral_1col-distral-2col-rewards.npy')[2][:200]
distral7_rewards = np.load('../../dist_2_col_SQL/res/4_5_6_7_8/Distral_1col-distral-2col-rewards.npy')[3][:200]
distral8_rewards = np.load('../../dist_2_col_SQL/res/4_5_6_7_8/Distral_1col-distral-2col-rewards.npy')[4][:200]

# distral4_rewards = np.asarray(distral4_rewards)
# distral4_rewards[distral4_rewards > 100] = 100

distral4_smooth = pd.Series(distral4_rewards).rolling(smoothing_window,min_periods=5).mean()
distral5_smooth = pd.Series(distral5_rewards).rolling(smoothing_window,min_periods=5).mean()
distral6_smooth = pd.Series(distral6_rewards).rolling(smoothing_window,min_periods=5).mean()
distral7_smooth = pd.Series(distral7_rewards).rolling(smoothing_window,min_periods=5).mean()
distral8_smooth = pd.Series(distral8_rewards).rolling(smoothing_window,min_periods=5).mean()

distral_tot = (distral4_smooth+distral5_smooth+distral6_smooth+distral7_smooth+distral8_smooth)/5.



# dqn4_rewards = np.load('../results/env4-sql0-rewards.npy')[:200]
# dqn5_rewards = np.load('../results/env5-sql0-rewards.npy')[:200]
# dqn6_rewards = np.load('../results/env6-sql0-rewards.npy')[:200]
# dqn7_rewards = np.load('../results/env7-sql0-rewards.npy')[:200]
# dqn8_rewards = np.load('../results/env8-sql0-rewards.npy')[:200]

# dqn4_rewards = np.concatenate((clip(dqn4_rewards[:100], 1), dqn4_rewards[100:]))
# dqn5_rewards = np.concatenate((clip(dqn5_rewards[:100], 1), dqn5_rewards[100:]))
# dqn6_rewards = np.concatenate((clip(dqn6_rewards[:100], 1), dqn6_rewards[100:]))
# dqn7_rewards = np.concatenate((clip(dqn7_rewards[:100], 1), dqn7_rewards[100:]))
# dqn8_rewards = np.concatenate((clip(dqn8_rewards[:100], 1), dqn8_rewards[100:]))

# dqn4_smooth = pd.Series(dqn4_rewards).rolling(smoothing_window,min_periods=5).mean()
# dqn5_smooth = pd.Series(dqn5_rewards).rolling(smoothing_window,min_periods=5).mean()
# dqn6_smooth = pd.Series(dqn6_rewards).rolling(smoothing_window,min_periods=5).mean()
# dqn7_smooth = pd.Series(dqn7_rewards).rolling(smoothing_window,min_periods=5).mean()
# dqn8_smooth = pd.Series(dqn8_rewards).rolling(smoothing_window,min_periods=5).mean()

# print(distral8_rewards)

# dqn_tot = (dqn4_smooth+dqn5_smooth+dqn6_smooth+dqn7_smooth+dqn8_smooth)/5.

plt.figure(figsize=(10,5))
plt.title('Distral 1 col SQL beta vs Distral 2 col SQL on env 4, 5, 6, 7, 8', fontsize='16')
plt.xlabel('Episodes ', fontsize='16')
plt.ylabel('Reward', fontsize='16')
# plt.ylabel('Duration', fontsize='16')

# plt.ylim(0,100)

plt.plot(distral_tot, label="Distral 2 col")
plt.plot(distral_tot_b, label="Distral 1 col beta")
# plt.plot(dqn_tot, label="Average Reward - SQL")

# plt.plot(distral4_smooth, label="Env 4 - Distral")
# plt.plot(distral5_smooth, label="Env 5 - Distral")
# plt.plot(distral6_smooth, label="Env 6 - Distral")
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
