import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def clip(data, dur_bool):
	if dur_bool:
		data = pd.DataFrame(data)
		# nums = np.arange(30,50)
		# data[data == 1000] = int(np.random.choice(nums, 1))
		data[data > 500] = 30
		return data.to_numpy().squeeze()
	else:
		data = pd.DataFrame(data)
		data[data < -500] = -5
		return data.to_numpy().squeeze()

smoothing_window = 20

distral4_rewards = np.load('dist_2_col_SQL/res/n_10/Distral_1col-distral-2col-rewards.npy')[0][:1000]
distral5_rewards = np.load('dist_2_col_SQL/res/n_10/Distral_1col-distral-2col-rewards.npy')[1][:1000]
distral6_rewards = np.load('dist_2_col_SQL/res/n_10/Distral_1col-distral-2col-rewards.npy')[2][:1000]
distral7_rewards = np.load('dist_2_col_SQL/res/n_10/Distral_1col-distral-2col-rewards.npy')[3][:1000]
distral8_rewards = np.load('dist_2_col_SQL/res/n_10/Distral_1col-distral-2col-rewards.npy')[4][:1000]

distral4_smooth = pd.Series(distral4_rewards).rolling(smoothing_window,min_periods=2).mean()
distral5_smooth = pd.Series(distral5_rewards).rolling(smoothing_window,min_periods=2).mean()
distral6_smooth = pd.Series(distral6_rewards).rolling(smoothing_window,min_periods=2).mean()
distral7_smooth = pd.Series(distral7_rewards).rolling(smoothing_window,min_periods=2).mean()
distral8_smooth = pd.Series(distral8_rewards).rolling(smoothing_window,min_periods=2).mean()


dist_mean_smooth = (distral4_smooth+distral5_smooth+distral6_smooth+distral7_smooth+distral8_smooth)/5.


base_env4_rewards = np.load('sql0_n_step_kl/n_10/env4-sql0-rewards.npy')[:1000]
base_env5_rewards = np.load('sql0_n_step_kl/n_10/env5-sql0-rewards.npy')[:1000]
base_env6_rewards = np.load('sql0_n_step_kl/n_10/env6-sql0-rewards.npy')[:1000]
base_env7_rewards = np.load('sql0_n_step_kl/n_10/env7-sql0-rewards.npy')[:1000]
base_env8_rewards = np.load('sql0_n_step_kl/n_10/env8-sql0-rewards.npy')[:1000]

base_env4_smooth = pd.Series(base_env4_rewards).rolling(smoothing_window,min_periods=2).mean()
base_env5_smooth = pd.Series(base_env5_rewards).rolling(smoothing_window,min_periods=2).mean()
base_env6_smooth = pd.Series(base_env6_rewards).rolling(smoothing_window,min_periods=2).mean()
base_env7_smooth = pd.Series(base_env7_rewards).rolling(smoothing_window,min_periods=2).mean()
base_env8_smooth = pd.Series(base_env8_rewards).rolling(smoothing_window,min_periods=2).mean()


base_mean_smooth = (base_env4_smooth+base_env5_smooth+base_env6_smooth+base_env7_smooth+base_env8_smooth)/5.


plt.figure(figsize=(10,5))
plt.title('Distral 2 col AC vs A3C on env 4, 5', fontsize='20')
plt.xlabel('Episodes ', fontsize='16')
plt.ylabel('Reward', fontsize='16')
# plt.ylabel('Duration', fontsize='16')

# plt.ylim(0,100)

plt.plot(dist_mean_smooth, label="Distral")
plt.plot(base_mean_smooth, label="SQL")

# plt.plot(distral4_smooth, label="Env 7 - Distral")
# plt.plot(distral5_smooth, label="Env 8 - Distral")
# plt.plot(distral6_smooth, label="Env 7 - Distral")
# plt.plot(distral7_smooth, label="Env 7 - Distral")
# plt.plot(distral8_smooth, label="Env 8 - Distral")


# plt.plot(base_env4_smooth, label="Env 4 - A3C")
# plt.plot(base_env5_smooth, label="Env 5 - A3C")
# plt.plot(base_env6_smooth, label="Env 6 - A3C")
# plt.plot(base_env7_smooth, label="Env 7 - A3C")
# plt.plot(base_env8_smooth, label="Env 8 - A3C")


plt.legend(loc='best', fontsize='12')
# plt.savefig('Benchmark-distral-vs-distral78-reward.eps', format='eps', dpi=1000)
plt.show()
# plt.close()
