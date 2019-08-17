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

distral4_durations = np.load('dist_2_col_SQL/res/4_5_6_7_8/Distral_1col-distral-2col-durations.npy')[0][:200]
distral5_durations = np.load('dist_2_col_SQL/res/4_5_6_7_8/Distral_1col-distral-2col-durations.npy')[1][:200]
distral6_durations = np.load('dist_2_col_SQL/res/4_5_6_7_8/Distral_1col-distral-2col-durations.npy')[2][:200]
distral7_durations = np.load('dist_2_col_SQL/res/4_5_6_7_8/Distral_1col-distral-2col-durations.npy')[3][:200]
distral8_durations = np.load('dist_2_col_SQL/res/4_5_6_7_8/Distral_1col-distral-2col-durations.npy')[4][:200]

distral4_smooth = pd.Series(distral4_durations).rolling(smoothing_window,min_periods=2).mean()
distral5_smooth = pd.Series(distral5_durations).rolling(smoothing_window,min_periods=2).mean()
distral6_smooth = pd.Series(distral6_durations).rolling(smoothing_window,min_periods=2).mean()
distral7_smooth = pd.Series(distral7_durations).rolling(smoothing_window,min_periods=2).mean()
distral8_smooth = pd.Series(distral8_durations).rolling(smoothing_window,min_periods=2).mean()


dist_2_SQL_mean_smooth = (distral4_smooth+distral5_smooth+distral6_smooth+distral7_smooth+distral8_smooth)/5.



dist_1_col_SQL4_durations = np.load('dist_1_col_SQL/res/4_5_6_7_8/Distral_1col-distral-2col-durations.npy')[0][:200]
dist_1_col_SQL5_durations = np.load('dist_1_col_SQL/res/4_5_6_7_8/Distral_1col-distral-2col-durations.npy')[1][:200]
dist_1_col_SQL6_durations = np.load('dist_1_col_SQL/res/4_5_6_7_8/Distral_1col-distral-2col-durations.npy')[2][:200]
dist_1_col_SQL7_durations = np.load('dist_1_col_SQL/res/4_5_6_7_8/Distral_1col-distral-2col-durations.npy')[3][:200]
dist_1_col_SQL8_durations = np.load('dist_1_col_SQL/res/4_5_6_7_8/Distral_1col-distral-2col-durations.npy')[4][:200]

dist_1_col_SQL4_smooth = pd.Series(dist_1_col_SQL4_durations).rolling(smoothing_window,min_periods=2).mean()
dist_1_col_SQL5_smooth = pd.Series(dist_1_col_SQL5_durations).rolling(smoothing_window,min_periods=2).mean()
dist_1_col_SQL6_smooth = pd.Series(dist_1_col_SQL6_durations).rolling(smoothing_window,min_periods=2).mean()
dist_1_col_SQL7_smooth = pd.Series(dist_1_col_SQL7_durations).rolling(smoothing_window,min_periods=2).mean()
dist_1_col_SQL8_smooth = pd.Series(dist_1_col_SQL8_durations).rolling(smoothing_window,min_periods=2).mean()


dist_1_SQL_mean_smooth = (dist_1_col_SQL4_smooth+dist_1_col_SQL5_smooth+dist_1_col_SQL6_smooth+dist_1_col_SQL7_smooth+dist_1_col_SQL8_smooth)/5.



dist_1_col_AC4_durations = np.load('dist_1_col_AC/res/4_5_6_7_8/Distral_1col-distral0-durations.npy')[:,0]
dist_1_col_AC5_durations = np.load('dist_1_col_AC/res/4_5_6_7_8/Distral_1col-distral0-durations.npy')[:,1]
dist_1_col_AC6_durations = np.load('dist_1_col_AC/res/4_5_6_7_8/Distral_1col-distral0-durations.npy')[:,2]
dist_1_col_AC7_durations = np.load('dist_1_col_AC/res/4_5_6_7_8/Distral_1col-distral0-durations.npy')[:,3]
dist_1_col_AC8_durations = np.load('dist_1_col_AC/res/4_5_6_7_8/Distral_1col-distral0-durations.npy')[:,4]

dist_1_col_AC4_smooth = pd.Series(dist_1_col_AC4_durations).rolling(smoothing_window,min_periods=2).mean()
dist_1_col_AC5_smooth = pd.Series(dist_1_col_AC5_durations).rolling(smoothing_window,min_periods=2).mean()
dist_1_col_AC6_smooth = pd.Series(dist_1_col_AC6_durations).rolling(smoothing_window,min_periods=2).mean()
dist_1_col_AC7_smooth = pd.Series(dist_1_col_AC7_durations).rolling(smoothing_window,min_periods=2).mean()
dist_1_col_AC8_smooth = pd.Series(dist_1_col_AC8_durations).rolling(smoothing_window,min_periods=2).mean()


dist_1_AC_mean_smooth = (dist_1_col_AC4_smooth+dist_1_col_AC5_smooth+dist_1_col_AC6_smooth+dist_1_col_AC7_smooth+dist_1_col_AC8_smooth)/5.



dist_2_col_AC4_durations = np.load('dist_2_col_AC/res/4_5_6_7_8/Distral_1col-distral0-durations.npy')[:,0]
dist_2_col_AC5_durations = np.load('dist_2_col_AC/res/4_5_6_7_8/Distral_1col-distral0-durations.npy')[:,1]
dist_2_col_AC6_durations = np.load('dist_2_col_AC/res/4_5_6_7_8/Distral_1col-distral0-durations.npy')[:,2]
dist_2_col_AC7_durations = np.load('dist_2_col_AC/res/4_5_6_7_8/Distral_1col-distral0-durations.npy')[:,3]
dist_2_col_AC8_durations = np.load('dist_2_col_AC/res/4_5_6_7_8/Distral_1col-distral0-durations.npy')[:,4]

dist_2_col_AC4_smooth = pd.Series(dist_2_col_AC4_durations).rolling(smoothing_window,min_periods=2).mean()
dist_2_col_AC5_smooth = pd.Series(dist_2_col_AC5_durations).rolling(smoothing_window,min_periods=2).mean()
dist_2_col_AC6_smooth = pd.Series(dist_2_col_AC6_durations).rolling(smoothing_window,min_periods=2).mean()
dist_2_col_AC7_smooth = pd.Series(dist_2_col_AC7_durations).rolling(smoothing_window,min_periods=2).mean()
dist_2_col_AC8_smooth = pd.Series(dist_2_col_AC8_durations).rolling(smoothing_window,min_periods=2).mean()


dist_2_AC_mean_smooth = (dist_2_col_AC4_smooth+dist_2_col_AC5_smooth+dist_2_col_AC6_smooth+dist_2_col_AC7_smooth+dist_2_col_AC8_smooth)/5.



# base_env4_durations = np.load('sql0_n_step_kl/4_5_6_7_8/env4-sql0-durations.npy')[:200]
# base_env5_durations = np.load('sql0_n_step_kl/4_5_6_7_8/env5-sql0-durations.npy')[:200]
# base_env6_durations = np.load('sql0_n_step_kl/4_5_6_7_8/env6-sql0-durations.npy')[:200]
# base_env7_durations = np.load('sql0_n_step_kl/4_5_6_7_8/env7-sql0-durations.npy')[:200]
# base_env8_durations = np.load('sql0_n_step_kl/4_5_6_7_8/env8-sql0-durations.npy')[:200]

# base_env4_smooth = pd.Series(base_env4_durations).rolling(smoothing_window,min_periods=2).mean()
# base_env5_smooth = pd.Series(base_env5_durations).rolling(smoothing_window,min_periods=2).mean()
# base_env6_smooth = pd.Series(base_env6_durations).rolling(smoothing_window,min_periods=2).mean()
# base_env7_smooth = pd.Series(base_env7_durations).rolling(smoothing_window,min_periods=2).mean()
# base_env8_smooth = pd.Series(base_env8_durations).rolling(smoothing_window,min_periods=2).mean()


# base_mean_smooth = (base_env4_smooth+base_env5_smooth+base_env6_smooth+base_env7_smooth+base_env8_smooth)/5.


plt.figure(figsize=(10,5))
plt.title('All Distral Algorithms and Architectures', fontsize='20')
plt.xlabel('Episodes ', fontsize='16')
# plt.ylabel('Average Reward', fontsize='16')
plt.ylabel('Average Duration', fontsize='16')

# plt.ylim(0,100)


plt.plot(dist_1_SQL_mean_smooth, label="Dist 1 col SQL")
plt.plot(dist_2_SQL_mean_smooth, label="Dist 2 col SQL")
plt.plot(dist_1_AC_mean_smooth, label="Dist 1 col AC")
plt.plot(dist_2_AC_mean_smooth, label="Dist 2 col AC")



# plt.plot(dist_mean_smooth, label="Distral")
# plt.plot(base_mean_smooth, label="SQL")

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
