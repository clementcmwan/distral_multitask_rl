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

distral4_rewards = np.load('res/pol_1_beta_5/Distral_1col-distral-2col-rewards.npy')[0][:200]
distral5_rewards = np.load('res/pol_1_beta_5/Distral_1col-distral-2col-rewards.npy')[1][:200]

distral4_smooth = pd.Series(distral4_rewards).rolling(smoothing_window,min_periods=5).mean()
distral5_smooth = pd.Series(distral5_rewards).rolling(smoothing_window,min_periods=5).mean()

#------------------------------------------------------------------------------------------------------

distral4_rewards_1 = np.load('res/pol_2_beta_5/Distral_1col-distral-2col-rewards.npy')[0][:200]
distral5_rewards_1 = np.load('res/pol_2_beta_5/Distral_1col-distral-2col-rewards.npy')[1][:200]

distral4_smooth_1 = pd.Series(distral4_rewards_1).rolling(smoothing_window,min_periods=5).mean()
distral5_smooth_1 = pd.Series(distral5_rewards_1).rolling(smoothing_window,min_periods=5).mean()

#------------------------------------------------------------------------------------------------------

distral4_rewards_2 = np.load('res/pol_3_beta_5/Distral_1col-distral-2col-rewards.npy')[0][:200]
distral5_rewards_2 = np.load('res/pol_3_beta_5/Distral_1col-distral-2col-rewards.npy')[1][:200]

distral4_smooth_2 = pd.Series(distral4_rewards_2).rolling(smoothing_window,min_periods=5).mean()
distral5_smooth_2 = pd.Series(distral5_rewards_2).rolling(smoothing_window,min_periods=5).mean()

#------------------------------------------------------------------------------------------------------

distral4_rewards_3 = np.load('res/pol_4_beta_5/Distral_1col-distral-2col-rewards.npy')[0][:200]
distral5_rewards_3 = np.load('res/pol_4_beta_5/Distral_1col-distral-2col-rewards.npy')[1][:200]

distral4_smooth_3 = pd.Series(distral4_rewards_3).rolling(smoothing_window,min_periods=5).mean()
distral5_smooth_3 = pd.Series(distral5_rewards_3).rolling(smoothing_window,min_periods=5).mean()

#------------------------------------------------------------------------------------------------------

distral4_rewards_4 = np.load('res/pol_5_beta_5/Distral_1col-distral-2col-rewards.npy')[0][:200]
distral5_rewards_4 = np.load('res/pol_5_beta_5/Distral_1col-distral-2col-rewards.npy')[1][:200]

distral4_smooth_4 = pd.Series(distral4_rewards_4).rolling(smoothing_window,min_periods=5).mean()
distral5_smooth_4 = pd.Series(distral5_rewards_4).rolling(smoothing_window,min_periods=5).mean()

#------------------------------------------------------------------------------------------------------

distral4_rewards_5 = np.load('res/pol_6_beta_5/Distral_1col-distral-2col-rewards.npy')[0][:200]
distral5_rewards_5 = np.load('res/pol_6_beta_5/Distral_1col-distral-2col-rewards.npy')[1][:200]

distral4_smooth_5 = pd.Series(distral4_rewards_5).rolling(smoothing_window,min_periods=5).mean()
distral5_smooth_5 = pd.Series(distral5_rewards_5).rolling(smoothing_window,min_periods=5).mean()

#------------------------------------------------------------------------------------------------------

distral4_rewards_6 = np.load('res/pol_7_beta_5/Distral_1col-distral-2col-rewards.npy')[0][:200]
distral5_rewards_6 = np.load('res/pol_7_beta_5/Distral_1col-distral-2col-rewards.npy')[1][:200]

distral4_smooth_6 = pd.Series(distral4_rewards_6).rolling(smoothing_window,min_periods=5).mean()
distral5_smooth_6 = pd.Series(distral5_rewards_6).rolling(smoothing_window,min_periods=5).mean()

#------------------------------------------------------------------------------------------------------

distral4_rewards_7 = np.load('res/pol_8_beta_5/Distral_1col-distral-2col-rewards.npy')[0][:200]
distral5_rewards_7 = np.load('res/pol_8_beta_5/Distral_1col-distral-2col-rewards.npy')[1][:200]

distral4_smooth_7 = pd.Series(distral4_rewards_7).rolling(smoothing_window,min_periods=5).mean()
distral5_smooth_7 = pd.Series(distral5_rewards_7).rolling(smoothing_window,min_periods=5).mean()

#------------------------------------------------------------------------------------------------------

distral4_rewards_8 = np.load('res/pol_9_beta_5/Distral_1col-distral-2col-rewards.npy')[0][:200]
distral5_rewards_8 = np.load('res/pol_9_beta_5/Distral_1col-distral-2col-rewards.npy')[1][:200]

distral4_smooth_8 = pd.Series(distral4_rewards_8).rolling(smoothing_window,min_periods=5).mean()
distral5_smooth_8 = pd.Series(distral5_rewards_8).rolling(smoothing_window,min_periods=5).mean()

#------------------------------------------------------------------------------------------------------

distral4_rewards_9 = np.load('res/pol_10_beta_5/Distral_1col-distral-2col-rewards.npy')[0][:200]
distral5_rewards_9 = np.load('res/pol_10_beta_5/Distral_1col-distral-2col-rewards.npy')[1][:200]

distral4_smooth_9 = pd.Series(distral4_rewards_9).rolling(smoothing_window,min_periods=5).mean()
distral5_smooth_9 = pd.Series(distral5_rewards_9).rolling(smoothing_window,min_periods=5).mean()

#------------------------------------------------------------------------------------------------------

distral4_rewards_10 = np.load('res/pol_50_beta_5/Distral_1col-distral-2col-rewards.npy')[0][:200]
distral5_rewards_10 = np.load('res/pol_50_beta_5/Distral_1col-distral-2col-rewards.npy')[1][:200]

distral4_smooth_10 = pd.Series(distral4_rewards_10).rolling(smoothing_window,min_periods=5).mean()
distral5_smooth_10 = pd.Series(distral5_rewards_10).rolling(smoothing_window,min_periods=5).mean()

#------------------------------------------------------------------------------------------------------

distral4_rewards_11 = np.load('res/pol_100_beta_5/Distral_1col-distral-2col-rewards.npy')[0][:200]
distral5_rewards_11 = np.load('res/pol_100_beta_5/Distral_1col-distral-2col-rewards.npy')[1][:200]

distral4_smooth_11 = pd.Series(distral4_rewards_11).rolling(smoothing_window,min_periods=5).mean()
distral5_smooth_11 = pd.Series(distral5_rewards_11).rolling(smoothing_window,min_periods=5).mean()



#------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------

distral4_rewards_new = np.load('res/pol_5_beta_2/Distral_1col-distral-2col-rewards.npy')[0][:200]
distral5_rewards_new = np.load('res/pol_5_beta_2/Distral_1col-distral-2col-rewards.npy')[1][:200]

distral4_smooth_new = pd.Series(distral4_rewards_new).rolling(smoothing_window,min_periods=5).mean()
distral5_smooth_new = pd.Series(distral5_rewards_new).rolling(smoothing_window,min_periods=5).mean()

#------------------------------------------------------------------------------------------------------

distral4_rewards_new_1 = np.load('res/pol_5_beta_3/Distral_1col-distral-2col-rewards.npy')[0][:200]
distral5_rewards_new_1 = np.load('res/pol_5_beta_3/Distral_1col-distral-2col-rewards.npy')[1][:200]

distral4_smooth_new_1 = pd.Series(distral4_rewards_new_1).rolling(smoothing_window,min_periods=5).mean()
distral5_smooth_new_1 = pd.Series(distral5_rewards_new_1).rolling(smoothing_window,min_periods=5).mean()

#------------------------------------------------------------------------------------------------------

distral4_rewards_new_2 = np.load('res/pol_5_beta_4/Distral_1col-distral-2col-rewards.npy')[0][:200]
distral5_rewards_new_2 = np.load('res/pol_5_beta_4/Distral_1col-distral-2col-rewards.npy')[1][:200]

distral4_smooth_new_2 = pd.Series(distral4_rewards_new_2).rolling(smoothing_window,min_periods=5).mean()
distral5_smooth_new_2 = pd.Series(distral5_rewards_new_2).rolling(smoothing_window,min_periods=5).mean()

#------------------------------------------------------------------------------------------------------

distral4_rewards_temp = np.load('Distral_1col-distral-2col-rewards.npy')[0][:200]
distral5_rewards_temp = np.load('Distral_1col-distral-2col-rewards.npy')[1][:200]

distral4_smooth_temp = pd.Series(distral4_rewards_temp).rolling(smoothing_window,min_periods=5).mean()
distral5_smooth_temp = pd.Series(distral5_rewards_temp).rolling(smoothing_window,min_periods=5).mean()


# dqn_tot = (dqn4_smooth+dqn5_smooth+dqn6_smooth+dqn7_smooth+dqn8_smooth)/5.

plt.figure(figsize=(10,5))
plt.title('Distral 1 col SQL with beta from 2 to 5 on envs 4 and 5', fontsize='16')
plt.xlabel('Episodes ', fontsize='16')
plt.ylabel('Average Reward', fontsize='16')
# plt.ylabel('Average Duration', fontsize='16')


plt.plot((distral4_smooth_new+distral5_smooth_new)/2., label="beta' = 5, beta = 2")
plt.plot((distral4_smooth_new_1+distral5_smooth_new_1)/2., label="beta' = 5, beta = 3")
plt.plot((distral4_smooth_new_2+distral5_smooth_new_2)/2., label="beta' = 5, beta = 4")

# plt.plot((distral4_smooth_temp+distral5_smooth_temp)/2., label="beta' = 5, beta = 4, unedited")



# plt.plot((distral4_smooth+distral5_smooth)/2., label="beta' = 1, beta = 5")

# plt.plot((distral4_smooth_1+distral5_smooth_1)/2., label="beta' = 2, beta = 5")

# plt.plot((distral4_smooth_2+distral5_smooth_2)/2., label="beta' = 3, beta = 5")

# plt.plot((distral4_smooth_3+distral5_smooth_2)/2., label="beta' = 4, beta = 5")

plt.plot((distral4_smooth_4+distral5_smooth_4)/2., label="beta' = 5, beta = 5")

# plt.plot((distral4_smooth_5+distral5_smooth_5)/2., label="beta' = 6, beta = 5")

# plt.plot((distral4_smooth_6+distral5_smooth_6)/2., label="beta' = 7, beta = 5")

# plt.plot((distral4_smooth_7+distral5_smooth_7)/2., label="beta' = 8, beta = 5")

# plt.plot((distral4_smooth_8+distral5_smooth_8)/2., label="beta' = 9, beta = 5")

# plt.plot((distral4_smooth_9+distral5_smooth_9)/2., label="beta' = 10, beta = 5")

# plt.plot((distral4_smooth_10+distral5_smooth_10)/2., label="beta' = 50, beta = 5")

# plt.plot((distral4_smooth_11+distral5_smooth_11)/2., label="beta' = 100, beta = 5")

# plt.plot(distral6_smooth, label="Env 6 - Distral")
# plt.plot(distral7_smooth, label="Env 7 - Distral")
# plt.plot(distral8_smooth, label="Env 8 - Distral")


plt.legend(loc='best', fontsize='16')
# plt.savefig('Benchmark-distral-vs-distral78-reward.eps', format='eps', dpi=1000)
plt.show()
# plt.close()
