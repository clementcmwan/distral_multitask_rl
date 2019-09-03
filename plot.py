import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# base_env4_rewards = np.load('sql0_n_step_kl/n_10/env4-sql0-rewards.npy')[:1000]
# base_env5_rewards = np.load('sql0_n_step_kl/n_10/env5-sql0-rewards.npy')[:1000]
# base_env6_rewards = np.load('sql0_n_step_kl/n_10/env6-sql0-rewards.npy')[:1000]
base_env7_rewards = np.load('a3c/a3c_res/env7-a3c-rewards.npy')[:200]
base_env8_rewards = np.load('a3c/a3c_res/env8-a3c-rewards.npy')[:200]

# base_env4_smooth = pd.Series(base_env4_rewards).rolling(smoothing_window,min_periods=2).mean()
# base_env5_smooth = pd.Series(base_env5_rewards).rolling(smoothing_window,min_periods=2).mean()
# base_env6_smooth = pd.Series(base_env6_rewards).rolling(smoothing_window,min_periods=2).mean()
base_env7_smooth = pd.Series(base_env7_rewards).rolling(smoothing_window,min_periods=2).mean()
base_env8_smooth = pd.Series(base_env8_rewards).rolling(smoothing_window,min_periods=2).mean()


# base_mean_smooth = (base_env4_smooth+base_env5_smooth+base_env6_smooth+base_env7_smooth+base_env8_smooth)/5.

# distral4_rewards = np.load('new_exp/dist_1_col_AC_2_betas/Distral_1col-distral0-rewards.npy')[:,0]
# distral5_rewards = np.load('new_exp/dist_1_col_AC_2_betas/Distral_1col-distral0-rewards.npy')[:,1]

distral4_rewards = np.load('dist_1_col_AC/Distral_1col-distral0-rewards.npy')[:,0]
distral5_rewards = np.load('dist_1_col_AC/Distral_1col-distral0-rewards.npy')[:,1]

print(distral4_rewards)

distral4_smooth = pd.Series(distral4_rewards).rolling(smoothing_window,min_periods=2).mean()
distral5_smooth = pd.Series(distral5_rewards).rolling(smoothing_window,min_periods=2).mean()

plt.figure(figsize=(10,5))
plt.title('Distral 1 col AC vs A3C on env 7, 8', fontsize='20')
plt.xlabel('Episodes ', fontsize='16')
plt.ylabel('Reward', fontsize='16')
# plt.ylabel('Duration', fontsize='16')


plt.plot(distral4_smooth, label="Env 7 - Distral")
plt.plot(distral5_smooth, label="Env 8 - Distral")
# plt.plot(distral6_smooth, label="Env 7 - Distral")
# plt.plot(distral7_smooth, label="Env 7 - Distral")
# plt.plot(distral8_smooth, label="Env 8 - Distral")


# plt.plot(base_env4_smooth, label="Env 4 - A3C")
# plt.plot(base_env5_smooth, label="Env 5 - A3C")
# plt.plot(base_env6_smooth, label="Env 6 - A3C")
plt.plot(base_env7_smooth, label="Env 7 - A3C")
plt.plot(base_env8_smooth, label="Env 8 - A3C")


plt.legend(loc='best', fontsize='12')
plt.show()
