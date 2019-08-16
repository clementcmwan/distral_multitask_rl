import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

smoothing_window = 20


a3c_rewards = np.load('env4-a3c-durations-run1.npy')

a3c_smooth = pd.Series(a3c_rewards).rolling(smoothing_window,min_periods=smoothing_window).mean()


plt.figure(figsize=(10,5))
plt.title('Benchmark Training A3C', fontsize='20')
plt.xlabel('Episodes ', fontsize='16')
plt.ylabel('Reward', fontsize='16')


plt.plot(a3c_smooth, label="Env 8")

plt.legend(loc='best', fontsize='20')
# plt.savefig('Benchmark-dqn-vs-distral78-reward.eps', format='eps', dpi=1000)
plt.show()
# plt.close()
