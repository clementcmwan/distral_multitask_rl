import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


### ------------------------- dist 2 col normal --------------------------------------

distilled_mag_env4_2col = np.load('Distral_2col-distilled_logit_norms.npy')[:,0]
policy_mag_env4_2col = np.load('Distral_2col-policy_logit_norms.npy')[:,0]

distilled_mag_env5_2col = np.load('Distral_2col-distilled_logit_norms.npy')[:,1]
policy_mag_env5_2col = np.load('Distral_2col-policy_logit_norms.npy')[:,1]

distilled_mag_env6_2col = np.load('Distral_2col-distilled_logit_norms.npy')[:,2]
policy_mag_env6_2col = np.load('Distral_2col-policy_logit_norms.npy')[:,2]

distilled_mag_env7_2col = np.load('Distral_2col-distilled_logit_norms.npy')[:,3]
policy_mag_env7_2col = np.load('Distral_2col-policy_logit_norms.npy')[:,3]

distilled_mag_env8_2col = np.load('Distral_2col-distilled_logit_norms.npy')[:,4]
policy_mag_env8_2col = np.load('Distral_2col-policy_logit_norms.npy')[:,4]


dist_2_col_distilled_avg = (distilled_mag_env4_2col+distilled_mag_env5_2col+
	distilled_mag_env6_2col+distilled_mag_env7_2col+distilled_mag_env8_2col)/5.

dist_2_col_policy_avg = (policy_mag_env4_2col+policy_mag_env5_2col+
	policy_mag_env6_2col+policy_mag_env7_2col+policy_mag_env8_2col)/5.

### ------------------------- dist 1 col normal --------------------------------------


distilled_mag_env4_1col = np.load('../dist_1_col_SQL_avg_mag/Distral_1col-distilled_logit_norms.npy')[:,0]
policy_mag_env4_1col = np.load('../dist_1_col_SQL_avg_mag/Distral_1col-policy_logit_norms.npy')[:,0]

distilled_mag_env5_1col = np.load('../dist_1_col_SQL_avg_mag/Distral_1col-distilled_logit_norms.npy')[:,1]
policy_mag_env5_1col = np.load('../dist_1_col_SQL_avg_mag/Distral_1col-policy_logit_norms.npy')[:,1]

distilled_mag_env6_1col = np.load('../dist_1_col_SQL_avg_mag/Distral_1col-distilled_logit_norms.npy')[:,2]
policy_mag_env6_1col = np.load('../dist_1_col_SQL_avg_mag/Distral_1col-policy_logit_norms.npy')[:,2]

distilled_mag_env7_1col = np.load('../dist_1_col_SQL_avg_mag/Distral_1col-distilled_logit_norms.npy')[:,3]
policy_mag_env7_1col = np.load('../dist_1_col_SQL_avg_mag/Distral_1col-policy_logit_norms.npy')[:,3]

distilled_mag_env8_1col = np.load('../dist_1_col_SQL_avg_mag/Distral_1col-distilled_logit_norms.npy')[:,4]
policy_mag_env8_1col = np.load('../dist_1_col_SQL_avg_mag/Distral_1col-policy_logit_norms.npy')[:,4]


dist_1_col_distilled_avg = (distilled_mag_env4_1col+distilled_mag_env5_1col+
	distilled_mag_env6_1col+distilled_mag_env7_1col+distilled_mag_env8_1col)/5.

dist_1_col_policy_avg = (policy_mag_env4_1col+policy_mag_env5_1col+
	policy_mag_env6_1col+policy_mag_env7_1col+policy_mag_env8_1col)/5.

### ------------------------- dist 1 col beta --------------------------------------

distilled_mag_env4_1col_b = np.load('../dist_1_col_SQL_beta/Distral_1col-beta-distilled_logit_norms.npy')[:,0]
policy_mag_env4_1col_b = np.load('../dist_1_col_SQL_beta/Distral_1col-beta-policy_logit_norms.npy')[:,0]

distilled_mag_env5_1col_b = np.load('../dist_1_col_SQL_beta/Distral_1col-beta-distilled_logit_norms.npy')[:,1]
policy_mag_env5_1col_b = np.load('../dist_1_col_SQL_beta/Distral_1col-beta-policy_logit_norms.npy')[:,1]

distilled_mag_env6_1col_b = np.load('../dist_1_col_SQL_beta/Distral_1col-beta-distilled_logit_norms.npy')[:,2]
policy_mag_env6_1col_b = np.load('../dist_1_col_SQL_beta/Distral_1col-beta-policy_logit_norms.npy')[:,2]

distilled_mag_env7_1col_b = np.load('../dist_1_col_SQL_beta/Distral_1col-beta-distilled_logit_norms.npy')[:,3]
policy_mag_env7_1col_b = np.load('../dist_1_col_SQL_beta/Distral_1col-beta-policy_logit_norms.npy')[:,3]

distilled_mag_env8_1col_b = np.load('../dist_1_col_SQL_beta/Distral_1col-beta-distilled_logit_norms.npy')[:,4]
policy_mag_env8_1col_b = np.load('../dist_1_col_SQL_beta/Distral_1col-beta-policy_logit_norms.npy')[:,4]

dist_1_col_distilled_avg_b = (distilled_mag_env4_1col_b+distilled_mag_env5_1col_b+
	distilled_mag_env6_1col_b+distilled_mag_env7_1col_b+distilled_mag_env8_1col_b)/5.

dist_1_col_policy_avg_b = (policy_mag_env4_1col_b+policy_mag_env5_1col_b+
	policy_mag_env6_1col_b+policy_mag_env7_1col_b+policy_mag_env8_1col_b)/5.



plt.figure(figsize=(10,5))
plt.title('Magnitude of logits for Distral algorithms via SQL on env 4', fontsize='20')
plt.xlabel('Episodes ', fontsize='16')
plt.ylabel('Average Magnitude', fontsize='16')


# plt.plot(dist_1_col_distilled_avg, label="Distilled dist 1 col")
plt.plot(5*dist_1_col_policy_avg, label="Task-specific dist 1 col")

# plt.plot(dist_2_col_distilled_avg, label="Distilled dist 2 col")
# plt.plot(dist_2_col_policy_avg, label="Task-specific dist 2 col")

plt.plot(0.8*dist_2_col_distilled_avg+5*dist_2_col_policy_avg, label="dist 2 col")

# plt.plot(dist_1_col_distilled_avg_b, label="Distilled dist 1 col beta")
# plt.plot(dist_1_col_policy_avg_b, label="Task-specific dist 1 col beta")

plt.ylim(0,10)



# plt.plot(distilled_mag_env4_1col, label="Distilled dist 1 col")
# plt.plot(policy_mag_env4_1col, label="Task-specific dist 1 col")

# plt.plot(distilled_mag_env4_1col_b, label="Distilled dist 1 col beta")
# plt.plot(policy_mag_env4_1col_b, label="Task-specific dist 1 col beta")

# plt.plot(distilled_mag_env4_2col, label="Distilled dist 2 col")
# plt.plot(policy_mag_env4_2col, label="Task-specific dist 2 col")

# plt.plot(distilled_mag_env5_1col, label="Distilled dist 1 col")
# plt.plot(policy_mag_env5_1col, label="Task-specific dist 1 col")

# plt.plot(distilled_mag_env5_2col, label="Distilled dist 2 col")
# plt.plot(policy_mag_env5_2col, label="Task-specific dist 2 col")


plt.legend(loc='best', fontsize='16')
plt.show()
# plt.close()