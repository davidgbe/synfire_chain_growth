import numpy as np
import time
from scipy.sparse import csc_matrix, csr_matrix, kron
import sparse

spk_dim = 50
t_step_size = 0.05e-3
stdp_time_dim = int(20e-3 / t_step_size)
t_steps = int(1000/t_step_size / 1e4)
t_lim = int(100e-3 / t_step_size)
print(f't_steps = {t_steps}')

np.random.seed(1)

def rand_spks_of_dim(*dim, spk_density_thresh):
	return np.where(np.random.rand(*dim) < spk_density_thresh, 1, 0)

### speed test non-sparse

for spk_density_thresh in [0.001, 0.005, 0.01]:
	print('sparsity:', spk_density_thresh)


	curr_spks_all = [rand_spks_of_dim(1, spk_dim, spk_density_thresh=spk_density_thresh) for t in range(t_steps)]
	spk_hist_all = [rand_spks_of_dim(stdp_time_dim, spk_dim, spk_density_thresh=spk_density_thresh) for t in range(t_steps)]

	### sparse

	spk_time_hist = []

	s1 = time.time()

	# indices = np.arange(0, spk_dim**2).reshape(spk_dim, spk_dim).T.reshape(spk_dim**2)

	# for i in range(100):
	for t_ctr in range(t_steps):
		curr_spks = curr_spks_all[t_ctr]
		spk_hist = spk_hist_all[t_ctr]

		#print(curr_spks)
		for idx in curr_spks.nonzero()[1]:
			spk_time_hist.append((t_ctr, idx))

		# print(spk_time_hist)
		for i in range(len(spk_time_hist)):
			if t_ctr - spk_time_hist[i][0] >= t_lim:
				spk_time_hist.pop(0)
			else:
				break


		a = csr_matrix(curr_spks)
		b = csr_matrix(spk_hist)
		o = kron(a, b)
		o_prime = o.todense()

		for spk_time, nrn_idx in spk_time_hist:
			o_prime[nrn_idx * spk_dim:(nrn_idx+1) * spk_dim, :(t_ctr - spk_time)]

	print(spk_time_hist)
	e1 = time.time()
	print(e1 - s1)