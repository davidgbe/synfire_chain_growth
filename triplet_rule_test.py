import numpy as np
import time
from scipy.sparse import csc_matrix, csr_matrix, kron
import sparse

###

# sparsity in simulation for timestep 0.05e-3
# is 2.5 Hz (driving rate) * ~4 spks per neuron * t_step_size = 2.5 * 4 * 0.05e-3 = ~0.0005

spk_dim = 50
t_step_size = 0.05e-3
stdp_time_dim = int(35e-3 / t_step_size)
t_steps = int(1000/t_step_size / 1e3 * 5)
t_lim = int(100e-3 / t_step_size)
print(f't_steps = {t_steps}')
print(f't_lim = {t_lim}')

np.random.seed(1)

def rand_spks_of_dim(*dim, spk_density_thresh):
	return np.where(np.random.rand(*dim) < spk_density_thresh, 1, 0)

### speed test non-sparse

for spk_density_thresh in [0.0005, 0.001, 0.005, 0.01]:
	print('sparsity:', spk_density_thresh)


	curr_spks_all = [rand_spks_of_dim(1, spk_dim, spk_density_thresh=spk_density_thresh) for t in range(t_steps)]
	spk_hist_all = [rand_spks_of_dim(stdp_time_dim, spk_dim, spk_density_thresh=spk_density_thresh) for t in range(t_steps)]

	### sparse

	spk_time_hist = []

	s1 = time.time()

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

		a = csc_matrix(curr_spks)
		b = csc_matrix(spk_hist)
		o = kron(a, b)
		# print(o.shape)
		o.multiply(np.ones((stdp_time_dim, 1)))

		o_prime = o.todense()
		# print(o_prime.shape)

		for spk_time, nrn_idx in spk_time_hist:
			o_prime[:(t_ctr - spk_time), nrn_idx * spk_dim:(nrn_idx+1) * spk_dim]

	print(spk_time_hist)
	e1 = time.time()
	print(e1 - s1)