import numpy as np
import time
from scipy.sparse import csc_matrix, csr_matrix, kron
import sparse

spk_dim = 50
stdp_time_dim = int(20e-3 / 0.0199999e-3)
n_trials = 300

np.random.seed(1)

def rand_spks_of_dim(*dim, spk_density_thresh):
	return np.where(np.random.rand(*dim) < spk_density_thresh, 1, 0)

### speed test non-sparse

for spk_density_thresh in [0.01, 0.05, 0.1]:
	print('sparsity:', spk_density_thresh)

	curr_spks = rand_spks_of_dim(1, spk_dim, spk_density_thresh=spk_density_thresh)
	spk_hist = rand_spks_of_dim(stdp_time_dim, spk_dim, spk_density_thresh=spk_density_thresh)

	s1 = time.time()

	for i in range(n_trials):
		np.outer(curr_spks, spk_hist)

	e1 = time.time()
	print(e1 - s1)

	s2 = time.time()

	for i in range(n_trials):
		np.tensordot(curr_spks, spk_hist, axes=0)

	e2 = time.time()
	print(e2 - s2)

	### sparse

	s3 = time.time()

	indices = np.arange(0, spk_dim**2).reshape(spk_dim, spk_dim).T.reshape(spk_dim**2)
	# print(indices)

	cache = []

	# for i in range(100):
	for i in range(n_trials):
		a = csr_matrix(curr_spks)
		b = csr_matrix(spk_hist)
		o = kron(a, b)

	e3 = time.time()
	print(e3 - s3)

	# s4 = time.time()

	# for i in range(n_trials):
	# 	a = sparse.COO(curr_spks)
	# 	b = sparse.COO(spk_hist)
	# 	o = kron(a, b)

	# e4 = time.time()
	# print(e4 - s4)