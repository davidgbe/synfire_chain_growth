import numpy as np
import matplotlib.pyplot as plt

t_b = 2e-3
tau_triple = 100e-3
a_2plus = 1.
a_2minus = 1.
a_3minus = 0.

def make_vec(t_start, t_b):
	return np.arange(4) * t_b + t_start

def compute_stdp(v1, v2):
	s = 0

	for t_v1 in v1:
		for t_v2 in v2:

			if t_v1 == t_v2:
				pass
			elif t_v1 < t_v2:
				s += a_2plus * np.exp(-(t_v2 - t_v1)/ tau)
				for t_v1_prime in v1:
					if t_v1_prime > t_v2:
						s -= a_3minus * np.exp(-(t_v1_prime - t_v1)/ tau_triple) * np.exp(-(t_v1_prime - t_v2)/ tau)

			else:
				s -= a_2minus * np.exp(-(t_v1 - t_v2)/ tau)
				for t_v2_prime in v2:
					if t_v2_prime > t_v1:
						s += a_3plus * np.exp(-(t_v2_prime - t_v2)/ tau_triple) * np.exp(-(t_v2_prime - t_v1)/ tau)
	return s

v1 = make_vec(0, t_b)

fig, ax = plt.subplots(1, 1, figsize=(6, 6), tight_layout=True)

t_phases = np.linspace(0, 8 * t_b, 1000)

i = 0
for tau in [20e-3]:
	for a_3plus in [0.1, 0.5, 1., 2.]:
		all_s = []
		for t_phase in t_phases:
			v2 = make_vec(t_phase, t_b)
			s = compute_stdp(v1, v2)
			all_s.append(s)

		ax.plot(t_phases * 1000, np.array(all_s), label=f'A3+ = {(a_3plus)}')
		ax.legend()
		ax.set_xlabel('Difference in burst onset time (ms)')
		ax.set_ylabel('Relative STDP')
		i += 1
fig.savefig('stdp_triple_positive_lag_a3-_0.png')



	