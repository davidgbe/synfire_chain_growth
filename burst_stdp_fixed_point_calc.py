import numpy as np
import matplotlib.pyplot as plt

t_b = 4e-3

def make_vec(t_start, t_b):
	return np.arange(4) * t_b + t_start

def compute_stdp(v1, v2):
	s = 0

	for t_v1 in v1:
		for t_v2 in v2:

			if t_v1 == t_v2:
				pass
			elif t_v1 < t_v2:
				s += np.exp(-(t_v2 - t_v1)/ tau)
			else:
				s -= np.exp(-(t_v1 - t_v2)/ tau)
				pass
	return s

v1 = make_vec(0, t_b)

fig, ax = plt.subplots(1, 1, figsize=(6, 6), tight_layout=True)

t_phases = np.linspace(0, 8 * t_b, 1000)

for tau in [20e-3]:
	all_s = []
	for t_phase in t_phases:
		v2 = make_vec(t_phase, t_b)
		s = compute_stdp(v1, v2)
		all_s.append(s)

	ax.plot(t_phases * 1000, all_s, label=f'STDP tau = {int(tau * 1000)} ms')
	ax.set_xlabel('Difference in burst onset time (ms)')
	ax.set_ylabel('Relative STDP')
ax.plot(t_phases * 1000, [8.25] * t_phases.shape[0], label=r'$\gamma (f_{i} - f_{i}^{0})$')
ax.legend()

fig.savefig('weight_evo_fixed_point_calc_long_isi.png')



	