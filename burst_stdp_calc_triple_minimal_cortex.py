import numpy as np
import matplotlib.pyplot as plt

t_b = 2e-3

tau_plus = 16.8e-3
tau_minus = 33.7e-3
tau_triple = 114e-3

a_2plus = 0
a_2minus = 7.1

a_3plus = 6.5
a_3minus = 0.

num_spikes = [2, 4, 6, 8, 10]

def make_vec(t_start, t_b, num_spikes):
	return np.arange(num_spikes) * t_b + t_start

def compute_stdp(v1, v2):
	s_a2_minus = 0
	s_a3_plus = 0

	for t_v1 in v1:
		for t_v2 in v2:

			# if t_v1 == t_v2:
			# 	pass
			# elif t_v1 < t_v2:
			# 	s += a_2plus * np.exp(-(t_v2 - t_v1)/ tau)
			# 	for t_v1_prime in v1:
			# 		if t_v1_prime > t_v2:
			# 			s -= a_3minus * np.exp(-(t_v1_prime - t_v1)/ tau_triple) * np.exp(-(t_v1_prime - t_v2)/ tau)

			if t_v1 > t_v2:
				s_a2_minus -= a_2minus * np.exp(-(t_v1 - t_v2)/ tau_minus)
				for t_v2_prime in v2:
					if t_v2_prime > t_v1:
						s_a3_plus += a_3plus * np.exp(-(t_v2_prime - t_v2)/ tau_triple) * np.exp(-(t_v2_prime - t_v1)/ tau_plus)
	return s_a2_minus, s_a3_plus

fig, ax = plt.subplots(1, 1, figsize=(6, 6), tight_layout=True)

for i_s in num_spikes:

	v1 = make_vec(0, t_b, i_s)

	t_phases = np.linspace(-2 * t_b, 8 * t_b, 1000)

	std = 0.3e-3
	samples = 10

	noises_1 = [np.random.normal(0, std, i_s) for x in range(samples)]
	noises_2 = [np.random.normal(0, std, i_s) for x in range(samples)]

	all_s = np.zeros((samples, t_phases.shape[0]))
	all_s_a2_minus = np.zeros((samples, t_phases.shape[0]))
	all_s_a3_plus = np.zeros((samples, t_phases.shape[0]))

	for i, (noise_i, noise_j) in enumerate(zip(noises_1, noises_2)):

		noise_i[0] = 0
		noise_j[0] = 0

		for i_p, t_phase in enumerate(t_phases):
			v2 = make_vec(t_phase, t_b, i_s)
			s_a2_minus, s_a3_plus = compute_stdp(v1 + noise_i, v2 + noise_j)

			all_s[i, i_p] = s_a2_minus + s_a3_plus
			all_s_a2_minus[i, i_p] = s_a2_minus
			all_s_a3_plus[i, i_p] = s_a3_plus

		# if i == 0:
		# 	ax.plot(t_phases * 1000, np.array(all_s), label=f'all', c='black', lw=0.5)
		# 	ax.plot(t_phases * 1000, np.array(all_s_a2_minus), label=f'A2-', c='blue', lw=0.5)
		# 	ax.plot(t_phases * 1000, np.array(all_s_a3_plus), label=f'A3+', c='red', lw=0.5)
		# else:
		# 	ax.plot(t_phases * 1000, np.array(all_s), c='black', lw=0.5)
		# 	ax.plot(t_phases * 1000, np.array(all_s_a2_minus), c='blue', lw=0.5)
		# 	ax.plot(t_phases * 1000, np.array(all_s_a3_plus), c='red', lw=0.5)

	ax.plot(t_phases * 1000, np.mean(all_s, axis=0), label=f'all', lw=0.5)
# ax.plot(t_phases * 1000, np.mean(all_s_a2_minus, axis=0), label=f'A2-', c='blue', lw=0.5)
# ax.plot(t_phases * 1000, np.mean(all_s_a3_plus, axis=0), label=f'A3+', c='red', lw=0.5)

ax.legend()
ax.set_xlabel('Difference in burst onset time (ms)')
ax.set_ylabel('Relative STDP')

fig.savefig('stdp_triple_pfister_cortex_pos_lag_num_spikes.png')



	