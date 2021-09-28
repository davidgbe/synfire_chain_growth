import os

base_path = os.curdir
scripts = [
	'submit_pop_single_no_silent.slurm',
]
drop_sev = 0.5

def replace_all(line, repl_dict):
	s = line
	for k, v in repl_dict.items():
		if k == 'TITLE': 
			continue
		s = s.replace(k, v)

	if 'TITLE' in repl_dict:
		s = s.replace('TITLE', repl_dict['TITLE'])

	return s

def format_title(params):
	title = ''
	for k, v in params.items():
		title += ('_' + k + '_' + v)
	return title

def iter_range(r, n):
	if n == 1:
		yield (0, r[0])
	else:
		for i in range(n):
			yield (i, i * (r[1] - r[0]) / (n - 1) + r[0])


alpha_1_range = (1e-2, 5e-2)
alpha_2_range = (1e-2, 5e-2)
gamma_range = (0, 2e-4)

for i, rng_seed in enumerate(range(2025, 2055)):
	for src_name in scripts:
		for j, alpha_1 in iter_range(alpha_1_range, 5):
			for k, alpha_2 in iter_range(alpha_2_range, 5):
				for l, gamma in iter_range(gamma_range, 5):
					name_parts = src_name.split('.')
					dst_name = name_parts[0] + '_' + str(i) + '_' + str(j) + '_' + str(k) + '_' + str(l) + '.' + name_parts[1]

					src = open(src_name, 'rt')
					dst = open(dst_name, 'wt')

					for line in src:
						params = {
							'SEED': str(rng_seed),
							'DROP_SEV': str(drop_sev),
							'ALPHA_1': str(alpha_1),
							'ALPHA_2': str(alpha_2),
							'GAMMA': str(gamma),
						}
						params['TITLE'] = format_title(params)
						line_replaced = replace_all(line, params)
						dst.write(line_replaced)
					src.close()
					dst.close()

					# os.system('sbatch ./' + dst_name)
					print(dst_name)
