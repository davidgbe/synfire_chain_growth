import os

base_path = os.curdir
scripts = [
	'submit_single_only.slurm',
	'submit_pop_single_no_silent.slurm',
	'submit_pop_single_silent.slurm',
]
drop_sev = 0.5

def replace_all(line, repl_dict):
	s = line
	for k, v in repl_dict.items():
		s = s.replace(k, v)
	return s

for i, rng_seed in enumerate(range(2025, 2060)):
	for src_name in scripts:
		name_parts = src_name.split('.')
		dst_name = name_parts[0] + '_' + str(i) + '.' + name_parts[1]

		src = open(src_name, 'rt')
		dst = open(dst_name, 'wt')

		for line in src:
			params = {
				'SEED': str(rng_seed),
				'DROP_SEV': str(drop_sev),
				'ALPHA_1': str(alpha_1),
				'ALPHA_2': str(alpha_2),
				'BETA': str(beta),
				'GAMMA': str(gamma),
			}
			params['TITLE'] = str(params)
			line_replaced = replace_all(line, params)
			dist.write(line_replaced)
		src.close()
		dst.close()

		os.system('sbatch ./' + dst_name)
		print(dst_name)
