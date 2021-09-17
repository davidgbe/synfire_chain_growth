import os
import numpy as np

base_path = os.curdir
scripts = [
	'submit_single_only.slurm',
]
drop_sev = 0.5

def replace_all(line, repl_dict):
	s = line
	for k, v in repl_dict.items():
		s = s.replace(k, v)
	return s

def format_title(params):
	title = ''
	for k, v in params.items():
		title += ('_' + k + '_' + v)
	return title

for i, rng_seed in enumerate(range(2025, 2060)):
	for src_name in scripts:
		for j, alpha_1 in enumerate(np.linspace(1e-2, 5e-2, 5)):
			for k, alpha_2 in enumerate(np.linspace(1e-2, 5e-2, 5)):
				name_parts = src_name.split('.')
				dst_name = name_parts[0] + '_' + str(i) + '_' + str(j) + '_' + str(k) + '.' + name_parts[1]

				src = open(src_name, 'rt')
				dst = open(dst_name, 'wt')

				for line in src:
					params = {
						'SEED': str(rng_seed),
						'DROP_SEV': str(drop_sev),
						'ALPHA_1': str(alpha_1),
						'ALPHA_2': str(alpha_2),
					}
					params['TITLE'] = format_title(params)
					line_replaced = replace_all(line, params)
					dst.write(line_replaced)
				src.close()
				dst.close()

				os.system('sbatch ./' + dst_name)
				print(dst_name)
