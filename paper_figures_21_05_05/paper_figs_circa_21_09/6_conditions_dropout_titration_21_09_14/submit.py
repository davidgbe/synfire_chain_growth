import os

base_path = os.curdir
scripts = [
	'submit_single_only.slurm',
	'submit_pop_single_no_silent.slurm',
	'submit_pop_single_silent.slurm',
]

def replace_all(line, repl_dict):
	s = line
	for k, v in repl_dict.items():
		s = s.replace(k, v)
	return s

for i, rng_seed in enumerate(range(2025, 2060)):
	for j, drop_sev in enumerate([0.3, 0.4, 0.5, 0.6]):
		for src_name in scripts:
			name_parts = src_name.split('.')
			dst_name = name_parts[0] + '_' + str(i) + '_' + str(j) + '.' + name_parts[1]

			src = open(src_name, 'rt')
			dst = open(dst_name, 'wt')

			for line in src:
				params = {
					'SEED': str(rng_seed),
					'DROP_SEV': str(drop_sev),
				}
				line_replaced = replace_all(line, params)
				dst.write(line_replaced)
			src.close()
			dst.close()

			os.system('sbatch ./' + dst_name)
			print(dst_name)
