import os

base_path = os.curdir
scripts = [
	'submit_single_only.slurm',
	'submit_pop_single_no_silent.slurm',
	'submit_pop_single_silent.slurm',
]

for i, rng_seed in enumerate(range(2025, 2030)):
	for j, drop_sev in enumerate([0.3, 0.4, 0.5, 0.6]):
		for src_name in scripts:
			name_parts = src_name.split('.')
			dst_name = name_parts[0] + '_' + str(i) + '.' + name_parts[1]

			src = open(src_name, 'rt')
			dst = open(dst_name, 'wt')

			for line in src:
				dst.write(line.replace('SEED', str(rng_seed)).replace('DROP_SEV', str(drop_sev)))
			src.close()
			dst.close()

			os.system('sbatch ./' + dst_name)
			print(dst_name)
