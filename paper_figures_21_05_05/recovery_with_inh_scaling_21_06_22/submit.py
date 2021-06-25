import os

base_path = os.curdir
scripts = ['submit_all_rules.slurm']

for i, rng_seed in enumerate(range(2025, 2045)):
	for src_name in scripts:
		name_parts = src_name.split('.')
		dst_name = name_parts[0] + '_' + str(i) + '.' + name_parts[1]

		src = open(src_name, 'rt')
		dst = open(dst_name, 'wt')

		for line in src:
			dst.write(line.replace('SEED', str(rng_seed)))
		src.close()
		dst.close()

		os.system('sbatch ./' + dst_name)
		print(dst_name)
