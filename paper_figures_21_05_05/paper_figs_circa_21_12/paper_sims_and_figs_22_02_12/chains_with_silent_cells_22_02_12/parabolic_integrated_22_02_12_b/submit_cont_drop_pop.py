import os
from collections import OrderedDict
import functools
from utils.file_io import *

base_path = os.curdir
scripts = [
	'submit_cont_drop_pop.slurm',
]
load_run_name = ['settle', 'GAMMA_0.01']

### functions

def replace_all(line, repl_dict):
	s = line
	for k, v in repl_dict.items():
		if k == 'TITLE' or k == 'LOADED_RUN_NAME': 
			continue
		s = s.replace(k, v)

	if 'LOADED_RUN_NAME' in repl_dict:
		s = s.replace('LOADED_RUN_NAME', repl_dict['LOADED_RUN_NAME'])

	if 'TITLE' in repl_dict:
		s = s.replace('TITLE', repl_dict['TITLE'])

	return s

def format_title(params):
	title = ''
	for k, v in params.items():
		if k == 'LOADED_RUN_NAME':
			v = v[-26:]
		title += ('_' + k + '_' + v)
	return title

def iter_range(r, n):
	if n == 1:
		yield (0, r[0])
	else:
		for i in range(n):
			yield (i, i * (r[1] - r[0]) / (n - 1) + r[0])

def map_to_list(func, l):
    '''
    Maps the list 'l' through the function 'func'
    Parameters
    ----------
    func : function
        Takes a single argument of type of 'l'
    l : list
    '''
    return list(map(func, l))

def reduce_mult(l):
    return functools.reduce(lambda e1, e2: e1 * e2, l, 1)

# multidimensional generalization of a cartesian proces
# given [2, 4, 6] and [2, 5, 8, 9] generates
# [[2, 2, 2, 2, 4, 4, 4, 4, 6, 6, 6, 6], [2, 5, 8, 9, 2, 5, 8, 9, 2, 5, 8, 9]]
def cartesian(*arrs):
    domain = map_to_list(lambda a: len(a), arrs)
    coordinate_lists = []
    for i, dim in enumerate(domain):
        coords = []
        mult = 1
        if i != len(domain) - 1:
            mult = reduce_mult(domain[i+1:])
        for e in arrs[i]:
            coords += (mult * [e])
        repeat_factor = reduce_mult(domain[0:i])
        if repeat_factor > 0:
            coords *= repeat_factor
        coordinate_lists.append(coords)
    return coordinate_lists

def pad_zeros(to_pad, length):
	padded = str(to_pad)
	while len(padded) < length:
		padded = '0' + padded
	return padded

### operating code

batch_size = 10

params = OrderedDict()

if type(load_run_name) is list:
    all_dirs = filter_list_by_name_frags(all_in_dir('./robustness'), load_run_name)
else:
    all_dirs = filter_list_by_name_frags(all_in_dir('./robustness'), [load_run_name])

params['ALPHA_1'] = [ str(5e-2) ]
params['BETA'] = [ str(1e-2) ]
params['GAMMA'] = [ str(1e-2) ]
params['SYN_PROP_DIST'] = [ str(12) ]
params['DROP_SEV'] = [str(0.55), str(0.6), str(0.65)]
params['LOADED_RUN_NAME'] = [d for d in all_dirs]

seeds = np.arange(len(params['LOADED_RUN_NAME'])) + 2090

run_name_to_seed = {}
for i, d in enumerate(params['LOADED_RUN_NAME']):
	run_name_to_seed[d] = seeds[i]

# for key in params.keys():
# 	if key == 'SEED' or key == 'LOADED_RUN_NAME':
# 		continue
# 	params[key] = [str(v[1]) for v in iter_range(params[key][0], params[key][1])]

all_values = cartesian(*(params.values()))
n_scripts = len(all_values[0])
n_scripts_exec = 0

for src_name in scripts:
	for n in range(0, n_scripts, batch_size):

		name_parts = src_name.split('.')
		dst_name = name_parts[0] + '_' + pad_zeros(n, 4) + '.' + name_parts[1]

		src = open(src_name, 'rt')
		dst = open(dst_name, 'wt')

		for line in src:
			if line.find('python run.py') >= 0:
				for batch_idx in range(batch_size):
					if n + batch_idx >= n_scripts:
						continue

					augmented_params = {}

					for param_idx, v in enumerate(params.keys()):
						augmented_params[v] = all_values[param_idx][n + batch_idx]

					augmented_params['SEED'] = run_name_to_seed[augmented_params['LOADED_RUN_NAME']]
					augmented_params['TITLE'] = format_title(augmented_params)

					line_replaced = replace_all(line, augmented_params)
					dst.write(line_replaced)
			else:
				dst.write(line)
		src.close()
		dst.close()

		os.system('sbatch ./' + dst_name)
		print(dst_name)
