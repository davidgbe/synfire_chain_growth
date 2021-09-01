from copy import deepcopy as copy
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import pandas as pd
from tqdm import tqdm
import pickle
from collections import OrderedDict
import os
from scipy.ndimage.interpolation import shift
import scipy.io as sio
from scipy.optimize import curve_fit
from functools import reduce
import argparse
import csv

from aux import *
from disp import *
from ntwk import LIFNtwkG
from utils.general import *
from utils.file_io import *

processed = []

with open('./data/data.csv', newline='') as data_file:
	reader = csv.reader(data_file, delimiter=' ', quotechar='|')
	current_dropout = None


	for row in reader:
		if len(row) == 0:
			continue
		if row[0] == 'new':
			processed.append({})
		elif row[0] == 'trial':
			current_dropout = float(row[1])
			processed[-1][current_dropout] = {'data': [], 'status': 'c', 'stats': {}}
			if len(row) > 2:
				processed[-1][current_dropout]['status'] = row[2]
		else:
			processed[-1][current_dropout]['data'].append(float(row[0]))

ratios = {0: [], 0.1: [], 0.2: [], 0.3: []}

for data_for_arch in processed:
	for dropout_per, meta in data_for_arch.items():
		c_mean, c_std = np.mean(meta['data'][:10]), np.std(meta['data'][:10])
		p_mean, p_std = np.mean(meta['data'][-10:]), np.std(meta['data'][-10:])

		if meta['status'] == 'c':
			ratios[dropout_per].append(p_mean / c_mean)

fig1, ax1 = plt.subplots()
ax1.set_title('')
ax1.set_ylabel('Perturbed / Unperturbed Propagation Speed Ratio')
ax1.set_xlabel('Dropout')

for dropout_per, data in ratios.items():
	print(np.mean(data), np.std(data), len(data))
	ax1.scatter([dropout_per] * len(data), data, s=8, c='black')

fig1.show()



