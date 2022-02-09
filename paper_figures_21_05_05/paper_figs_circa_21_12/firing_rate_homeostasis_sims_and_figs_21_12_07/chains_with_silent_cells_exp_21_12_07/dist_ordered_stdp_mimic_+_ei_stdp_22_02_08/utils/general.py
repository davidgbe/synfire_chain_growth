import numpy as np
import datetime

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

def time_stamp(s=False):
    if s:
        return datetime.datetime.now().strftime('%Y-%m-%d--%H:%M--%S')
    else:
        return datetime.datetime.now().strftime('%Y-%m-%d--%H:%M')

def zero_pad(arg, size):
    s = str(arg)
    while len(s) < size:
        s = '0' + s
    return s

def rand_bin_array_with_percentage_ones(l, num_ones):
    a = np.zeros(l)
    a[:num_ones] = 1
    np.random.shuffle(a)
    return a

def sprs_mat_with_rand_percent_cnxns(shape, row_percent):
    if type(row_percent) is GaussianPDF:
        num_ones_vec = shape[0] * row_percent(shape[1])
        num_ones_vec[num_ones_vec < 0] = 0
        num_ones_vec[num_ones_vec > shape[0]] = shape[0]
        num_ones_vec = num_ones_vec.astype(int)
        stacked = np.stack([rand_bin_array_with_percentage_ones(shape[0], num_ones) for num_ones in num_ones_vec])
    else:
        num_ones = int(row_percent * shape[0])
        stacked = np.stack([rand_bin_array_with_percentage_ones(shape[0], num_ones) for i in range(shape[1])])
    return stacked

def outer_product_n_dim(*args):
    params = [p for p in args]
    outer_product = np.meshgrid(*params, sparse=False, indexing='ij')
    return np.stack([p_vals.flatten() for p_vals in outer_product], axis=1)

# returns a subset of dataframe
def select(df, selection):
    criteria = []
    for col in selection:
        criteria.append(df[col] == selection[col])
    return df[np.all(criteria, axis=0)]

# returns a list of unique values in the given Pandas dataframe for each column name specified 
def to_unique_vals(df, col_names):
    if type(col_names) is str:
        col_names = [col_names]
    return tuple([df[col_name].unique() for col_name in col_names])

class GaussianPDF(object):
    '''Class for drawing from Gaussian distribution'''

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, size):
        return np.random.normal(self.mean, self.std, size)
