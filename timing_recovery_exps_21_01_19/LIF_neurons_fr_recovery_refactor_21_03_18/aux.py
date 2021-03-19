# miscellaneous useful functions and classes
import numpy as np
import os
from copy import deepcopy as copy


class Generic(object):
    """Class for generic object."""
    
    def __init__(self, **kwargs):
        
        for k, v in kwargs.items():
            self.__dict__[k] = v


def c_tile(x, n):
    """Create tiled matrix where each of n cols is x."""
    return np.tile(x.flatten()[:, None], (1, n))


def r_tile(x, n):
    """Create tiled matrix where each of n rows is x."""
    return np.tile(x.flatten()[None, :], (n, 1))


class BatchedArray(object):

    def __init__(self, generator, chunk_size, num_chunks, batched_dim=0):
        self.generator = generator
        self.chunk_size = chunk_size
        self.num_chunks = num_chunks
        self.len = chunk_size * num_chunks
        self.batched_dim = batched_dim

        data = next(generator)

        if num_chunks > 1:
            next_data = next(generator)
            self.data = np.concatenate([data, next_data], axis=batched_dim)
            self.min_in_scope = 0
            self.max_in_scope = 2 * chunk_size
        else:
            self.data = data
            self.min_in_scope = 0
            self.max_in_scope = chunk_size

        self.dims =  self.data.shape
        self.n_dims = len(self.dims)

        self.chunk_count = 0

    def gen_batch_mask(self, start=None, stop=None, step=None):
        batch_mask = [slice(None)] * self.batched_dim + [slice(start, stop, step)] + [slice(None)] * (self.n_dims - self.batched_dim - 1)
        return tuple(batch_mask)

    def slide(self):
        self.data[self.gen_batch_mask(start=0, stop=self.chunk_size)] = self.data[self.gen_batch_mask(start=self.chunk_size, stop=2*self.chunk_size)]
        self.data[self.gen_batch_mask(start=self.chunk_size, stop=2*self.chunk_size)] = next(self.generator)

        self.chunk_count += 1
        self.min_in_scope += self.chunk_size
        self.max_in_scope += self.chunk_size

    def manage_slice(self, sl):
        if sl.start is None or sl.stop is None:
            raise IndexError('Min and max must be specified explicitly')

        if sl.start < self.min_in_scope:
            raise IndexError('Requested min is out of scope')

        if sl.stop > (self.max_in_scope + self.chunk_size):
            raise IndexError('Requested max is out of scope')

        if sl.stop < sl.start:
            raise IndexError('Max of range must be larged than min')

        if (sl.start <= self.max_in_scope - self.chunk_size) and (sl.stop > self.max_in_scope):
            raise IndexError('Must slice along batches')

        if sl.stop - sl.start > 2 * self.chunk_size:
            raise IndexError('Requested range must be smaller than chunk size')

        if sl.stop > self.max_in_scope and (self.chunk_count < self.num_chunks - 2):
            self.slide()

    def manage_scalar_index(self, index):
        if index < self.min_in_scope:
            raise IndexError('Requested index is out of scope')

        if index > self.max_in_scope + self.chunk_size:
            raise IndexError('Requested index is out of scope')

        if index >= self.max_in_scope and self.chunk_count < self.num_chunks - 1:
            self.slide()

    def shift_slice(self, key):
        return slice(key.start - self.chunk_count * self.chunk_size, key.stop - self.chunk_count * self.chunk_size, key.step)

    def shift_scalar_index(self, key):
        return key - self.chunk_count * self.chunk_size

    def amend_batched_key(self, key):
        if type(key) is slice:
            self.manage_slice(key)
            key = self.shift_slice(key)
        elif type(key) is int:
            self.manage_scalar_index(key)
            key = self.shift_scalar_index(key)
        else:
            raise TypeError('Batched key must be slice or int')
        return key

    def handle_get_or_set(self, key, values=None):
        if type(key) is tuple and self.n_dims != len(key):
            raise IndexError('Wrong number of dims for array')
        if (type(key) is slice or type(key) is int) and self.n_dims > 1:
            raise IndexError('Too few dims for array')

        if type(key) is tuple:
            batched_key = key[self.batched_dim]
            batched_key = self.amend_batched_key(batched_key)

            amended_key = list(key)
            amended_key[self.batched_dim] = batched_key

            if values is not None:
                self.data[tuple(amended_key)] = values
            else:
                return self.data[tuple(amended_key)]
        else:
            key = self.amend_batched_key(key)  
            if values is not None:
                self.data[key] = values
            else:
                return self.data[key]

    def __getitem__(self, key):
        return self.handle_get_or_set(key)

    def __setitem__(self, key, values):
        self.handle_get_or_set(key, values)

    def __len__(self):
        return self.len


def batched_array_gen(length, batch_size, func):
    def gen():
        for i in range(0, length, batch_size):
            yield func(i, i + batch_size)
    return gen


def batched_array(batch_size, shape, modifier, batch_dim=0):
    length = shape[batch_dim]
    shape_for_partial = list(shape)
    shape_for_partial[batch_dim] = batch_size
    def func(start, stop):
        return modifier(start, stop) * np.ones(shape_for_partial)
    return BatchedArray(batched_array_gen(length, batch_size, func)(), batch_size, np.ceil(length/batch_size))


def batched_zeros(batch_size, shape, batch_dim=0):
    def modifier(start, stop):
        return 0.
    return batched_array(batch_size, shape, modifier, batch_dim=batch_dim)


def batched_nans(batch_size, shape, batch_dim=0):
    def modifier(start, stop):
        return np.nan
    return batched_array(batch_size, shape, modifier, batch_dim=batch_dim)


def batched_falses(batch_size, shape, batch_dim=0):
    length = shape[batch_dim]
    shape_for_partial = list(shape)
    shape_for_partial[batch_dim] = batch_size
    def func(start, stop):
        return np.zeros(shape_for_partial, dtype=bool)
    return BatchedArray(batched_array_gen(length, batch_size, func)(), batch_size, np.ceil(length/batch_size))


def burst_count(ndarr):
	cnts_per_nrn = ndarr.sum(axis=0)
	return cnts_per_nrn, cnts_per_nrn.mean(), cnts_per_nrn.std()


def uncertainty_plot(ax, x, y, y_stds):
	ax.plot(x, y)
	ax.fill_between(x, y - y_stds, y + y_stds)


def bin_occurrences(occurrences, min_idx=0, max_idx=None, bin_size=1):
    if max_idx is None:
        max_idx = occurrences.max() + 1
    binned = np.zeros(max_idx - min_idx)
    for n in occurrences:
        if n >= max_idx or n < min_idx:
            raise IndexError(f'index {n} is out of bounds for min {min_idx} and max {max_idx}')
        binned[n - min_idx] += 1
    return binned


def calc_degree_dist(mat):
    degree_freqs = bin_occurrences(np.count_nonzero(mat, axis=1))
    return np.arange(len(degree_freqs)), degree_freqs


def rand_n_ones_in_vec_len_l(n, l):
    if n > l:
        raise ValueError('n cannot be greater than l')
    vec = np.concatenate([np.ones(n, int), np.zeros(l - n, int)])
    return vec[np.random.permutation(l)]


def rand_per_row_mat(n, shape):
    return np.stack([rand_n_ones_in_vec_len_l(n, shape[1]) for i in range(shape[0])])


def mat_1_if_under_val(val, shape):
    return np.where(np.random.rand(*shape) < val, 1, 0)


def dropout_on_mat(mat, percent):
    dropout_indices = rand_n_ones_in_vec_len_l(int((1. - percent) * mat.shape[1]), mat.shape[1])
    m = copy(mat)
    for idx, val in enumerate(dropout_indices):
        m[:, idx] = val * m[:, idx]
    return m
