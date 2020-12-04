# miscellaneous useful functions and classes
import numpy as np
import os


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


def burst_count(ndarr):
	cnts_per_nrn = ndarr.sum(axis=0)
	return cnts_per_nrn, cnts_per_nrn.mean(), cnts_per_nrn.std()

def uncertainty_plot(ax, x, y, y_stds):
	ax.plot(x, y)
	ax.fill_between(x, y - y_stds, y + y_stds)