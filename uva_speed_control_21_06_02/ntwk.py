"""
Classes/functions for a few biological spiking network models.
"""
from copy import deepcopy as copy
import numpy as np
from scipy.sparse import csc_matrix, csr_matrix, kron
import scipy.io as sio
import os

from utils.general import zero_pad
from aux import Generic, c_tile, r_tile, dropout_on_mat

cc = np.concatenate

# Conductance-based LIF network
class LIFNtwkG(object):
    """Network of leaky integrate-and-fire neurons with *conductance-based* synapses."""
    
    def __init__(self, c_m, g_l, e_l, v_th, v_r, t_r, e_s, t_s, w_r, w_u, i_b, f_b, t_b, sigma_b, sparse=False):
        # ntwk size
        n = next(iter(w_r.values())).shape[0]

        # process inputs
        if type(t_r) in [float, int]:
            t_r = t_r * np.ones(n)
            
        if type(v_r) in [float, int]:
            v_r = v_r * np.ones(n)
            
        self.n = n
        self.c_m = c_m
        self.g_l = g_l
        self.t_m = c_m / g_l
        self.e_l = e_l
        self.v_th = v_th
        self.v_r = v_r
        self.t_r = t_r
        
        self.e_s = e_s
        self.t_s = t_s

        self.i_b = i_b # indices with bursting cells
        self.f_b = f_b # burst frequency
        self.t_b = t_b # burst duration
        self.sigma_b = sigma_b
        
        if sparse:  # sparsify connectivity if desired
            self.w_r = {k: csc_matrix(w_r_) for k, w_r_ in w_r.items()}
            self.w_u = {k: csc_matrix(w_u_) for k, w_u_ in w_u.items()} if w_u is not None else w_u
        else:
            self.w_r = w_r
            self.w_u = w_u

        self.syns = list(self.e_s.keys())

    def run(self, dt, clamp, i_ext, spks_u=None):
        """
        Run simulation.
        
        :param dt: integration timestep (s)
        :param clamp: dict of times to clamp certain variables (e.g. to initialize)
        :param i_ext: external current inputs (either 1D or 2D array, length = num timesteps for smln)
        :param spks_up: upstream inputs
        """
        n = self.n
        n_t = len(i_ext)
        syns = self.syns
        c_m = self.c_m
        g_l = self.g_l
        e_l = self.e_l
        v_th = self.v_th
        v_r = self.v_r
        t_r = self.t_r
        t_r_int = np.round(t_r/dt).astype(int)
        e_s = self.e_s
        t_s = self.t_s
        w_r = self.w_r
        w_u = self.w_u

        burst_base = np.zeros((int(self.t_b/dt)))
        l_b = len(burst_base)

        def burst():
            burst = copy(burst_base)
            for i in np.arange(0, self.t_b, 1./self.f_b):
                if i == 0:
                    burst[int(i)] = 1
                else:
                    noise = np.random.normal(0, self.sigma_b)
                    burst[int(i/dt + noise)] = 1
            return burst
        
        # make data storage arrays
        gs = {syn: np.nan * np.zeros((n_t, n)) for syn in syns}
        vs = np.nan * np.zeros((n_t, n))
        spks = np.zeros((n_t, n), dtype=bool)
        
        rp_ctr = np.zeros(n, dtype=int)
        
        # convert float times in clamp dict to time idxs
        ## convert to list of tuples sorted by time
        tmp_v = sorted(list(clamp.v.items()), key=lambda x: x[0])
        tmp_spk = sorted(list(clamp.spk.items()), key=lambda x: x[0])
        clamp = Generic(
            v={int(round(t_/dt)): f_v for t_, f_v in tmp_v},
            spk={int(round(t_/dt)): f_spk for t_, f_spk in tmp_spk})
        
        # loop over timesteps
        for t_ctr in range(len(i_ext) - l_b - 1):

            # update conductances
            for syn in syns:
                if t_ctr == 0:
                    gs[syn][t_ctr, :] = 0
                else:
                    g = gs[syn][t_ctr-1, :]
                    # get weighted spike inputs
                    ## recurrent
                    inp = w_r[syn].dot(spks[t_ctr-1, :])
                    ## upstream
                    if spks_u is not None:
                        if syn in w_u:
                            inp += w_u[syn].dot(spks_u[t_ctr-1, :])
                    
                    # update conductances from weighted spks
                    gs[syn][t_ctr, :] = g + (dt/t_s[syn])*(-gs[syn][t_ctr-1, :]) + inp
            
            # update voltages
            if t_ctr in clamp.v:  # check for clamped voltages
                vs[t_ctr, :] = clamp.v[t_ctr]
            else:  # update as per diff eq
                v = vs[t_ctr-1, :]
                # get total current input
                i_total = -g_l*(v - e_l)  # leak
                i_total += np.sum([-gs[syn][t_ctr, :]*(v - e_s[syn]) for syn in syns], axis=0)  # synaptic
                i_total += i_ext[t_ctr]  # external
                
                # update v
                vs[t_ctr, :] = v + (dt/c_m)*i_total
                
                # clamp v for cells still in refrac period
                vs[t_ctr, rp_ctr > 0] = v_r[rp_ctr > 0]
            
            # update spks
            if t_ctr in clamp.spk:  # check for clamped spikes
                spks[t_ctr, :] = clamp.spk[t_ctr]
            else:  # check for threshold crossings
                crossed_thresh = vs[t_ctr, :] >= self.v_th
                crossed_thresh_bursting = crossed_thresh & self.i_b
                crossed_thresh_spiking = crossed_thresh & (~self.i_b)

                spks[t_ctr, crossed_thresh_spiking.nonzero()[0]] = 1
                b = burst()
                for j in np.arange(l_b):
                    try:
                        spks[t_ctr + j, crossed_thresh_bursting.nonzero()[0]] = b[j]
                    except IndexError as e:
                        breakpoint()

            # reset v and update refrac periods for nrns that spiked
            vs[t_ctr, spks[t_ctr, :]] = v_r[spks[t_ctr, :]]
            rp_ctr[spks[t_ctr, :]] = t_r_int[spks[t_ctr, :]] + 1
            
            # decrement refrac periods
            rp_ctr[rp_ctr > 0] -= 1
            
        t = dt*np.arange(n_t, dtype=float)
        
        # convert spks to spk times and cell idxs (for easy access l8r)
        tmp = spks.nonzero()
        spks_t = dt * tmp[0]
        spks_c = tmp[1]
        
        return Generic(dt=dt, t=t, gs=gs, vs=vs, spks=spks, spks_t=spks_t, spks_c=spks_c, i_ext=i_ext, ntwk=self)