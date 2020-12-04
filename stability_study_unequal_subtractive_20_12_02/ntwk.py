"""
Classes/functions for a few biological spiking network models.
"""
from copy import deepcopy as copy
import numpy as np
from scipy.sparse import csc_matrix
import scipy.io as sio
import os

from utils.general import zero_pad
from aux import Generic, c_tile, r_tile

cc = np.concatenate

# Conductance-based LIF network
class LIFNtwkG(object):
    """Network of leaky integrate-and-fire neurons with *conductance-based* synapses."""
    
    def __init__(self, c_m, g_l, e_l, v_th, v_r, t_r, e_s, t_s, w_r, w_u, plasticity_indices, connectivity, W_max, m, eta, epsilon, dt, gamma,
        sparse=False, output=True, output_freq=1000, homeo=True):
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
        
        if sparse:  # sparsify connectivity if desired
            self.w_r = {k: csc_matrix(w_r_) for k, w_r_ in w_r.items()}
            self.w_u = {k: csc_matrix(w_u_) for k, w_u_ in w_u.items()} if w_u is not None else w_u
        else:
            self.w_r = w_r
            self.w_u = w_u

        self.syns = list(self.e_s.keys())

        self.plasticity_indices = copy(plasticity_indices)
        self.plasticity_mask = np.zeros((self.w_r['E'].shape[0]), dtype=bool)
        self.plasticity_mask[plasticity_indices] = True

        w_r_e_plastic_mask = np.stack([self.plasticity_mask for i in range(self.w_r['E'].shape[0])])
        self.w_r_e_plastic_mask = w_r_e_plastic_mask & w_r_e_plastic_mask.T

        w_r_e_plastic = self.w_r['E'][self.w_r_e_plastic_mask].reshape(len(self.plasticity_indices), len(self.plasticity_indices))
        self.connections = connectivity

        self.W_max = W_max
        self.m = m
        self.w_max = self.W_max / self.m
        self.eta = eta
        self.epsilon = epsilon
        self.tau_stdp = 0.02 # 20 ms
        self.tau_cut = int(self.tau_stdp / 0.02e-3)
        self.taus = np.arange(self.tau_cut)
        self.kernel = np.exp(-self.taus * dt / self.tau_stdp)

        self.gamma = gamma

        self.homeo = homeo

        self.output = output
        self.output_freq = output_freq

    def update_w(self, t_ctr, spks, dt):
        # STDP update for calculation of delta_stdp(t)
        curr_spks = spks[-1, :]

        o = np.zeros([spks.shape[1], spks.shape[1]])
        for i_t in range(spks.shape[0] - 1):
            o += np.outer(curr_spks, self.kernel[i_t] * spks[-(i_t + 1), :])

        w_r_e_plastic = self.w_r['E'][self.w_r_e_plastic_mask].reshape(len(self.plasticity_indices), len(self.plasticity_indices))
        normed_w_r_e = w_r_e_plastic / self.w_max + (self.connections * 0.001)

        delta_stdp = np.multiply(normed_w_r_e, (o - o.T))
        np.fill_diagonal(delta_stdp, 0.)


        # calculation of balance of outgoing weight totals (theta_i*)
        theta_out = np.sum(w_r_e_plastic, axis=0) + np.sum(delta_stdp, axis=0) - self.W_max
        if not self.homeo:
            theta_out[theta_out < 0] = 0
        theta_out[theta_out < 0] = self.gamma * theta_out[theta_out < 0]
        theta_out = theta_out.reshape(1, theta_out.shape[0])

        # calculation of balance of incoming weight totals (theta_*i)
        theta_in = np.sum(w_r_e_plastic, axis=1) + np.sum(delta_stdp, axis=1) - self.W_max
        if not self.homeo:
            theta_in[theta_in < 0] = 0
        theta_in[theta_in < 0] = self.gamma * theta_in[theta_in < 0]
        theta_in = theta_in.reshape(theta_in.shape[0], 1)

        weight_update = self.eta * (delta_stdp - self.epsilon * (theta_in * self.connections + theta_out * self.connections))

        np.fill_diagonal(weight_update, 0.)

        self.w_r['E'][self.w_r_e_plastic_mask] += weight_update.flatten()
        under_zero = self.w_r_e_plastic_mask & (self.w_r['E'] < 0)
        self.w_r['E'][under_zero] = 0
        over_w_max = self.w_r_e_plastic_mask & (self.w_r['E'] > self.w_max)
        self.w_r['E'][over_w_max] = self.w_max
        
    def run(self, dt, clamp, i_ext, output_dir_name, spks_u=None):
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

        if self.output:
            output_dir = f'./data/{output_dir_name}'
            os.makedirs(output_dir)

        
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

        burst_t = np.arange(0, t_r_int[0], int(t_r_int[0] / 3), dtype=int)
        
        # loop over timesteps
        for t_ctr in range(len(i_ext) - t_r_int[0]):
            
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
                spks_for_t_ctr = vs[t_ctr, :] >= v_th

                for b_t in burst_t:
                    spks[b_t + t_ctr, spks_for_t_ctr] = 1

            stdp_start = t_ctr - self.tau_cut 
            stdp_start = 0 if stdp_start < 0 else stdp_start
            self.update_w(t_ctr, spks[stdp_start:(t_ctr + 1), self.plasticity_indices], dt)

            if self.output and (t_ctr % self.output_freq == 0):
                sio.savemat(output_dir + '/' + f'{zero_pad(int(t_ctr / self.output_freq), 6)}', {'w_r_e': self.w_r['E']})

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