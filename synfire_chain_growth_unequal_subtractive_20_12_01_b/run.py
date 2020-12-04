from copy import deepcopy as copy
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.linalg import block_diag
import pandas as pd
from tqdm import tqdm
import pickle
from collections import OrderedDict
import os
from scipy.ndimage.interpolation import shift
from functools import reduce

from aux import *
from disp import *
from ntwk import LIFNtwkG
from utils.general import *

cc = np.concatenate

# PARAMS
## NEURON AND NETWORK MODEL
M = Generic(
    # Excitatory membrane
    C_M_E=1e-6,  # membrane capacitance
    G_L_E=.4e-3,  # membrane leak conductance (T_M (s) = C_M (F/cm^2) / G_L (S/cm^2))
    E_L_E=-.06,  # membrane leak potential (V)
    V_TH_E=-.05,  # membrane spike threshold (V)
    T_R_E=6e-3,  # refractory period (s)
    E_R_E=-0.055, # reset voltage (V)
    
    # Inhibitory membrane
    #C_M_I=1e-6,
    #G_L_E=.1e-3, 
    #E_L_I=-.06,
    #V_TH_E=-.05,
    #T_R_I=.002,
    
    # syn rev potentials and decay times
    E_E=0, E_I=-.07, T_E=.004, T_I=.004,
    
    W_U_E=0.5e-3,
    W_MAX=0.26 * 0.004,
    M=9.,
    ETA=0.00005,
    EPSILON=0.05,
    
    W_E_I_R=0,
    W_I_E_R=0,
    
    N_EXC=50,
    N_INH=1,
    
    DRIVING_HZ=2.5, # 10 Hz lambda Poisson input to system
    N_DRIVING_CELLS=10,
    PROJECTION_NUM=5,
    W_INITIAL=0.,
    
    # OTHER INPUTS
    SGM_N=.5e-9,  # noise level (A*sqrt(s))
    I_EXT_B=0,  # additional baseline current input
)

## SMLN
S = Generic(RNG_SEED=0, DT=0.1e-3)

print('T_M_E =', 1000*M.C_M_E/M.G_L_E, 'ms')  # E cell membrane time constant (C_m/g_m)

### RUN_TEST function

def run_test(m, show_connectivity=True, repeats=1, n_show_only=None, add_noise=True, dropouts=[{'E': 0, 'I': 0}]):
    
    n_cells_driven = m.N_DRIVING_CELLS * m.PROJECTION_NUM
    
    w_u_e = m.W_U_E * 4 / m.PROJECTION_NUM * block_diag(*[np.random.rand(m.PROJECTION_NUM, m.PROJECTION_NUM) for n in range(m.N_DRIVING_CELLS)])
    
    ## input weights
    w_u = {
        # localized inputs to trigger activation from start of chain
        'E': np.block([
            [ w_u_e, np.zeros([n_cells_driven, m.N_EXC + m.N_INH]) ],
            [ np.zeros([m.N_EXC + m.N_INH - n_cells_driven, m.N_EXC + m.N_INH + n_cells_driven]) ],
        ]),

        'I': np.zeros((m.N_EXC + m.N_INH, n_cells_driven + m.N_EXC + m.N_INH)),
    }
    
    w_initial_max = m.W_INITIAL
    
    e_i_r = np.stack([rand_n_ones_in_vec_len_l(int(m.N_EXC), m.N_EXC) for i in range(m.N_INH)])
    
    i_e_r = np.stack([rand_n_ones_in_vec_len_l(int(m.N_INH), m.N_INH) for i in range(m.N_EXC)])
    
    connectivity = np.stack([rand_n_ones_in_vec_len_l(m.N_EXC, m.N_EXC) for i in range(m.N_EXC)])
    w_e_e_r = np.random.rand(m.N_EXC, m.N_EXC) * connectivity * w_initial_max
    np.fill_diagonal(w_e_e_r, 0.)
    
    ## recurrent weights
    w_r_base = {
        'E': np.block([
            [ w_e_e_r, np.zeros((m.N_EXC, m.N_INH)) ],
            [ e_i_r * m.W_E_I_R,  np.zeros((m.N_INH, m.N_INH)) ],
        ]),
        'I': np.block([
            [ np.zeros((m.N_EXC, m.N_EXC)), i_e_r * m.W_I_E_R ],
            [ np.zeros((m.N_INH, m.N_EXC)), np.zeros((m.N_INH, m.N_INH)) ],
        ]),
    }
    
    w_r_for_dropouts = []
    for dropout in dropouts:
        w_r = copy(w_r_base)
        # w_r['E'][:, :m.N_EXC] = dropout_on_mat(w_r['E'][:, :m.N_EXC], dropout['E'])
        # w_r['I'][:, m.N_EXC:] = dropout_on_mat(w_r['I'][:, m.N_EXC:], dropout['I'])
        w_r_for_dropouts.append(w_r)
    
    # generate timesteps and initial excitatory input window
    t = np.arange(0, S.T, S.DT)
    
    all_rsps = []

    # run simulation for same set of parameters
    for rp_idx in range(repeats):
        show_trial = (type(n_show_only) is int and rp_idx < n_show_only)

        rsps_for_trial = []

        ## external currents
        if add_noise:
            i_ext = m.SGM_N/S.DT * np.random.randn(len(t), m.N_EXC + m.N_INH) + m.I_EXT_B
        else:
            i_ext = m.I_EXT_B * np.ones((len(t), m.N_EXC + m.N_INH))

        ## inp spks
        spks_u_base = np.zeros((len(t), n_cells_driven + m.N_EXC + m.N_INH), dtype=int)

        # trigger inputs
        spks_u = copy(spks_u_base)
        correlated_inputs = np.concatenate([np.random.poisson(m.DRIVING_HZ * S.DT, size=(len(t), 1)) for i in range(m.N_DRIVING_CELLS)], axis=1)
        
        spks_u[:, :n_cells_driven] = np.repeat(correlated_inputs, m.PROJECTION_NUM, axis=1)
        neuron_active_mask = np.random.rand(spks_u.shape[0], n_cells_driven) < 0.8
        spks_u[:, :n_cells_driven] = spks_u[:, :n_cells_driven] & neuron_active_mask
        
        rsps_for_trial = []
        
        for d_idx, dropout in enumerate(dropouts):
            
            w_r_for_dropout = w_r_for_dropouts[d_idx]
            
            w_max = m.W_MAX / m.M
            
            if show_connectivity:
                graph_weights(copy(w_r_for_dropout), copy(w_u), v_max=w_max)

            ntwk = LIFNtwkG(
                c_m=m.C_M_E,
                g_l=m.G_L_E,
                e_l=m.E_L_E,
                v_th=m.V_TH_E,
                v_r=m.E_R_E,
                t_r=m.T_R_E,
                e_s={'E': M.E_E, 'I': M.E_I},
                t_s={'E': M.T_E, 'I': M.T_E},
                w_r=copy(w_r_for_dropout),
                w_u=copy(w_u),
                plasticity_indices=np.arange(m.N_EXC),
                connectivity=connectivity,
                W_max=m.W_MAX,
                m=m.M,
                eta=m.ETA,
                epsilon=m.EPSILON,
                dt=S.DT,
                gamma=m.gamma,
                output_freq=100,
            )

            clamp = Generic(
                v={0: np.repeat(m.E_L_E, m.N_EXC + m.N_INH)}, spk={})

            # run smln
            rsp = ntwk.run(dt=S.DT, clamp=clamp, i_ext=i_ext, spks_u=spks_u)
            
            if show_connectivity:
                graph_weights(rsp.ntwk.w_r, rsp.ntwk.w_u, v_max=w_max)
            
            w_r_e = rsp.ntwk.w_r['E'][:m.N_EXC, :m.N_EXC]
            graph_weight_matrix(w_r_e, 'Exc->Exc Weights\n', v_max=w_max)
            graph_weight_matrix(np.sqrt(np.dot(w_r_e, w_r_e.T)), 'W * W.T \n', v_max=w_max)
                
            rsps_for_trial.append({
                'spks_t': copy(rsp.spks_t),
                'spks_c': copy(rsp.spks_c),
                'spks_u': spks_u.nonzero(),
                'w_r': copy(rsp.ntwk.w_r)
            })
        all_rsps.append(rsps_for_trial)
    return all_rsps

def quick_plot(m, repeats=1, show_connectivity=True, n_show_only=None, add_noise=True, dropouts=[{'E': 0, 'I': 0}]):
    all_rsps = run_test(m, show_connectivity=show_connectivity, repeats=repeats,
                        n_show_only=n_show_only, add_noise=add_noise, dropouts=dropouts)
    
    colors = ['black', 'blue', 'green', 'purple', 'brown']
        
    for idx_r, rsps in enumerate(all_rsps):
        for idx_do, rsp_for_dropout in enumerate(rsps):
            
            raster = np.stack([rsp_for_dropout['spks_t'], rsp_for_dropout['spks_c']])

            exc_raster = raster[:, raster[1, :] < m.N_EXC]
            inh_raster = raster[:, raster[1, :] >= m.N_EXC]

            show_trial = (type(n_show_only) is None) or (type(n_show_only) is int and idx_r < n_show_only)
            print(show_trial)
            # if show_trial:
            gs = gridspec.GridSpec(1, 1)
            fig = plt.figure(figsize=(18, 6), tight_layout=True)
            axs = [fig.add_subplot(gs[0])]

            axs[0].scatter(exc_raster[0, :] * 1000, exc_raster[1, :], s=1, c='black', zorder=0, alpha=1)
            axs[0].scatter(inh_raster[0, :] * 1000, inh_raster[1, :], s=1, c='red', zorder=0, alpha=1)

            axs[0].set_ylim(-1, m.N_EXC + m.N_INH)
            axs[0].set_xlim(0, S.T * 1000)
            axs[0].set_ylabel('Cell Index')
            axs[0].set_xlabel('Time (ms)')
            
            set_font_size(axs[0], 14)
    return all_rsps

S.T = 500
S.DT = 0.2e-3
m2 = copy(M)
m2.EPSILON = 0.001
m2.ETA = 0.00001
m2.gamma = 0.1
m2.W_E_I_R = 1e-5
m2.W_I_E_R = m2.W_E_I_R * m2.N_EXC
m2.T_R_E = 10e-3
m2.W_MAX = 0.26 * 0.004
m2.M = 9
m2.W_INITIAL = m2.W_MAX / (m2.M * m2.N_EXC)

all_rsps = quick_plot(m2)
