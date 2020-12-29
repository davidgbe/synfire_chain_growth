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
    T_R_E=2e-3,  # refractory period (s)
    E_R_E=-0.06, # reset voltage (V)
    
    # Inhibitory membrane
    #C_M_I=1e-6,
    #G_L_E=.1e-3, 
    #E_L_I=-.06,
    #V_TH_E=-.05,
    #T_R_I=.002,
    
    # syn rev potentials and decay times
    E_E=0, E_I=-.07, E_A=-.06, T_E=.004, T_I=.004, T_A=.006,
    
    W_U_E=0.26 * 0.004 * 0.25,
    W_MAX=0.26 * 0.004,
    M=9.,
    ETA=0.00005,
    EPSILON=0.05,
    
    W_E_I_R=0,
    W_I_E_R=0,
    
    N_EXC=50,
    N_INH=50,
    
    DRIVING_HZ=2.5, # 10 Hz lambda Poisson input to system
    N_DRIVING_CELLS=1,
    PROJECTION_NUM=5,
    W_INITIAL=0.,
    
    # OTHER INPUTS
    SGM_N=.5e-10,  # noise level (A*sqrt(s))
    I_EXT_B=0,  # additional baseline current input
)

## SMLN
S = Generic(RNG_SEED=0, DT=0.1e-3)

print('T_M_E =', 1000*M.C_M_E/M.G_L_E, 'ms')  # E cell membrane time constant (C_m/g_m)

def generate_chain_weight_mat(m):
    chain_order = np.arange(1, int(m.N_EXC / m.PROJECTION_NUM), dtype=int)
    chain_order = np.insert(chain_order, 0, 0)
    w_syn = np.zeros((m.N_EXC, m.N_EXC))
    for idx, layer_idx in enumerate(chain_order):
        layer_start = m.PROJECTION_NUM * layer_idx
        if idx + 1 < len(chain_order):
            next_layer_idx = chain_order[idx + 1]
            next_layer_start = m.PROJECTION_NUM * next_layer_idx

            w_syn[next_layer_start : next_layer_start + m.PROJECTION_NUM, layer_start : layer_start + m.PROJECTION_NUM] = np.ones((m.PROJECTION_NUM, m.PROJECTION_NUM))

        if idx + 2 < len(chain_order):
            skip_layer_idx = chain_order[idx + 2]
            skip_layer_start = m.PROJECTION_NUM * skip_layer_idx

            w_syn[skip_layer_start : skip_layer_start + m.PROJECTION_NUM, layer_start : layer_start + m.PROJECTION_NUM] = np.ones((m.PROJECTION_NUM, m.PROJECTION_NUM))
    w_syn *= m.W_MAX / (m.M)
    return w_syn

### RUN_TEST function

def run_test(m, output_dir_name, show_connectivity=True, repeats=1, n_show_only=None,
    add_noise=True, dropouts=[{'E': 0, 'I': 0}]):
    
    n_cells_driven = m.N_DRIVING_CELLS * m.PROJECTION_NUM
    
    w_u_e_drive = np.zeros(m.N_DRIVING_CELLS)
    w_u_e_drive[0] = m.W_U_E
    w_u_e = np.repeat(np.diag(w_u_e_drive), m.PROJECTION_NUM, axis=0)
    
    ## input weights
    w_u = {
        # localized inputs to trigger activation from start of chain
        'E': np.block([
            [ w_u_e, np.zeros([n_cells_driven, m.N_EXC + m.N_INH]) ],
            [ np.zeros([m.N_EXC + m.N_INH - n_cells_driven, m.N_EXC + m.N_INH + m.N_DRIVING_CELLS]) ],
        ]),

        'I': np.zeros((m.N_EXC + m.N_INH, m.N_DRIVING_CELLS + m.N_EXC + m.N_INH)),

        'A': np.zeros((m.N_EXC + m.N_INH, m.N_DRIVING_CELLS + m.N_EXC + m.N_INH)),
    }
    
    e_i_r = np.stack([rand_n_ones_in_vec_len_l(10, m.N_EXC) for i in range(m.N_INH)])
    
    i_e_r = np.stack([rand_n_ones_in_vec_len_l(20, m.N_INH) for i in range(m.N_EXC)])
    
    w_e_e_r = generate_chain_weight_mat(m)
    w_e_e_r[:m.N_EXC, :m.N_EXC] += m.RAND_WEIGHT_MAX * np.random.rand(m.N_EXC, m.N_EXC)
    np.fill_diagonal(w_e_e_r, 0.)

    connectivity = np.ones((m.N_EXC, m.N_EXC))
    
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
        'A': np.block([
            [ m.W_A * np.diag(np.ones((m.N_EXC))), np.zeros((m.N_EXC, m.N_INH)) ],
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
        spks_u_base = np.zeros((len(t), m.N_DRIVING_CELLS + m.N_EXC + m.N_INH), dtype=int)

                # trigger inputs
        activation_times = np.concatenate([np.random.poisson(m.DRIVING_HZ * S.DT, size=(len(t), 1)) for i in range(m.N_DRIVING_CELLS)], axis=1)
        spks_u = copy(spks_u_base)
        spks_u[:, :m.N_DRIVING_CELLS] = np.zeros((len(t), m.N_DRIVING_CELLS))
        burst_t = np.arange(0, 4 * int(m.BURST_T / S.DT), int(m.BURST_T / S.DT))

        for t_idx, driving_cell_idx in zip(*activation_times.nonzero()):
            try:
                spks_u[burst_t + t_idx, driving_cell_idx] = 1
            except IndexError as e:
                pass
        
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
                e_s={'E': M.E_E, 'I': M.E_I, 'A': M.E_A},
                t_s={'E': M.T_E, 'I': M.T_E, 'A': M.T_A},
                w_r=copy(w_r_for_dropout),
                w_u=copy(w_u),
                plasticity_indices=np.arange(m.N_EXC),
                connectivity=connectivity,
                W_max=m.W_MAX,
                m=m.M,
                eta=m.ETA,
                epsilon=m.EPSILON,
                dt=S.DT,
                gamma=m.GAMMA,
                alpha=m.ALPHA,
                fr_set_points=m.FR_SET_POINTS,
                output_freq=1000,
                homeo=True,
            )

            clamp = Generic(
                v={0: np.repeat(m.E_L_E, m.N_EXC + m.N_INH)}, spk={})

            # run smln
            rsp = ntwk.run(dt=S.DT, clamp=clamp, i_ext=i_ext,
                            output_dir_name=f'{output_dir_name}_{rp_idx}_{d_idx}', spks_u=spks_u,
                            dropouts=[(m.DROPOUT_TIME, dropouts[d_idx])],
                            )
            
            if show_connectivity:
                graph_weights(rsp.ntwk.w_r, rsp.ntwk.w_u, v_max=w_max)
            
            w_r_e = rsp.ntwk.w_r['E'][:m.N_EXC, :m.N_EXC]
            graph_weight_matrix(w_r_e, 'Exc->Exc Weights\n', v_max=w_max)
            graph_weight_matrix(np.sqrt(np.dot(w_r_e, w_r_e.T)), 'W * W.T \n', v_max=w_max)
                
            rsps_for_trial.append({
                'spks_t': copy(rsp.spks_t),
                'spks_c': copy(rsp.spks_c),
                'spks_u': spks_u.nonzero(),
                'w_r': copy(rsp.ntwk.w_r),
                'activation_times': activation_times.nonzero(),
            })
        all_rsps.append(rsps_for_trial)
    return all_rsps

def quick_plot(m, repeats=1, show_connectivity=True, n_show_only=None, add_noise=True, dropouts=[{'E': 0, 'I': 0}]):
    output_dir_name = f'{time_stamp(s=True)}'

    all_rsps = run_test(m, output_dir_name=output_dir_name, show_connectivity=show_connectivity,
                        repeats=repeats, n_show_only=n_show_only, add_noise=add_noise, dropouts=dropouts)

    output_dir = f'./figures/{output_dir_name}'
    os.makedirs(output_dir)
        
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
            fig.savefig(f'{output_dir}/{idx_r}_{idx_do}.png')

            activation_times = np.stack(rsp_for_dropout['activation_times'])
            driving_activation_times = activation_times[:, activation_times[1, :] == 0][0, :]

            midpoint = int(len(driving_activation_times) / 2)
            
            plot_params = {
                'early': driving_activation_times[:3],
                'late': driving_activation_times[-3:],
                'middle': driving_activation_times[midpoint:midpoint + 80],
            }

            for title, trigger_times in plot_params.items():
                for t_idx, trigger_time in enumerate(trigger_times):
                    gs = gridspec.GridSpec(1, 1)
                    fig = plt.figure(figsize=(10, 6), tight_layout=True)
                    axs = [fig.add_subplot(gs[0])]

                    t_window = (trigger_time * S.DT, trigger_time * S.DT + 30e-3)

                    exc_raster_for_win = exc_raster[:, (exc_raster[0, :] >= t_window[0]) & (exc_raster[0, :] < t_window[1])]
                    inh_raster_for_win = inh_raster[:, (inh_raster[0, :] >= t_window[0]) & (inh_raster[0, :] < t_window[1])]

                    axs[0].scatter(exc_raster_for_win[0, :] * 1000, exc_raster_for_win[1, :], s=1, c='black', zorder=0, alpha=1)
                    axs[0].scatter(inh_raster_for_win[0, :] * 1000, inh_raster_for_win[1, :], s=1, c='red', zorder=0, alpha=1)

                    axs[0].set_ylim(-1, m.N_EXC + m.N_INH)
                    axs[0].set_xlim(t_window[0] * 1000, t_window[1] * 1000)
                    axs[0].set_ylabel('Cell Index')
                    axs[0].set_xlabel('Time (ms)')

                    fig.savefig(f'{output_dir}/{title}_{idx_r}_{idx_do}_{t_idx}.png')
    return all_rsps

S.T = 4.
S.DT = 0.05e-3
m2 = copy(M)

m2.EPSILON = 0.
m2.ETA = 0 #0.0000015
m2.GAMMA = 0.

m2.W_A = 5e-5
m2.W_E_I_R = 4e-5
m2.W_I_E_R = 0.3e-5
m2.T_R_E = 1e-3
m2.W_MAX = 0.26 * 0.004 * .09
m2.W_U_E = 0.26 * 0.004 * .15
m2.M = 5

m2.FR_SET_POINTS = 3 * m2.DRIVING_HZ * S.DT * np.ones(m2.N_EXC)
m2.ALPHA = 0.5
m2.RAND_WEIGHT_MAX = m2.W_MAX / (m2.M * m2.N_EXC)
m2.DROPOUT_TIME = 150.

m2.BURST_T = 2e-3

all_rsps = quick_plot(m2, dropouts=[
    {'E': 0.0, 'I': 0},
    ])
