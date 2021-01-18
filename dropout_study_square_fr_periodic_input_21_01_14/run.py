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
from functools import reduce
import argparse

from aux import *
from disp import *
from ntwk import LIFNtwkG
from utils.general import *
from utils.file_io import *

cc = np.concatenate

parser = argparse.ArgumentParser()
parser.add_argument('--title', metavar='T', type=str, nargs=1)
parser.add_argument('--fr_scale', metavar='F', type=float, nargs=1)
args = parser.parse_args()

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
        w_syn[layer_start : layer_start + m.PROJECTION_NUM, layer_start : layer_start + m.PROJECTION_NUM] = rand_per_row_mat(m.CON_PER_ROW, (m.PROJECTION_NUM, m.PROJECTION_NUM))
        if idx + 1 < len(chain_order):
            next_layer_idx = chain_order[idx + 1]
            next_layer_start = m.PROJECTION_NUM * next_layer_idx

            w_syn[next_layer_start : next_layer_start + m.PROJECTION_NUM, layer_start : layer_start + m.PROJECTION_NUM] = rand_per_row_mat(m.CON_PER_ROW, (m.PROJECTION_NUM, m.PROJECTION_NUM))

    w_syn *= m.W_MAX / m.M
    return w_syn

### RUN_TEST function

def run_test(m, output_dir_name, show_connectivity=True, repeats=1, n_show_only=None,
    add_noise=True, dropouts=[{'E': 0, 'I': 0}], w_r_e=None, w_r_i=None):
    
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

    connectivity = np.ones((m.N_EXC, m.N_EXC))

    if w_r_e is None:
        w_e_e_r = generate_chain_weight_mat(m)
        w_e_e_r += m.RAND_WEIGHT_MAX * np.random.rand(m.N_EXC, m.N_EXC)

        np.fill_diagonal(w_e_e_r, 0.)

        e_i_r = np.stack([rand_n_ones_in_vec_len_l(3, m.N_EXC) for i in range(m.N_INH)])

        w_r_e = np.block([
            [ w_e_e_r, np.zeros((m.N_EXC, m.N_INH)) ],
            [ e_i_r * m.W_E_I_R,  np.zeros((m.N_INH, m.N_INH)) ],
        ])

    if w_r_i is None:
        i_e_r = np.stack([rand_n_ones_in_vec_len_l(40, m.N_INH) for i in range(m.N_EXC)])
        
        w_r_i = np.block([
            [ np.zeros((m.N_EXC, m.N_EXC)), i_e_r * m.W_I_E_R ],
            [ np.zeros((m.N_INH, m.N_EXC)), np.zeros((m.N_INH, m.N_INH)) ],
        ])
    
    ## recurrent weights
    w_r_base = {
        'E': w_r_e,
        'I': w_r_i,
        'A': np.block([
            [ m.W_A * np.diag(np.ones((m.N_EXC))), np.zeros((m.N_EXC, m.N_INH)) ],
            [ np.zeros((m.N_INH, m.N_EXC)), np.zeros((m.N_INH, m.N_INH)) ],
        ]),
    }
    
    w_r_for_dropouts = []
    for dropout in dropouts:
        w_r = copy(w_r_base)
        
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
        activation_times = np.zeros((len(t), m.N_DRIVING_CELLS))
        for t_ctr in np.arange(0, S.T, 1./m.DRIVING_HZ):
            activation_times[int(t_ctr/S.DT), :] = 1

        np.concatenate([np.random.poisson(m.DRIVING_HZ * S.DT, size=(len(t), 1)) for i in range(m.N_DRIVING_CELLS)], axis=1)
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
                fr_baseline=m.FR_BASELINE,
                stdp_scale=m.STDP_SCALE,
                output_freq=1000,
                homeo=True,
                weight_update=True,
            )

            clamp = Generic(v={0: np.repeat(m.E_L_E, m.N_EXC + m.N_INH)}, spk={})

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

def quick_plot(m, run_title='', w_r_e=None, w_r_i=None, repeats=1, show_connectivity=True, n_show_only=None, add_noise=True, dropouts=[{'E': 0, 'I': 0}]):
    output_dir_name = f'{run_title}_{time_stamp(s=True)}:{zero_pad(int(np.random.rand() * 9999), 4)}'

    all_rsps = run_test(m, output_dir_name=output_dir_name, show_connectivity=show_connectivity,
                        repeats=repeats, n_show_only=n_show_only, add_noise=add_noise, dropouts=dropouts,
                        w_r_e=w_r_e, w_r_i=w_r_i)

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

def process_single_activation(exc_raster, m):
    # extract first spikes
    first_spk_times = -1 * np.ones(m.N_EXC)
    for i in range(exc_raster.shape[1]):
        nrn_idx = exc_raster[1, i]
        if first_spk_times[nrn_idx] < 0:
            first_spk_times[nrn_idx] = exc_raster[0, i]

    # extract spks per neuron
    spks_per_nrn = bin_occurences(exc_raster[1, :])
    return first_spk_times, spks_per_nrn

S.T = 230.
S.DT = 0.05e-3
m2 = copy(M)

m2.EPSILON = 0. # deprecated
m2.ETA = 0.000001
m2.GAMMA = 0. # deprecated

m2.W_A = 5e-4 
m2.W_E_I_R = 10e-5
m2.W_I_E_R = 0.2e-5
m2.T_R_E = 1e-3
m2.W_MAX = 0.26 * 0.004 * .33
m2.W_U_E = 0.26 * 0.004 * .15
m2.M = 5

m2.ALPHA = 1.5e-3
m2.FR_BASELINE = args.fr_scale[0]
m2.STDP_SCALE = 0.00001

m2.RAND_WEIGHT_MAX = m2.W_MAX / (m2.M * m2.N_EXC)
m2.DROPOUT_TIME = 30.

m2.BURST_T = 2e-3

m2.CON_PER_ROW = 2

def load_weight_matrices(direc, num):
    file_names = sorted(all_files_from_dir(direc))
    file = file_names[num]
    loaded = sio.loadmat(os.path.join(direc, file))
    return loaded['w_r_e'], loaded['w_r_i']

w_r_e, w_r_i = load_weight_matrices('./data/2021-01-12--09:20--27:2015_0_0', -1)

fr_set_points = [4.]

for fr_sp in fr_set_points:
    m3 = copy(m2)
    m3.FR_SET_POINTS = fr_sp
    all_rsps = quick_plot(m3, w_r_e=w_r_e, w_r_i=w_r_i, run_title=args.title[0], dropouts=[
        {'E': 0.3, 'I': 0},
        ])