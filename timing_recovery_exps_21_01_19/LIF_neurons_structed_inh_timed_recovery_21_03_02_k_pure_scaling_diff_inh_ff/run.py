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

from aux import *
from disp import *
from ntwk import LIFNtwkG
from utils.general import *
from utils.file_io import *

cc = np.concatenate

parser = argparse.ArgumentParser()
parser.add_argument('--title', metavar='T', type=str, nargs=1)
parser.add_argument('--w_a', metavar='A', type=float, nargs=1)
parser.add_argument('--fr_penalty', metavar='P', type=float, nargs=1)
parser.add_argument('--stdp_scale', metavar='S', type=float, nargs=1)
parser.add_argument('--beta', metavar='S', type=float, nargs=1)

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
    E_E=0, E_I=-.07, E_A=-.07, T_E=.004, T_I=.004, T_A=.006,
    
    N_EXC=1200,
    N_INH=1200,
    
    DRIVING_HZ=5., # 2 Hz lambda Poisson input to system
    N_DRIVING_CELLS=1,
    PROJECTION_NUM=20,
    
    # OTHER INPUTS
    SGM_N=.5e-10,  # noise level (A*sqrt(s))
    I_EXT_B=0,  # additional baseline current input
)

## SMLN
S = Generic(RNG_SEED=0, DT=0.1e-3)

print('T_M_E =', 1000*M.C_M_E/M.G_L_E, 'ms')  # E cell membrane time constant (C_m/g_m)


def generate_ff_chain(size, unit_size, unit_funcs, ff_deg=[0, 1], tempering=[1., 1.]):
    if size % unit_size != 0:
        raise ValueError('unit_size does not evenly divide size')

    if len(ff_deg) != len(tempering):
        raise ValueError('ff_deg and tempering must be the same length')

    n_units = int(size / unit_size)
    chain_order = np.arange(0, n_units, dtype=int)
    mat = np.zeros((size, size))

    for idx in chain_order:
        layer_start = unit_size * idx
        for j, ff_idx in enumerate(ff_deg):
            if idx + ff_idx < len(chain_order) and idx + ff_idx >= 0:
                ff_layer_idx = chain_order[idx + ff_idx]
                ff_layer_start = unit_size * ff_layer_idx

                mat[ff_layer_start : ff_layer_start + unit_size, layer_start : layer_start + unit_size] = unit_funcs[j]() * tempering[j]
    return mat

def generate_exc_ff_chain(m):
    def rec_unit_func():
        return mat_1_if_under_val(m.CON_PROB_R, (m.PROJECTION_NUM, m.PROJECTION_NUM))

    def ff_unit_func():
        return mat_1_if_under_val(m.CON_PROB_FF, (m.PROJECTION_NUM, m.PROJECTION_NUM))

    unit_funcs = [rec_unit_func, ff_unit_func]

    return m.W_INITIAL / m.PROJECTION_NUM * generate_ff_chain(m.N_EXC, m.PROJECTION_NUM, unit_funcs)

def generate_local_con(m, ff_deg=[0, 1, 2]):
    def unit_func():
        return np.random.rand(m.PROJECTION_NUM, m.PROJECTION_NUM)

    unit_funcs = [unit_func] * len(ff_deg)

    return m.RAND_WEIGHT_MAX * generate_ff_chain(m.N_EXC, m.PROJECTION_NUM, unit_funcs, ff_deg=ff_deg, tempering=[1.] * len(ff_deg))

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

    def solid_unit_func():
        return np.ones((m.PROJECTION_NUM, m.PROJECTION_NUM))

    def rand_unit_func():
        return np.random.rand(m.PROJECTION_NUM, m.PROJECTION_NUM)

    if w_r_e is None:
        w_e_e_r = generate_exc_ff_chain(m)

        np.fill_diagonal(w_e_e_r, 0.)

        con_per_i = m.E_I_CON_PER_LINK * m.N_EXC / m.PROJECTION_NUM
        print(con_per_i)
        e_i_r = rand_per_row_mat(int(con_per_i), (m.N_INH, m.N_EXC))

        # e_i_r += np.where(np.random.rand(m.N_EXC, m.N_INH) > (1. - 0.05 * 150. / m.N_INH), np.random.rand(m.N_EXC, m.N_INH), 0.)

        w_r_e = np.block([
            [ w_e_e_r, np.zeros((m.N_EXC, m.N_INH)) ],
            [ e_i_r * m.W_E_I_R,  np.zeros((m.N_INH, m.N_INH)) ],
        ])

    if w_r_i is None:

        i_e_r = np.ones((m.N_EXC, m.N_INH))

        w_r_i = np.block([
            [ np.zeros((m.N_EXC, m.N_EXC)), i_e_r * 0.2e-5 ],
            [ np.zeros((m.N_INH, m.N_EXC)), np.zeros((m.N_INH, m.N_INH)) ],
        ])
    
    ## recurrent weights
    w_r = {
        'E': w_r_e,
        'I': w_r_i,
        'A': np.block([
            [ m.W_A * np.diag(np.ones((m.N_EXC))), np.zeros((m.N_EXC, m.N_INH)) ],
            [ np.zeros((m.N_INH, m.N_EXC)), np.zeros((m.N_INH, m.N_INH)) ],
        ]),
    }
    
    all_rsps = []

    # run simulation for same set of parameters
    for rp_idx in range(repeats):
        show_trial = (type(n_show_only) is int and rp_idx < n_show_only)

        rsps_for_trial = []

        inputs = []

        for sim_len in [S.T1, S.T2]:
            # generate timesteps and initial excitatory input window
            t = np.arange(0, sim_len, S.DT)

            ## external currents
            if add_noise:
                i_ext = m.SGM_N/S.DT * np.random.randn(len(t), m.N_EXC + m.N_INH) + m.I_EXT_B
            else:
                i_ext = m.I_EXT_B * np.ones((len(t), m.N_EXC + m.N_INH))

            ## inp spks
            spks_u_base = np.zeros((len(t), m.N_DRIVING_CELLS + m.N_EXC + m.N_INH), dtype=int)

            # trigger inputs
            activation_times = np.zeros((len(t), m.N_DRIVING_CELLS))
            for t_ctr in np.arange(0, sim_len, 1./m.DRIVING_HZ):
                activation_times[int(t_ctr/S.DT), :] = 1

            np.concatenate([np.random.poisson(m.DRIVING_HZ * S.DT, size=(len(t), 1)) for i in range(m.N_DRIVING_CELLS)], axis=1)
            spks_u = copy(spks_u_base)
            spks_u[:, :m.N_DRIVING_CELLS] = np.zeros((len(t), m.N_DRIVING_CELLS))
            burst_t = np.arange(0, 4 * int(m.BURST_T / S.DT), int(m.BURST_T / S.DT))

            for t_idx, driving_cell_idx in zip(*activation_times.nonzero()):
                input_noise_t = np.array(np.random.normal(scale=m.INPUT_STD / S.DT), dtype=int)
                try:
                    spks_u[burst_t + t_idx + input_noise_t, driving_cell_idx] = 1
                except IndexError as e:
                    pass

            inputs.append({'i_ext': copy(i_ext), 'spks_u': copy(spks_u), 'activation_times': copy(activation_times)})
        
        for d_idx, dropout in enumerate(dropouts):

            ntwk = LIFNtwkG(
                c_m=m.C_M_E,
                g_l=m.G_L_E,
                e_l=m.E_L_E,
                v_th=m.V_TH_E,
                v_r=m.E_R_E,
                t_r=m.T_R_E,
                e_s={'E': M.E_E, 'I': M.E_I, 'A': M.E_A},
                t_s={'E': M.T_E, 'I': M.T_E, 'A': M.T_A},
                w_r=copy(w_r),
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
                stdp_scale=m.STDP_SCALE,
                beta=m.BETA,
                output_freq=100000,
                output=False,
                homeo=True,
                weight_update=False,
            )

            clamp = Generic(v={0: np.repeat(m.E_L_E, m.N_EXC + m.N_INH)}, spk={})

            # run smln
            sim_num = 0
            rsp = ntwk.run(dt=S.DT, clamp=clamp, i_ext=inputs[sim_num]['i_ext'], output_dir_name='',
                            spks_u=inputs[sim_num]['spks_u'], dropouts=[], m=m, repairs=[],
                            )

            ## process first run to calculate intereuron spike times

            raster = np.stack([rsp.spks_t, rsp.spks_c])
            exc_raster = raster[:, raster[1, :] < m.N_EXC]
            inh_raster = raster[:, raster[1, :] >= m.N_EXC]
            activation_times_nonzero = np.stack(activation_times.nonzero())
            driving_activation_times = activation_times_nonzero[:, activation_times_nonzero[1, :] == 0][0, :]

            # grab just the first activation to use as a timing template
            trigger_time = driving_activation_times[0]
            t_start_env = (trigger_time * S.DT - 5e-3, trigger_time * S.DT + 5e-3)
            nrn_1_spk_timing_for_activation = exc_raster[0, (exc_raster[0, :] >= t_start_env[0]) & (exc_raster[0, :] < t_start_env[1])]
            prop_start = nrn_1_spk_timing_for_activation[0]
            t_window = (prop_start, prop_start + 110e-3)

            # get exc and inh rasters for the first activation
            exc_raster_for_win = exc_raster[:, (exc_raster[0, :] >= t_window[0]) & (exc_raster[0, :] < t_window[1])]
            inh_raster_for_win = inh_raster[:, (inh_raster[0, :] >= t_window[0]) & (inh_raster[0, :] < t_window[1])]

            # extract the first spike times of the exc cells
            first_spk_times = process_single_activation(exc_raster_for_win, m)
            first_spk_times -= first_spk_times[0]

            # now, redesign our I --> E connections to reflect an anti-Hebbian STDP rule
            tau_pocket_end = 3e-3 # 3 ms
            tau_pocket_start = -1e-3
            inh_cell_to_pocket_exc_cell = np.zeros((m.N_EXC, m.N_INH))

            for i_spk_idx in range(inh_raster.shape[1]):
                inh_spk_time = inh_raster[0, i_spk_idx]
                inh_spk_cell_idx = inh_raster[1, i_spk_idx]
                for exc_cell_idx, first_spk_time in enumerate(first_spk_times):
                    if first_spk_time - inh_spk_time < tau_pocket_end and first_spk_time - inh_spk_time > tau_pocket_start:
                        inh_cell_to_pocket_exc_cell[exc_cell_idx, int(inh_spk_cell_idx) - m.N_EXC] = 1

            i_e_r = (1 - inh_cell_to_pocket_exc_cell)

            w_r_i_remodeled = np.block([
                [ np.zeros((m.N_EXC, m.N_EXC)), i_e_r * m.W_I_E_R ],
                [ np.zeros((m.N_INH, m.N_EXC)), np.zeros((m.N_INH, m.N_INH)) ],
            ])

            w_r_remodeled = copy(w_r)
            w_r_remodeled['I'] = w_r_i_remodeled
            # w_r_remodeled['E'][:m.N_EXC, :m.N_EXC] *= 1.2

            ntwk = LIFNtwkG(
                c_m=m.C_M_E,
                g_l=m.G_L_E,
                e_l=m.E_L_E,
                v_th=m.V_TH_E,
                v_r=m.E_R_E,
                t_r=m.T_R_E,
                e_s={'E': M.E_E, 'I': M.E_I, 'A': M.E_A},
                t_s={'E': M.T_E, 'I': M.T_E, 'A': M.T_A},
                w_r=copy(w_r_remodeled),
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
                stdp_scale=m.STDP_SCALE,
                beta=m.BETA,
                output_freq=100000,
                homeo=True,
                weight_update=False,
            )

            clamp = Generic(v={0: np.repeat(m.E_L_E, m.N_EXC + m.N_INH)}, spk={})

            # run smln
            sim_num = 1
            rsp = ntwk.run(dt=S.DT, clamp=clamp, i_ext=inputs[sim_num]['i_ext'],
                            output_dir_name=f'{output_dir_name}_{rp_idx}_{d_idx}', spks_u=inputs[sim_num]['spks_u'],
                            dropouts=[(m.DROPOUT_TIME, dropouts[d_idx])], m=m, repairs=m.REPAIRS,
                            )
                
            rsps_for_trial.append({
                'spks_t': copy(rsp.spks_t),
                'spks_c': copy(rsp.spks_c),
                'spks_u': inputs[sim_num]['spks_u'].nonzero(),
                'w_r': copy(rsp.ntwk.w_r),
                'activation_times': inputs[sim_num]['activation_times'].nonzero(),
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

    robustness_output_dir = f'./robustness/{output_dir_name}'
    os.makedirs(robustness_output_dir)
        
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
            axs[0].set_xlim(0, S.T1 * 1000)
            axs[0].set_ylabel('Cell Index')
            axs[0].set_xlabel('Time (ms)')
            
            set_font_size(axs[0], 14)
            fig.savefig(f'{output_dir}/{idx_r}_{idx_do}.png')

            activation_times = np.stack(rsp_for_dropout['activation_times'])
            driving_activation_times = activation_times[:, activation_times[1, :] == 0][0, :]

            plot_params = {
                'early': driving_activation_times[:20],
                'middle': driving_activation_times[driving_activation_times > int(m.DROPOUT_TIME / S.DT)][:20],
                'late': driving_activation_times[-20:],
            }

            spk_timings = {
                'early': [],
                'middle': [],
                'late': [],
            }

            for title, trigger_times in plot_params.items():
                for t_idx, trigger_time in enumerate(trigger_times):
                    gs = gridspec.GridSpec(1, 1)
                    fig = plt.figure(figsize=(10, 6), tight_layout=True)
                    axs = [fig.add_subplot(gs[0])]

                    t_start_env = (trigger_time * S.DT - 5e-3, trigger_time * S.DT + 5e-3)

                    nrn_1_spk_timing_for_activation = exc_raster[0, (exc_raster[0, :] >= t_start_env[0]) & (exc_raster[0, :] < t_start_env[1])]

                    prop_start = nrn_1_spk_timing_for_activation[0]

                    t_window = (prop_start, prop_start + 110e-3)

                    exc_raster_for_win = exc_raster[:, (exc_raster[0, :] >= t_window[0]) & (exc_raster[0, :] < t_window[1])]
                    inh_raster_for_win = inh_raster[:, (inh_raster[0, :] >= t_window[0]) & (inh_raster[0, :] < t_window[1])]

                    first_spk_times = process_single_activation(exc_raster_for_win, m)
                    first_spk_times -= first_spk_times[0]

                    spk_timings[title].append(first_spk_times)

                    cell_order = np.arange(m.N_EXC)[~np.isnan(first_spk_times)]
                    trimmed_first_spk_times = first_spk_times[~np.isnan(first_spk_times)]

                    def line(x, A):
                        return A * x

                    try:
                        popt, pcov = curve_fit(line, trimmed_first_spk_times, cell_order)
                    except ValueError as e:
                        breakpoint()
                    success = len(trimmed_first_spk_times) >= (0.95 * m.N_EXC)

                    print(len(trimmed_first_spk_times) >= (0.95 * m.N_EXC))
                    print(popt[0])

                    sio.savemat(robustness_output_dir + '/' + f'title_{title}_dropout_{idx_do}_tidx_{t_idx}', {
                        'prop': popt[0],
                        'success': success,
                        'first_spk_times': first_spk_times,
                    })

                    axs[0].scatter(exc_raster_for_win[0, :] * 1000, exc_raster_for_win[1, :], s=1, c='black', zorder=0, alpha=1)
                    axs[0].scatter(inh_raster_for_win[0, :] * 1000, inh_raster_for_win[1, :], s=1, c='red', zorder=0, alpha=1)

                    axs[0].plot(trimmed_first_spk_times, line(trimmed_first_spk_times, popt[0]), lw=0.5, c='blue', zorder=1)

                    axs[0].set_ylim(-1, m.N_EXC + m.N_INH)
                    axs[0].set_xlim(t_window[0] * 1000, t_window[1] * 1000)
                    axs[0].set_ylabel('Cell Index')
                    axs[0].set_xlabel('Time (ms)')

                    fig.savefig(f'{output_dir}/{title}_{idx_r}_{idx_do}_{t_idx}.png')

            n_activations_to_compare = 10
            n_activations_to_compare = np.min([n_activations_to_compare, len(spk_timings['early']), len(spk_timings['late'])])

            timing_diffs = np.ones((n_activations_to_compare, m.N_EXC))

            for i in range(n_activations_to_compare):
                early_timings = spk_timings['early'][i]
                late_timings = spk_timings['late'][-1 * i]
                timing_diffs[i, :] = late_timings - early_timings

            gs = gridspec.GridSpec(1, 1)
            fig = plt.figure(figsize=(10, 6), tight_layout=True)
            axs = [fig.add_subplot(gs[0])]

            mean_diff = np.nanmean(timing_diffs, axis=0) * 1000
            std_diff = np.nanstd(timing_diffs, axis=0) * 1000

            axs[0].fill_between(np.arange(m.N_EXC), mean_diff + std_diff, mean_diff - std_diff, color='red', alpha=0.2)
            axs[0].plot(np.arange(m.N_EXC), mean_diff, lw=0.7, color='red')

            axs[0].set_xlabel('Cell Index')
            axs[0].set_ylabel('Timing Difference (ms)')

            fig.savefig(f'{output_dir}/timing_diffs.png')

    return all_rsps

def process_single_activation(exc_raster, m):
    # extract first spikes
    first_spk_times = np.nan * np.ones(m.N_EXC)
    for i in range(exc_raster.shape[1]):
        nrn_idx = int(exc_raster[1, i])
        if np.isnan(first_spk_times[nrn_idx]):
            first_spk_times[nrn_idx] = exc_raster[0, i]
    return first_spk_times

S.T1 = 0.2
S.T2 = 0.2 * (1 + 1 + 9.99)
S.DT = 0.2e-3
m2 = copy(M)

m2.EPSILON = 0. # deprecated
m2.ETA = 0.000001
m2.GAMMA = 0. # deprecated

m2.W_A = args.w_a[0] # 5e-4 
m2.W_E_I_R = 2e-5
m2.W_I_E_R = 0.3e-5 # 0.5e-5
m2.T_R_E = 1e-3
m2.W_MAX = 0.26 * 0.004 * 10
m2.W_INITIAL = 0.26 * 0.004 * 1.3
m2.W_U_E = 0.26 * 0.004 * .35
m2.M = 20

m2.ALPHA = args.fr_penalty[0] # 1.5e-3
m2.STDP_SCALE = args.stdp_scale[0] # 0.00001
m2.BETA = args.beta[0]
m2.FR_SET_POINTS = 4. * m2.DRIVING_HZ * S.DT

m2.RAND_WEIGHT_MAX = m2.W_INITIAL / (m2.M * m2.N_EXC)
m2.DROPOUT_TIME = 0.195

n_repairs = 10
repair_times = 0.200 * np.arange(n_repairs) + 0.395
repair_setpoints = 0.1 * np.arange(n_repairs) + 1.0
m2.REPAIRS = [(rep_time, repair_setpoints) for rep_time, repair_setpoints in zip(repair_times, repair_setpoints)]

m2.BURST_T = 2e-3
m2.CON_PROB_FF = .5
m2.CON_PROB_R = 0.
m2.E_I_CON_PER_LINK = 0.2

m2.INPUT_STD = 1e-3

# c_m = np.zeros((300))
# c_m[:150] = m2.C_M_E
# c_m[150:] = m2.C_M_E * np.random.rand(150) + 0.75 * m2.C_M_E
# m2.C_M_E = c_m

def load_weight_matrices(direc, num):
    file_names = sorted(all_files_from_dir(direc))
    file = file_names[num]
    loaded = sio.loadmat(os.path.join(direc, file))
    return loaded['w_r_e'], loaded['w_r_i']

def clip(f):
    f_str = str(f)
    f_str = f_str[:(f_str.find('.') + 2)]
    return f_str

print(m2.W_E_I_R * 1e5)

title = f'noise_ff_{clip(m2.W_INITIAL / (0.26 * 0.004))}_pf_{clip(m2.CON_PROB_FF)}_pr_{clip(m2.CON_PROB_R)}_eir_{clip(m2.W_E_I_R * 1e5)}_ier_{clip(m2.W_I_E_R * 1e5)}_dropout_sweep'

for i in range(10):
    all_rsps = quick_plot(m2, run_title=title, dropouts=[
        {'E': 0.5, 'I': 0},
    ])