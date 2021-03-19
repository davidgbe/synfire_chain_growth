import numpy as np
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import scipy.io as sio

def process_activation_and_graph(raster, m, t_start, t_end, activation_num, robustness_output_dir, figures_output_dir,
                        t_bound_upper_graph=200e-3, success_thresh=0.85):
    gs = gridspec.GridSpec(1, 1)
    fig = plt.figure(figsize=(10, 6), tight_layout=True)
    axs = [fig.add_subplot(gs[0])]

    first_spk_times, exc_raster_for_win, inh_raster_for_win, t_window = process_single_activation(raster, m, t_start, t_bound_upper_graph)

    cell_order = np.arange(m.N_EXC)[~np.isnan(first_spk_times)]
    trimmed_first_spk_times = first_spk_times[~np.isnan(first_spk_times)]

    def line(x, A):
        return A * x
    try:
        popt, pcov = curve_fit(line, trimmed_first_spk_times, cell_order)
    except ValueError as e:
        pass
    success = len(trimmed_first_spk_times) >= (success_thresh * m.N_EXC)

    print(success)
    print(popt[0])

    sio.savemat(robustness_output_dir + '/' + f'tidx_{activation_num}', {
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

    fig.savefig(f'{figures_output_dir}/{activation_num}.png')

def process_single_activation(raster, m, t_start, t_bound_upper_graph):
    exc_raster = raster[:, raster[1, :] < m.N_EXC]
    inh_raster = raster[:, raster[1, :] >= m.N_EXC]

    nrn_1_spk_timing_for_activation = exc_raster[0, (exc_raster[0, :] >= t_start) & (exc_raster[0, :] < (t_start + 10e-3))]

    prop_start = nrn_1_spk_timing_for_activation[0]

    t_window = (prop_start, prop_start + t_bound_upper_graph)

    exc_raster_for_win = exc_raster[:, (exc_raster[0, :] >= t_window[0]) & (exc_raster[0, :] < t_window[1])]
    inh_raster_for_win = inh_raster[:, (inh_raster[0, :] >= t_window[0]) & (inh_raster[0, :] < t_window[1])]

    first_spk_times = get_first_spk_times(exc_raster_for_win, m)
    first_spk_times -= np.nanmean(first_spk_times[:m.PROJECTION_NUM])

    return first_spk_times, exc_raster_for_win, inh_raster_for_win, t_window

def get_first_spk_times(exc_raster, m):
    # extract first spikes
    first_spk_times = np.nan * np.ones(m.N_EXC)
    for i in range(exc_raster.shape[1]):
        nrn_idx = int(exc_raster[1, i])
        if np.isnan(first_spk_times[nrn_idx]):
            first_spk_times[nrn_idx] = exc_raster[0, i]
    return first_spk_times