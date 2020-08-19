import os
import sys
from tempfile import mkdtemp
import numpy as np
from scipy.ndimage import gaussian_filter
from numba import njit, int8, int32, prange
from sigpyproc.Readers import FilReader
import matplotlib.pyplot as plt
from astropy import log
from astropy.table import Table
import tqdm



def quick_chan_rebin(counts, current_rebin):
    """
    Examples
    --------
    >>> counts = np.array([np.arange(0, 10), np.arange(2, 12),
    ...                    np.arange(1, 11), np.arange(3, 13),
    ...                    np.arange(1, 11), np.arange(3, 13)])
    >>> reb = quick_chan_rebin(counts, 2)
    >>> np.allclose(reb, [[2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
    ...                   [4, 6, 8, 10, 12, 14, 16, 18, 20, 22],
    ...                   [4, 6, 8, 10, 12, 14, 16, 18, 20, 22]])
    True
    """
    nchan = counts.shape[0]
    nbin = counts.shape[1]

    n = int(nchan // current_rebin)

    rebinned_counts = np.sum(
        counts[:n * current_rebin, :].reshape((n, current_rebin, nbin)), axis=1)
    return rebinned_counts


@njit
def quick_resample(counts, current_rebin):
    """
    Examples
    --------
    >>> counts = np.array([np.arange(1, 11), np.arange(3, 13)])
    >>> reb = quick_resample(counts, 2)
    >>> np.allclose(reb, [[ 3,  7, 11, 15, 19], [ 7, 11, 15, 19, 23]])
    True
    """
    nchan = counts.shape[0]
    nbin = counts.shape[1]

    n = int(nbin // current_rebin)

    reshaped = np.copy(counts[:, :n * current_rebin]).reshape((nchan, n, current_rebin))
    rebinned_counts = np.zeros((reshaped.shape[0], reshaped.shape[1]))
    for i in range(reshaped.shape[2]):
        rebinned_counts[:, :] += reshaped[:, :, i]
    return rebinned_counts


@njit
def roll_and_sum(array, sum_array, N):
    """
    Examples
    --------
    >>> array = np.arange(10)
    >>> sum_array = np.zeros(10)
    >>> np.allclose(roll_and_sum(array, sum_array, 3), np.roll(array, 3))
    True
    >>> # Check that it is in-place
    >>> sum_array is roll_and_sum(array, sum_array, 3)
    True
    """
    from_end = array.size - N
    idx = from_end
    for i in range(N):
        sum_array[i] += array[idx]
        idx += 1

    idx = N
    for i in range(from_end):
        sum_array[idx] += array[i]
        idx += 1
    return sum_array


@njit
def _dedisperse(data, shifts, sum_array):
    for i in range(data.shape[0]):
         roll_and_sum(data[i], sum_array, shifts[i])
    return sum_array


@njit
def dedisperse(data, shifts):
    N = data.shape[1]
    sum_array = np.zeros(N)
    sh = normalize_shifts(-shifts, N)
    return _dedisperse(data, sh, sum_array)


@njit
def normalize_shifts(shifts, N):
    """
    Examples
    --------
    >>> a = np.array([-1, 0, 2, 4])
    >>> b = normalize_shifts(a, 3)
    >>> np.all(b == np.array([2, 0, 2, 1]))
    True
    """
    new_shifts = np.zeros(shifts.size, dtype=np.int32)

    for i in range(shifts.size):
        shift = shifts[i]
        shift = np.rint(shift)
        while shift < 0:
            shift += N
        while shift >= N:
            shift -= N
        new_shifts[i] = shift

    return new_shifts


@njit
def dedispersion_shifts(nchan, dm, start_freq, bandwidth, sample_time):
    dfreq = bandwidth / nchan
    stop_freq = start_freq + bandwidth
    center_freq = (stop_freq + start_freq) / 2
    ref_delay = 4149 * dm * center_freq**(-2)

    shifts = np.zeros(nchan)
    for i in range(nchan):
        # DM delay
        chan_freq = start_freq + i * dfreq
        delay = 4149 * dm * chan_freq ** (-2) - ref_delay
        shifts[i] = int(np.rint(delay // sample_time))

    return shifts


@njit
def delta_delay(dm, start_freq, stop_freq):
    delay1 = 4149. * dm * start_freq ** (-2)
    delay2 = 4149. * dm * stop_freq ** (-2)
    return delay1 - delay2


@njit
def dedispersion_plan(nchan, dmmin, dmmax, start_freq, bandwidth, sample_time):
    """
    Examples
    --------
    >>> tDM = dedispersion_plan(10, 0, 10, 1400, 128, 0.0005)
    >>> np.isclose(tDM[0], 0)
    True
    >>> np.isclose(tDM[-1], 10., atol=1)
    True
    """
    # dfreq = bandwidth / nchan
    stop_freq = start_freq + bandwidth
    f0 = np.float(start_freq)
    f1 = np.float(stop_freq)

    max_N = delta_delay(np.float(dmmax), f0, f1) / sample_time
    min_N = delta_delay(np.float(dmmin), f0, f1) / sample_time

    trial_N = np.arange(min_N, max_N + 1)
    trial_DM = trial_N * sample_time / 4149. * (f0 ** (-2) - f1 ** (-2)) ** (-1)
    # print(trial_DM)
    return trial_DM


@njit(parallel=True)
def _dedispersion_search(data, trial_DMs, nchan, start_freq, bandwidth, sample_time):
    ndm = trial_DMs.size
    maxvalues = np.zeros_like(trial_DMs)
    stds = np.zeros_like(trial_DMs)
    best_snrs = np.zeros_like(trial_DMs)
    best_windows = np.zeros(ndm, int32)
    for i in prange(ndm):
        dm = trial_DMs[i]
        shifts = dedispersion_shifts(nchan, dm, start_freq, bandwidth, sample_time)
        dedisp = dedisperse(data, shifts)

        dedisp_shift = (dedisp - np.mean(dedisp)).reshape((1, dedisp.size))

        best_snr = 0
        best_win = 0
        for window_pow in range(0, 4):
            window = 1 << window_pow  # 2**window_pow
            reb = quick_resample(dedisp_shift, window)
            snr = np.max(reb) / np.std(reb)
            if snr > best_snr:
                best_snr = snr
                best_win = window

        maxvalues[i] = np.max(dedisp_shift)
        stds[i] = np.std(dedisp_shift)
        best_windows[i] = best_win
        best_snrs[i] = best_snr
    return maxvalues, stds, best_snrs, best_windows


def dedispersion_search(data, dmmin, dmmax, start_freq, bandwidth, sample_time, show=False):
    # log.info("Starting search with Dedispersion")
    nchan = data.shape[0]

    # log.info("Creating dedispersion plan")
    trial_DMs = dedispersion_plan(nchan, dmmin, dmmax, start_freq, bandwidth, sample_time)

    # log.info("Allocating dedispersion plane")

    if show:
        tempfile = os.path.join(mkdtemp(), 'dummy.npy')
        dedispersed_plane = \
            np.memmap(tempfile, dtype=np.float, mode='w+',
                      shape=(trial_DMs.size, data.shape[1]))
        maxvalues = np.zeros_like(trial_DMs)
        stds = np.zeros_like(trial_DMs)
        best_snrs = np.zeros_like(trial_DMs)
        best_windows = np.zeros_like(trial_DMs, dtype=int)
        for i, dm in enumerate(trial_DMs):
            shifts = dedispersion_shifts(nchan, dm, start_freq, bandwidth, sample_time)
            # print(i, dm, dedispersed_plane.shape, shifts)
            dedisp = dedisperse(data, shifts)
            dedispersed_plane[i, :] = dedisp

            dedisp_shift = (dedisp - np.mean(dedisp)).reshape((1, dedisp.size))
            best_snr = 0
            best_win = 0
            for window_pow in range(0, 4):
                window = 1 << window_pow
                reb = quick_resample(dedisp_shift, window)
                snr = np.max(reb) / np.std(reb)
                if snr > best_snr:
                    best_snr = snr
                    best_win = window

            maxvalues[i] = np.max(dedisp_shift)
            stds[i] = np.std(dedisp_shift)
            best_windows[i] = best_win
            best_snrs[i] = best_snr
    else:
        maxvalues, stds, best_snrs, best_windows = \
            _dedispersion_search(data, trial_DMs, nchan, start_freq, bandwidth, sample_time)

    table = Table({'DM': trial_DMs, 'max': maxvalues, 'std': stds, 'snr': best_snrs, 'rebin': best_windows})
    if show:
        return table, dedispersed_plane
    return table


def apply_dm_shifts_to_data(data, shifts):
    new_data = np.copy(data)
    for i, d in enumerate(data):
        new_data[i] = np.roll(data[i], -np.rint(shifts[i]).astype(int))
    return new_data
