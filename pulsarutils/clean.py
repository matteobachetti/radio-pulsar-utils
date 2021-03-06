import os
import time
import pickle
from dataclasses import dataclass

import matplotlib as mpl
# mpl.use('Agg')

import numpy as np
import tqdm
import matplotlib.pyplot as plt

from scipy.signal import medfilt, savgol_filter
from scipy.ndimage import uniform_filter1d, gaussian_filter, median_filter
from astropy import log
from astropy.table import Table

from sigpyproc.Readers import FilReader

from hendrics.efsearch import h_test
from .dedispersion import dedispersion_plan, dedisperse, dedispersion_shifts, delta_delay
from .dedispersion import dedispersion_search as fast_dedispersion_search
from .dedispersion import quick_resample, quick_chan_rebin, apply_dm_shifts_to_data
from .stats import mad, ref_mad, get_bad_chans


@dataclass
class PulseInfo():
    nbin = 0
    nchan = 0
    ph0 = None
    amp = None
    width = None
    noise_level = None
    dm = None
    pulse_freq = None
    start_freq = None
    bandwidth = None
    dedisp_profile = None
    allprofs = None
    disp_profile = None

    disp_z2 = None
    disp_z6 = None
    disp_z12 = None
    disp_z20 = None
    disp_H = None
    disp_M = None

    dedisp_z2 = None
    dedisp_z6 = None
    dedisp_z12 = None
    dedisp_z20 = None
    dedisp_H = None
    dedisp_M = None


def get_noisier_channels(array):
    newdata = np.copy(array)
    spec = newdata.mean(1)
    smooth_spec = medfilt(spec, 7)
    mad = ref_mad(spec)
    badchans = spec > smooth_spec + 5 * mad
    clean_spec = np.copy(spec)
    clean_spec[badchans] = 0

    return badchans


def renormalize_data(array, diagnostic_figure=None, badchans_mask=None, baseline_window=101,
                     cut_outliers=False):
    from scipy import signal
    renorm_data = np.copy(array).astype(float)
    if badchans_mask is None:
        badchans_mask = np.zeros(renorm_data.shape[0], dtype=bool)

    lc = renorm_data[~badchans_mask, :].mean(0)
    baseline_window = min(baseline_window, lc.size // 100 * 2 + 1)
    lc_smooth = gaussian_filter(lc, baseline_window)
    factor = np.median(lc_smooth) / lc_smooth
    for i, newd in enumerate(renorm_data):
        renorm_data[i, :] *= factor

    spec = renorm_data.mean(1)

    for i, newd in enumerate(renorm_data):
        renorm_data[i, :] = (newd - spec[i]) / spec[i]

    renorm_data[badchans_mask, :] = 0
    # renorm_data[np.roll(badchans_mask, 1), :] = 0
    # renorm_data[np.roll(badchans_mask, -1), :] = 0

    if cut_outliers:
        lc = renorm_data.mean(0)
        for rebin_window in range(0, 5):
            window = 1<<rebin_window
            lc_rebin = uniform_filter1d(lc, window)
            thresh_up = 5 * np.std(lc_rebin[::window])
            thresh_down = -3 * np.std(lc_rebin[::window])

            bad_bins = (lc_rebin > + thresh_up) | (lc_rebin < thresh_down)
            # plt.plot(lc_rebin + rebin_window, alpha=0.5, lw=0.5, c='grey')
            # plt.axhline(thresh + rebin_window)
        # plt.plot(lc, alpha=0.5)
        renorm_data[:, bad_bins] = 0
        # plt.plot(lc)
        # plt.plot(renorm_data.mean(0), alpha=0.5, zorder=10)
        # plt.plot(lc_smooth, lw=2)
        # plt.show()

    return renorm_data


def measure_channel_variability(array, badchans_mask=None):
    newdata = np.copy(array)
    if badchans_mask is None:
        badchans_mask = np.zeros(newdata.shape[0], dtype=bool)

    spec = np.std(newdata, axis=1)

    ordered = np.sort(spec[~badchans_mask])
    q1 = ordered[spec.size // 4]
    q2 = ordered[spec.size // 2]
    q3 = ordered[spec.size // 4 * 3]
    lowlim = q2 - 2 * (q2 - q1)
    hilim = q2 + 2 * (q3 - q2)

    badchans = (spec < lowlim) | (spec > hilim) | badchans_mask

    clean_spec = np.copy(spec)
    clean_spec[badchans] = 0

    return badchans


def dedispersion_search(info, dmmin, dmmax):
    # log.info("Starting search with Dedispersion")
    data = info.allprofs
    start_freq = info.start_freq
    bandwidth = info.bandwidth
    sample_time = 1 / info.pulse_freq / info.nbin

    nchan = data.shape[0]

    # log.info("Creating dedispersion plan")
    trial_DMs = dedispersion_plan(nchan, dmmin, dmmax, start_freq, bandwidth, sample_time)

    # log.info("Allocating dedispersion plane")

    dedispersed_plane = \
        np.memmap('dummy.npy', dtype=np.float, mode='w+', shape=(trial_DMs.size, data.shape[1]))

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
            window = 2 ** window_pow
            reb = quick_resample(dedisp_shift, window)
            snr = np.max(reb) / np.std(reb)
            if snr > best_snr:
                best_snr = snr
                best_win = window

        maxvalues[i] = np.max(dedisp_shift)
        stds[i] = np.std(dedisp_shift)
        best_windows[i] = best_win
        best_snrs[i] = best_snr

    table = Table({'DM': trial_DMs, 'max': maxvalues, 'std': stds, 'snr': best_snrs, 'rebin': best_windows})
    return dedispersed_plane, table


def digitize(data):
    if isinstance(data, np.int):
        return data
    std = mad(data)
    data = (data - np.median(data)) / std * 3
    data[data < 0] = 0
    return np.rint(data).astype(int)


def plot_diagnostics(info, outname='info.jpg', dmmin=200, dmmax=800, t0=0, show=False):
    array = info.allprofs
    start_freq = info.start_freq
    bandwidth = info.bandwidth
    sample_time = 1 / info.pulse_freq / info.nbin
    nchan = array.shape[0]
    trial_DMs = dedispersion_plan(nchan, dmmin, dmmax, start_freq, bandwidth, sample_time)
    allfreqs = np.linspace(info.start_freq, info.start_freq + info.bandwidth, info.nchan +1)
    freqs = (allfreqs[:-1] + allfreqs[1:]) / 2
    df = freqs[1] - freqs[0]
    nchan = info.nchan

    dedispersed_plane, table = \
        dedispersion_search(info, dmmin, dmmax)

    maxsnr = np.argmax(table['snr'])
    dm = table['DM'][maxsnr]
    snr = table['snr'][maxsnr]
    window = table['rebin'][maxsnr]

    shifts = dedispersion_shifts(nchan, dm, start_freq, bandwidth, sample_time)
    dedisp = apply_dm_shifts_to_data(array, shifts)
    array = quick_resample(array, window)
    dedisp = quick_resample(dedisp, window)
    dedispersed_plane = quick_resample(dedispersed_plane, window)

    samples = np.arange(array.shape[1])
    times = samples * sample_time * window + t0

    fig = plt.figure(figsize=(10,8), dpi=50)
    gs = plt.GridSpec(3, 3, height_ratios=(1.5, 1, 1), width_ratios=[0.5, 0.5, 1], hspace=0.01, wspace=0.01)
    ax00 = plt.subplot(gs[2, 0:2])
    ax01 = plt.subplot(gs[2, 2], sharex=ax00, sharey=ax00)
    ax10 = plt.subplot(gs[1, 0:2], sharex=ax00)
    ax11 = plt.subplot(gs[1, 2], sharex=ax00, sharey=ax10)
    ax21 = plt.subplot(gs[0, 2], sharex=ax00)
    ax20 = plt.subplot(gs[0, 0])
    ax20b = plt.subplot(gs[0, 1])

    for ax in [ax20, ax20b, ax21, ax10, ax11]:
        ax.tick_params(labelbottom=False)

    for ax in [ax21, ax11, ax01]:
        ax.tick_params(labelleft=False)

    ax01.set_xlabel("Time (s)")
    ax00.set_xlabel("Time (s)")
    ax00.set_ylabel("Frequency (MHz)")
    ax10.set_ylabel("Flux (arbitrary units)")

    ax20.set_ylabel("Trial DM")
    ax20.set_xlabel("Max - Median (arbitrary units)")
    ax20b.set_xlabel("H test (arbitrary units)")

    ax00.pcolormesh(times, allfreqs, array, rasterized=True)
    ax01.pcolormesh(times, allfreqs, dedisp, rasterized=True)
    ax10.plot(times, array.mean(0), rasterized=True)
    ax11.plot(times, dedisp.mean(0), rasterized=True)
    ax21.pcolormesh(times, trial_DMs, dedispersed_plane, rasterized=True)
    ax20.plot(-table['snr'], table['DM'])
    digitized_plane = digitize(dedispersed_plane)
    h_values = [-h_test(row, nmax=row.size // 10)[0] for row in digitized_plane]

    ax20b.plot(h_values, trial_DMs)
    ax00.set_xlim(t0, times[-1])

    text = f"""
    Obs. Date: {info.date}
    Freq:      {start_freq}--{start_freq + bandwidth}
    Best DM:   {dm}
    Best SNR:  {snr}
    """
    ax20.text(0.5, 0.5, text, va="center", ha="center", transform=ax20.transAxes)

    plt.savefig(f'{outname}',bbox_inches='tight')
    if show:
        plt.show()
    plt.close(fig)


def dm_broadening(dm, freq, df):
    return 8300 * dm * df / freq**3


def search_by_chunks(fname, chunk_length=None, new_sample_time=None, tmin=0, dmmin=200, dmmax=800, surelybad=[]):
    log.info(f"Opening file {fname}")
    fname_root = os.path.basename(fname).split('.')[0]

    mask = get_bad_chans(fname)
    for bad_chan in surelybad:
        mask[bad_chan] = True

    fil = FilReader(fname)
    header = fil.header
    nsamples = header['nsamples']
    sample_time = header['tsamp']
    start_freq = header['fbottom']
    stop_freq = header['ftop']
    bandwidth = header['bandwidth']
    sample_time = header['tsamp']
    nchan = header['nchans']
    foff = header['foff']
    date = header['tstart']

    delta = delta_delay(dmmax, start_freq, stop_freq)
    log.info(f"Expected delay at DM {dmmax} between {start_freq}--{stop_freq} MHz: {delta:2} s")
    if chunk_length is None:
        chunk_length = delta

    step = max(int(chunk_length / sample_time) * 2, 128)

    # Get the _minimum_ DM broadening to set the automatic rebinning
    dm_dt = dm_broadening(dmmin, start_freq, np.abs(foff))

    if new_sample_time is None:
        new_sample_time = max(dm_dt / 10, sample_time)

    log.info(f"Expected broadening in spectral bin at DM {dmmin} at {start_freq} MHz: {dm_dt * 1e6:2} us")

    sampl_ratio = new_sample_time / sample_time
    N = 1
    if sampl_ratio >= 2:
        N = int(np.rint(sampl_ratio))
        new_sample_time = N * sample_time
        log.info(f"Data will be resampled to {new_sample_time} s")

    for istart in tqdm.tqdm(range(0, nsamples, step // 2)):
        chunk_size = min(step, nsamples - istart)
        t0 = istart * sample_time
        if t0 < tmin:
            continue
        info = PulseInfo()
        if chunk_size < step // 2:
            continue
        iend = istart + chunk_size
        array = fil.readBlock(istart, chunk_size, as_filterbankBlock=False)

        array = renormalize_data(array, badchans_mask=mask)
        print(not np.any(array[mask, :]))
        # array = (array - array.min()) / np.std(array)
        if foff < 0:
            array = array[::-1]

        if N > 1:
            array = quick_resample(array, N)

        info.allprofs = array
        info.start_freq = start_freq
        info.bandwidth = bandwidth
        info.nbin = array.shape[1]
        info.nchan = array.shape[0]
        info.date = date
        info.pulse_freq = 1 / (info.nbin * new_sample_time)

        table = fast_dedispersion_search(info.allprofs, dmmin, dmmax, start_freq, bandwidth, new_sample_time)
        plot_diagnostics(info, outname=f'{fname_root}_{istart}-{iend}.jpg', dmmin=dmmin, dmmax=dmmax, t0=t0, show=True)

        if np.any(table['snr'] > 6):
            plot_diagnostics(info, outname=f'{fname_root}_{istart}-{iend}.jpg', dmmin=dmmin, dmmax=dmmax, t0=t0)
            pickle.dump(info, open(f'{fname_root}_{istart}-{iend}.pkl', 'wb'))


def cleanup_data(fname, outname):
    mask = get_bad_chans(fname)
    fil = FilReader(fname)



def main_search(args=None):
    import argparse
    parser = \
        argparse.ArgumentParser(description="Clean the data and search for FRBs")
    parser.add_argument("fnames", help="Input binary files in filterbank format", type=str, nargs='+')
    args = parser.parse_args(args)

    for fname in args.fnames:
        # save_data_to_job(fname, new_sample_time=0.001, tmin=2250, dmmin=300, dmmax=400,
        #     surelybad=np.concatenate((np.arange(32, 37), np.arange(230, 241))))
        # save_data_to_job(fname, new_sample_time=0.001, dmmin=700, dmmax=850,
        #     surelybad=[1, 2, 5, 6, 7])
        search_by_chunks(fname, new_sample_time=None, dmmin=300, dmmax=400)


def main_clean(args=None):
    import argparse
    parser = \
        argparse.ArgumentParser(description="Clean the data")
    parser.add_argument("fnames", help="Input binary files in filterbank format", type=str, nargs='+')
    args = parser.parse_args(args)

    for fname in args.fnames:
        # save_data_to_job(fname, new_sample_time=0.001, tmin=2250, dmmin=300, dmmax=400,
        #     surelybad=np.concatenate((np.arange(32, 37), np.arange(230, 241))))
        # save_data_to_job(fname, new_sample_time=0.001, dmmin=700, dmmax=850,
        #     surelybad=[1, 2, 5, 6, 7])
        search_by_chunks(fname, new_sample_time=None, dmmin=300, dmmax=400)

