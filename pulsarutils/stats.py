import os
import numpy as np
from scipy.signal import medfilt, savgol_filter
from statsmodels.robust import mad
import matplotlib.pyplot as plt
from sigpyproc.Readers import FilReader
import tqdm
from astropy import log


def ref_mad(array, window=1):
    """Ref. Median Absolute Deviation of an array, rolling median-subtracted.

    If a data series is noisy, it is difficult to determine the underlying
    statistics of the original series. Here, the MAD is calculated in a rolling
    window, and the minimum is saved, because it will likely be the interval
    with less noise.

    Parameters
    ----------
    array : ``numpy.array`` object or list
        Input data
    window : int or float
        Number of bins of the window

    Returns
    -------
    ref_std : float
        The reference MAD
    """

    return mad(np.diff(array)) / np.sqrt(2)


def get_spectral_stats(fname, chunksize=100000):
    log.info("Getting spectral statistics...")
    fil = FilReader(fname)
    header = fil.header
    nsamples = header['nsamples']
    spectrum = 0.
    spectrsq = 0.
    for istart in tqdm.tqdm(range(0, nsamples, chunksize)):
        size = min(chunksize, nsamples - istart)
        array = fil.readBlock(istart, size, as_filterbankBlock=False)
        # print(arrayq)
        local_spec = array.astype(float).sum(1)
        local_sq = ((array).astype(float) **2).sum(1)
        spectrum += local_spec
        spectrsq += local_sq

    mean_spec = spectrum / nsamples
    mean_spectrsq = spectrsq / nsamples
    std_spec = np.sqrt(mean_spectrsq - mean_spec ** 2)
    return mean_spec, std_spec


def get_bad_chans(fname, show=False, cache=None):
    if cache is None:
        cache = fname + '.badchans'
    if os.path.exists(cache):
        return np.loadtxt(cache)
    mean_spec, mean_std = get_spectral_stats(fname)
    badchans = np.zeros(mean_spec.size, dtype=bool)
    chans = np.arange(mean_spec.size)

    for spec in (mean_spec, mean_std):
        smooth_spec = medfilt(spec, 11)
        spec_mad = ref_mad(spec)
        threshold = smooth_spec + 4 * spec_mad
        badchans = badchans | (spec > threshold)
        clean_spec = np.copy(spec)
        clean_spec[badchans] = smooth_spec[badchans]
        if show:
            plt.figure()
            plt.plot(chans, spec, color='grey', drawstyle='steps-mid')
            plt.plot(chans, smooth_spec, color='k', drawstyle='steps-mid')
            plt.plot(chans, threshold, color='r', drawstyle='steps-mid')
            plt.plot(chans, clean_spec, color='b', lw=2, drawstyle='steps-mid')
    if show:
        plt.show()
    print(f"Bad chans: {chans[badchans]}")
    np.savetxt(cache, [badchans])
    return badchans


def main(args=None):
    import argparse
    parser = \
        argparse.ArgumentParser(description="Get bad channels")
    parser.add_argument("fnames", help="Input binary files", type=str, nargs='+')
    args = parser.parse_args(args)

    for fname in args.fnames:
        get_bad_chans(fname, show=True)
