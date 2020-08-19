import pytest
import numpy as np
from pulsarutils.dedispersion import dedispersion_shifts


def simulate_test_data(
        dm=150, tsamp=0.0005, nsamples=1024, nchan=128,
        start_freq=1200., bandwidth=200., signal=1., noise=0.5):
    array = np.zeros((nchan, nsamples))
    array[:, nsamples // 2] = signal
    array = np.abs(np.random.normal(array, noise))
    # array = np.random.poisson(array).astype(int)

    nchan = array.shape[0]


    shifts = dedispersion_shifts(nchan, dm, start_freq, bandwidth, tsamp)
    for i in range(nchan):
        array[i, :] = np.roll(array[i, :], int(shifts[i]))

    header = {'bandwidth': bandwidth,
              'fbottom': start_freq,
              'foff': bandwidth / nchan,
              'nchans': nchan,
              'nsamples': nsamples,
              'tsamp': tsamp}

    return array, header
