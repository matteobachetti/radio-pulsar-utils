import pytest
import numpy as np
from pulsarutils.simulate import simulate_test_data
from pulsarutils.dedispersion import dedispersion_search



class TestDedispersion(object):
    @classmethod
    def setup_class(cls):
        array, header = simulate_test_data(150)
        cls.data = array
        cls.header = header

    def test_dedispersion_search(self):
        arr = self.data
        header = self.header
        nsamples = header['nsamples']
        nchan = header['nchans']
        start_freq = header['fbottom']
        bandwidth = header['bandwidth']
        sample_time = header['tsamp']
        foff = header['foff']
        table = dedispersion_search(arr, 100, 200., start_freq, bandwidth, sample_time)
        assert np.isclose(table['DM'][np.argmax(table['snr'])], 150, atol=1)

    def test_dedispersion_search_slow(self):
        arr = self.data
        header = self.header
        nsamples = header['nsamples']
        nchan = header['nchans']
        start_freq = header['fbottom']
        bandwidth = header['bandwidth']
        sample_time = header['tsamp']
        foff = header['foff']
        table, plane = dedispersion_search(arr, 100, 200., start_freq, bandwidth, sample_time, show=True)
        assert np.isclose(table['DM'][np.argmax(table['snr'])], 150, atol=1)
