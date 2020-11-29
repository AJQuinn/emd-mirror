
import unittest

import numpy as np


class test_spectra(unittest.TestCase):

    def setUp(self):

        # Create core signal
        seconds = 10
        self.sample_rate = 2000
        self.f1 = 5
        self.f2 = 18
        time_vect = np.linspace(0, seconds, int(seconds * self.sample_rate))

        self.x1 = np.cos(2 * np.pi * self.f1 * time_vect)[:, None]
        self.x2 = 2 * np.cos(2 * np.pi * self.f2 * time_vect)[:, None]

    def test_frequency_transform(self):
        from ..spectra import frequency_transform

        tol = 1e-3  # Relatively generous tol due to edge effects

        # Check first signal
        IP, IF, IA = frequency_transform(self.x1, self.sample_rate, 'hilbert')
        assert(IP.max() - (2 * np.pi) < tol)
        assert(IP.min() < tol)
        assert(IA.mean() - 1 < tol)
        assert(IF.mean() - self.f1 < tol)

        # Check second signal
        IP, IF, IA = frequency_transform(self.x2, self.sample_rate, 'hilbert')
        assert(IP.max() - (2 * np.pi) < tol)
        assert(IP.min() < tol)
        assert(IA.mean() - 2 < tol)
        assert(IF.mean() - self.f2 < tol)

    def test_freq_from_phase(self):
        from ..spectra import freq_from_phase

        tst = freq_from_phase(np.linspace(0, 2 * np.pi, 48), 47)
        assert(np.allclose(tst, 1))

        tst = freq_from_phase(np.linspace(0, 2 * np.pi * .5, 48), 47)
        assert(np.allclose(tst, .5))

        tst = freq_from_phase(np.linspace(0, 2 * np.pi * 2, 48), 47)
        assert(np.allclose(tst, 2))

    def test_phase_from_freq(self):
        from ..spectra import phase_from_freq

        tol = 1e-6

        phs = phase_from_freq(np.ones((100,)), sample_rate=100)
        assert(phs.max() - np.pi < tol)

    def test_hilberthunang_1d(self):
        from ..spectra import hilberthuang_1d

        IF = np.linspace(0, 12, 13)[:, None]
        IA = np.ones_like(IF)

        # We should 2 bins with 5 frequencies in each bin.
        edges = np.linspace(0, 12, 3)
        spec = hilberthuang_1d(IF, IA, edges, mode='amplitude')
        assert(np.all(spec[:, 0] == [6, 6]))

        # We should 4 bins with 3 frequencies in each bin.
        edges = np.linspace(0, 12, 5)
        spec = hilberthuang_1d(IF, IA, edges, mode='amplitude')
        assert(np.all(spec[:, 0] == [3, 3, 3, 3]))

        IA = IA * 2
        # We should 4 bins with 3 frequencies in each bin, energy should be 12
        # per bin (3*(2**2))
        edges = np.linspace(0, 12, 5)
        spec = hilberthuang_1d(IF, IA, edges, mode='energy')
        assert(np.all(spec[:, 0] == [12, 12, 12, 12]))

    def test_hilberthuang(self):
        from ..spectra import hilberthuang

        IF = np.linspace(0, 12, 13)[:, None]
        IA = np.ones_like(IF)
        edges = np.linspace(0, 13, 3)

        hht = hilberthuang(IF, IA, edges)

        # Check total amplitude is equal in HHT and IA
        assert(hht.sum() == IA.sum())

        assert(np.all(hht[0, :7] == np.array([1., 1., 1., 1., 1., 1., 1.])))
        assert(np.all(hht[1, :7] == np.array([0., 0., 0., 0., 0., 0., 0.])))
        assert(np.all(hht[1, 7:] == np.array([1., 1., 1., 1., 1., 1.])))
        assert(np.all(hht[0, 7:] == np.array([0., 0., 0., 0., 0., 0.])))


class test_hists(unittest.TestCase):

    def test_hist_bins_from_data(self):
        from ..spectra import define_hist_bins_from_data

        data = np.linspace(0, 1, 16)
        edges, bins = define_hist_bins_from_data(data)

        assert(np.all(edges == np.array([0., .25, .5, .75, 1.])))
        assert(np.all(bins == np.array([0.125, 0.375, 0.625, 0.875])))

    def test_hist_bins(self):
        from ..spectra import define_hist_bins

        edges, bins = define_hist_bins(0, 1, 5)

        edges = np.round(edges, 6)  # Sometimes returns float errors 0.30000000000000004
        bins = np.round(bins, 6)  # Sometimes returns float errors 0.30000000000000004

        assert(np.all(edges == np.array([0., 0.2, 0.4, 0.6, 0.8, 1.])))
        assert(np.all(bins == np.array([0.1, 0.3, 0.5, 0.7, 0.9])))


class test_holo(unittest.TestCase):

    def test_holo(self):
        from ..spectra import holospectrum, define_hist_bins

        f_edges1, f_bins1 = define_hist_bins(0, 10, 5)
        f_edges2, f_bins2 = define_hist_bins(0, 1, 5)

        if1 = np.array([2, 6])[:, None]
        if2 = np.array([.2, .3])[:, None, None]
        ia2 = np.array([1, 2])[:, None, None]

        holo = holospectrum(if1, if2, ia2, f_edges1, f_edges2, squash_time=False)

        assert(np.all(holo.shape == (2, 5, 5)))
