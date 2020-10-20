
import unittest

import numpy as np

from ..sift import sift, ensemble_sift, complete_ensemble_sift, \
                   mask_sift, get_config
from ..utils import abreu2010


class test_sift_defaults(unittest.TestCase):

    def setUp(self):

        # Create core signal
        seconds = 5.1
        sample_rate = 2000
        f1 = 2
        f2 = 18
        time_vect = np.linspace(0, seconds, int(seconds * sample_rate))

        x = abreu2010(f1, .2, 0, sample_rate, seconds)
        self.x = x + np.cos(2.3 * np.pi * f2 * time_vect) + np.linspace(-.5, 1, len(time_vect))

    def test_sift_default(self):
        """Check basic sift runs with some simple settings"""
        imf = sift(self.x)
        assert(imf.shape[0] == self.x.shape[0])  # just checking that it ran

    def test_ensemble_sift_default(self):
        """Check ensemble sift runs with some simple settings"""
        imf = ensemble_sift(self.x[:500], max_imfs=3)
        assert(imf.shape[0] == self.x[:500].shape[0])  # just checking that it ran

    def test_complete_ensemble_sift_default(self):
        """Check complete ensemble sift runs with some simple settings"""
        imf = complete_ensemble_sift(self.x[:200])
        assert(imf[0].shape[0] == self.x[:200].shape[0])  # just checking that it ran

    def test_mask_sift_default(self):
        """Check adaptive mask sift runs with some simple settings"""
        imf = mask_sift(self.x[:200], max_imfs=5, mask_freqs='zc')
        assert(imf.shape[0] == self.x[:200].shape[0])  # just checking that it ran


class test_sift_behaviour(unittest.TestCase):

    def get_resid(self, x, x_bar):
        ss_orig = np.power(x, 2).sum()
        ss_resid = np.power(x - x_bar, 2).sum()
        return (ss_orig - ss_resid) / ss_orig

    def check_diff(self, val, target, eta=1e-3):
        return np.abs(val - target) < eta

    def setUp(self):

        # Create core signal
        seconds = 5.1
        sample_rate = 2000
        f1 = 2
        f2 = 18
        time_vect = np.linspace(0, seconds, int(seconds * sample_rate))

        x = abreu2010(f1, .2, 0, sample_rate, seconds)
        self.x = x + np.cos(2.3 * np.pi * f2 * time_vect) + np.linspace(-.5, 1, len(time_vect))

        self.imf_kwargs = {}
        self.envelope_opts = {'interp_method': 'splrep'}
        self.imf = sift(self.x, imf_opts=self.imf_kwargs, envelope_opts=self.envelope_opts)

    def test_complete_decomposition(self):
        """Test that IMFs are complete description of signal"""

        assert(np.allclose(self.x, self.imf.sum(axis=1)) is True)

    # Implement the four checks from https://doi.org/10.1016/j.ymssp.2007.11.028
    def test_sift_multiplied_by_constant(self):
        """Test that sifting a scaled signal only changes the scaling of the IMFs"""
        x2 = self.imf[:, 0] * 3
        imf2 = sift(x2, imf_opts=self.imf_kwargs)

        tst = self.check_diff(self.get_resid(self.imf[:, 0] * 3, imf2[:, 0]), 1)
        assert(tst)

    def test_sift_plus_constant(self):
        """Test that sifting a signal plus a constant only changes the last IMF"""
        x3 = self.x + 2
        imf3 = sift(x3, imf_opts=self.imf_kwargs)

        tst = list()
        for ii in range(imf3.shape[1]):
            tmp = self.imf[:, ii]
            if ii == self.imf.shape[1] - 1:
                tmp = tmp + 2
            tst.append(self.check_diff(self.get_resid(tmp, imf3[:, ii]), 1))

        assert(all(tst) is True)

    def test_sift_of_imf(self):
        """Test that sifting an IMF returns the IMF"""
        x4 = self.imf[:, 0].copy()
        imf4 = sift(x4, imf_opts=self.imf_kwargs)

        tst = list()
        for ii in range(imf4.shape[1]):
            if ii == 0:
                target = 1
            else:
                target = 0
            tst.append(self.check_diff(self.get_resid(x4, imf4[:, ii]), target))

        assert(all(tst) is True)

    def test_sift_of_reversed_signal(self):
        """Test that sifting a reversed signal only reverses the IMFs"""
        x5 = self.x[::-1]
        imf5 = sift(x5, imf_opts=self.imf_kwargs)

        tst = self.check_diff(self.get_resid(self.imf[::-1, 0], imf5[:, 0]), 1)

        assert(tst)

    # Test mask sifts
    def test_get_next_imf_mask(self):
        from ..sift import get_next_imf_mask

        # sift with mask above signal should return zeros
        # mask has to be waaay above signal in a noiseless time-series
        next_imf, continue_flag = get_next_imf_mask(self.imf[:, 0, None], 0.25, 1)
        mask_power = np.sum(np.power(next_imf, 2))

        assert(mask_power < 1)

        # sift with mask below signal should return original signal
        next_imf, continue_flag = get_next_imf_mask(self.imf[:, 0, None], 0.0001, 1)
        power = np.sum(np.power(self.imf[:, 0], 2))
        mask_power = np.sum(np.power(next_imf, 2))

        assert(power - mask_power < 1)


class test_sift_config(unittest.TestCase):

    def test_config(self):

        # Get sift config
        conf = get_config('sift')
        # Check a couple of options
        assert(conf['max_imfs'] is None)
        assert(conf['extrema_opts/pad_width'] == 2)
        assert(conf['extrema_opts/loc_pad_opts/mode'] == 'reflect')

        # Get ensemble sift config
        conf = get_config('ensemble_sift')
        # Check a couple of options
        assert(conf['max_imfs'] is None)
        assert(conf['extrema_opts/pad_width'] == 2)
        assert(conf['extrema_opts/loc_pad_opts/mode'] == 'reflect')

        # Get mask sift config
        conf = get_config('ensemble_sift')
        # Check a couple of options
        assert(conf['nensembles'] == 4)
        assert(conf['max_imfs'] is None)
        assert(conf['extrema_opts/pad_width'] == 2)
        assert(conf['extrema_opts/loc_pad_opts/mode'] == 'reflect')
