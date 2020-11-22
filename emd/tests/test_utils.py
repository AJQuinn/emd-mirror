import unittest

import numpy as np


class test_amplitude_normalise(unittest.TestCase):

    def setUp(self):

        # Create core signal
        seconds = 5.1
        sample_rate = 2000
        f1 = 2
        f2 = 18
        time_vect = np.linspace(0, seconds, int(seconds * sample_rate))
        am = np.sin(2*np.pi*f1*time_vect) + 1

        self.x = am * np.cos(2.3 * np.pi * f2 * time_vect)

    def test_amplitude_normalise(self):
        from ..utils import amplitude_normalise

        amp_norm = amplitude_normalise(self.x[:, np.newaxis])

        # start signal should range between +/- 2
        assert(2 - np.max(self.x) < 1e-2)
        assert(2 + np.min(self.x) < 1e-2)

        # norm signal should range between +/-
        assert(1 - np.max(amp_norm) < 1e-2)
        assert(1 + np.min(amp_norm) < 1e-2)


class test_epochs(unittest.TestCase):

    def setUp(self):

        # Create core signal
        seconds = 5
        sample_rate = 2000
        f2 = 5
        time_vect = np.linspace(0, seconds, int(seconds * sample_rate))

        self.x = np.cos(2 * np.pi * f2 * time_vect + np.pi/2)

    def test_find_extrema_locked_epochs(self):
        from ..utils import find_extrema_locked_epochs

        trls = find_extrema_locked_epochs(self.x, 40)

        assert(trls.shape[0] == 25)
        assert(np.all(np.unique(np.diff(trls)) == 40))
