
import unittest

import numpy as np


class test_cycles(unittest.TestCase):

    def setUp(self):
        self.sample_rate = 1000
        self.seconds = 2
        self.pad_time = .1
        nsamples = int((self.sample_rate * self.seconds) + (2*self.pad_time*self.sample_rate))
        self.time_vect = np.linspace(-self.pad_time,
                                     self.seconds+self.pad_time,
                                     nsamples)
        self.signal = np.sin(2 * np.pi * 10 * self.time_vect)[:, None]

    def cycle_generator(self, f, phase=0, distort=None):
        from ..cycles import get_cycle_vector
        from ..spectra import frequency_transform

        x = np.sin(2 * np.pi * f * self.time_vect + phase)[:, None]

        # Add a wobble
        if distort is not None:
            x[distort - 25:distort + 25, 0] += np.linspace(-.1, .1, 50)

        # This is a perfect sin so we can use normal hilbert
        IP, IF, IA = frequency_transform(x, self.sample_rate, 'hilbert')
        # Find good cycles
        cycles = get_cycle_vector(IP, return_good=True)[:, 0]

        return cycles

    def test_simple_cycle_counting(self):

        # Test basic cycle detection
        uni_cycles = np.unique(self.cycle_generator(4))
        assert(np.all(uni_cycles == np.arange(-1, 8)))

        uni_cycles = np.unique(self.cycle_generator(5, phase=1.5 * np.pi))
        print(uni_cycles)
        assert(np.all(uni_cycles == np.arange(-1, 10)))

    def test_cycle_count_with_bad_start_and_end(self):

        # Test basic cycle detection
        cycles = self.cycle_generator(4, phase=0)
        uni_cycles = np.unique(cycles)
        assert(np.all(uni_cycles == np.arange(-1, 8)))
        assert(cycles[50] == -1)
        assert(cycles[2150] == -1)

        cycles = self.cycle_generator(5, phase=0)
        uni_cycles = np.unique(cycles)
        assert(np.all(uni_cycles == np.arange(-1, 10)))
        assert(cycles[50] == -1)
        assert(cycles[2150] == -1)

    def test_cycle_count_with_bad_in_middle(self):

        cycles = self.cycle_generator(4, phase=1.5 * np.pi, distort=1100)
        uni_cycles = np.unique(cycles)
        assert(np.all(uni_cycles == np.arange(-1, 7)))
        assert(cycles[1100] == -1)

    def test_cycle_control_points(self):
        from ..cycles import get_control_points

        x = np.sin(2*np.pi*np.linspace(0, 1, 1280))
        cycles = np.ones_like(x, dtype=int)
        ctrl = get_control_points(x, cycles)

        # We accept a 1 sample error in ctrl point location...
        ref = 1280*np.linspace(0, 1, 5)
        assert(np.abs(ctrl-ref).max())


class test_cycles_support(unittest.TestCase):

    def setUp(self):
        from ..cycles import get_cycle_vector, get_subset_vector, get_chain_vector
        from ..spectra import frequency_transform
        from .._cycles_support import get_cycle_stat_from_samples

        X = np.sin(2*np.pi*10*np.linspace(0, 2, 512))
        X = X * (2-np.cos(2*np.pi*1*np.linspace(0, 2, 512)))
        IP, IF, IA = frequency_transform(X, 512, 'hilbert')

        self.cycle_vect = get_cycle_vector(IP[:, 0], return_good=False)
        self.max_amps = get_cycle_stat_from_samples(IA[:, 0], self.cycle_vect, np.max)

        valids = self.max_amps > 1.5
        self.subset_vect = get_subset_vector(valids)
        self.chain_vect = get_chain_vector(self.subset_vect)

    def test_cycle_maps(self):

        from .._cycles_support import map_sample_to_cycle, map_cycle_to_samples
        # Test 1 - q2 should contain 350
        q1 = map_sample_to_cycle(self.cycle_vect, 350)
        q2 = map_cycle_to_samples(self.cycle_vect, q1[0])
        assert(350 in q2)

        from .._cycles_support import map_cycle_to_subset, map_subset_to_cycle
        # Test 2 - should recover 9
        q3 = map_subset_to_cycle(self.subset_vect, 9)
        q4 = map_cycle_to_subset(self.subset_vect, q3)
        assert(q4[0] == 9)

        from .._cycles_support import map_sample_to_subset, map_subset_to_sample
        # Test 3 - should recover 350
        q5 = map_sample_to_subset(self.subset_vect, self.cycle_vect, 350)
        q6 = map_subset_to_sample(self.subset_vect, self.cycle_vect, q5[0])
        assert(350 in q6)

        from .._cycles_support import map_chain_to_subset, map_subset_to_chain
        # Test 4 - Should recover 7
        q7 = map_subset_to_chain(self.chain_vect, 7)
        q8 = map_chain_to_subset(self.chain_vect, q7)
        assert(7 in q8)

        from .._cycles_support import map_cycle_to_chain
        # Test 5 - check that third cycle with -1 in subset doesn't have a chain
        q9 = map_cycle_to_chain(self.chain_vect, self.subset_vect, np.where(self.subset_vect == -1)[0][3])
        assert(q9 is None)


class test_cycles_object(unittest.TestCase):

    def setUp(self):
        from ..spectra import frequency_transform
        from ..cycles import Cycles

        X = np.sin(2*np.pi*10*np.linspace(0, 2, 512))
        self.X = X * (2-np.cos(2*np.pi*1*np.linspace(0, 2, 512)))
        self.IP, self.IF, self.IA = frequency_transform(X, 512, 'hilbert')

        self.C = Cycles(self.IP[:, 0])

    def test_cycle_object_metrics(self):
        from ..cycles import cf_ascending_zero_sample

        self.C.compute_cycle_metric('max_amp', self.IA[:, 0], np.max)
        self.C.compute_cycle_timings()

        self.C.compute_cycle_metric('asc_samp', self.X, cf_ascending_zero_sample, mode='augmented')

        xx = np.arange(self.C.ncycles)
        self.C.add_cycle_metric('range', xx)

        conditions = ['max_amp>0.75']
        self.C.pick_cycle_subset(conditions)
        self.C.compute_chain_timings()

        df = self.C.get_metric_dataframe()
        assert(len(df['max_amp']) == self.C.ncycles)
        df = self.C.get_metric_dataframe(subset=True)
        assert(len(df['max_amp']) == 20)
        conditions = ['max_amp>0.75', 'range>5']
        df = self.C.get_metric_dataframe(conditions=conditions)
        assert(len(df['max_amp']) == 14)

    def test_cycle_object_iteration(self):
        from ..cycles import phase_align
        pa, phasex = phase_align(self.IP, self.IF, self.C)


class test_kdt_match(unittest.TestCase):

    def test_kdt(self):
        x = np.linspace(0, 1)
        y = np.linspace(0, 1, 10)

        from ..cycles import kdt_match
        x_inds, y_inds = kdt_match(x, y, K=2)

        assert(all(y_inds == np.arange(10)))

        xx = np.array([0, 5, 11, 16, 22, 27, 33, 38, 44, 49])
        assert(all(x_inds == xx))


def test_get_cycle_vals():
    from ..cycles import get_cycle_stat

    x = np.array([-1, 0, 0, 0, 0, 1, 1, 2, 2, 2, -1])
    y = np.ones_like(x)

    # Compute the average of y within bins of x
    bin_avg = get_cycle_stat(x, y)
    print(bin_avg)
    assert(np.all(bin_avg == [1., 1., 1.]))

    # Compute sum of y within bins of x and return full vector
    bin_avg = get_cycle_stat(x, y, out='samples', func=np.sum)
    assert(np.allclose(bin_avg, np.array([np.nan, 4., 4., 4., 4., 2., 2., 3., 3., 3., np.nan]), equal_nan=True))

    # Compute the sum of y within bins of x
    bin_counts = get_cycle_stat(x, y, func=np.sum)
    print(bin_counts)
    assert(np.all(bin_counts == [4, 2, 3]))
