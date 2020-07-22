
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
        from ..cycles import get_cycle_inds
        from ..spectra import frequency_stats

        x = np.sin(2 * np.pi * f * self.time_vect + phase)[:, None]

        # Add a wobble
        if distort is not None:
            x[distort - 25:distort + 25, 0] += np.linspace(-.1, .1, 50)

        # This is a perfect sin so we can use normal hilbert
        IP, IF, IA = frequency_stats(x, self.sample_rate, 'hilbert')
        # Find good cycles
        cycles = get_cycle_inds(IP, return_good=True)[:, 0]

        return cycles

    def test_simple_cycle_counting(self):

        # Test basic cycle detection
        uni_cycles = np.unique(self.cycle_generator(4))
        assert(np.all(uni_cycles == np.arange(9)))

        uni_cycles = np.unique(self.cycle_generator(5, phase=1.5 * np.pi))
        assert(np.all(uni_cycles == np.arange(11)))

    def test_cycle_count_with_bad_start_and_end(self):

        # Test basic cycle detection
        cycles = self.cycle_generator(4, phase=0)
        uni_cycles = np.unique(cycles)
        assert(np.all(uni_cycles == np.arange(9)))
        assert(cycles[50] == 0)
        assert(cycles[2150] == 0)

        cycles = self.cycle_generator(5, phase=0)
        uni_cycles = np.unique(cycles)
        assert(np.all(uni_cycles == np.arange(11)))
        assert(cycles[50] == 0)
        assert(cycles[2150] == 0)

    def test_cycle_count_with_bad_in_middle(self):

        cycles = self.cycle_generator(4, phase=1.5 * np.pi, distort=1100)
        uni_cycles = np.unique(cycles)
        assert(np.all(uni_cycles == np.arange(8)))
        assert(cycles[1100] == 0)

    def test_cycle_chain(self):
        from ..cycles import get_cycle_chain

        cycles = self.cycle_generator(4, phase=0)
        chain = get_cycle_chain(cycles)
        assert(np.all(chain[0] == np.array([1, 2, 3, 4, 5, 6, 7, 8])))

        cycles = self.cycle_generator(4, phase=0, distort=1200)
        chain = get_cycle_chain(cycles)
        assert(np.all(chain == [[1, 2, 3, 4], [5, 6, 7]]))

        chain = get_cycle_chain(cycles, drop_first=True)
        assert(np.all(chain == np.array([[2, 3, 4], [6, 7]])))

        chain = get_cycle_chain(cycles, drop_last=True)
        assert(np.all(chain == np.array([[1, 2, 3], [5, 6]])))

        chain = get_cycle_chain(cycles, drop_first=True, drop_last=True)
        assert(np.all(chain == np.array([[2, 3], [6]])))

        cycles = self.cycle_generator(4, phase=0, distort=800)
        chain = get_cycle_chain(cycles)
        assert(np.all(chain == np.array([[1, 2], [3, 4, 5, 6, 7]])))

        chain = get_cycle_chain(cycles, min_chain=3)
        assert(np.all(chain == np.array([3, 4, 5, 6, 7])))


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

    x = np.array([0, 1, 1, 1, 1, 2, 2, 3, 3, 3, 0])
    y = np.ones_like(x)

    # Compute the average of y within bins of x
    bin_avg = get_cycle_stat(x, y)
    assert(np.all(bin_avg == [1., 1., 1., 1.]))

    # Compute sum of y within bins of x and return full vector
    bin_avg = get_cycle_stat(x, y, mode='full', func=np.sum)
    assert(np.allclose(bin_avg, np.array([np.nan, 4., 4., 4., 4., 2., 2., 3., 3., 3., np.nan]), equal_nan=True))

    # Compute the sum of y within bins of x
    bin_counts = get_cycle_stat(x, y, func=np.sum)
    assert(np.all(bin_counts == [2, 4, 2, 3]))
