
import unittest

import numpy as np


class test_spectra(unittest.TestCase):

    def setUp( self ):

        # Create core signal
        seconds = 10
        self.sample_rate = 2000
        self.f1 = 5
        self.f2 = 18
        time_vect = np.linspace(0,seconds,int(seconds*self.sample_rate))

        self.x1 =  np.cos( 2*np.pi*self.f1*time_vect )[:,None]
        self.x2 =  2*np.cos( 2*np.pi*self.f2*time_vect )[:,None]


    def test_frequency_stats( self ):
        from ..spectra import frequency_stats

        tol = 1e-3 # Relatively generous tol due to edge effects

        # Check first signal
        IP,IF,IA = frequency_stats( self.x1, self.sample_rate, 'hilbert' )
        assert( IP.max()-(2*np.pi) < tol )
        assert( IP.min() < tol )
        assert( IA.mean()-1 < tol )
        assert( IF.mean()-self.f1 < tol )

        # Check second signal
        IP,IF,IA = frequency_stats( self.x2, self.sample_rate, 'hilbert' )
        assert( IP.max()-(2*np.pi) < tol )
        assert( IP.min() < tol )
        assert( IA.mean()-2 < tol )
        assert( IF.mean()-self.f2 < tol )

    def test_freq_from_phase( self ):
        from ..spectra import freq_from_phase

        tst = freq_from_phase( np.linspace(0,2*np.pi,48), 47 )
        assert( np.allclose(tst,1) )

        tst = freq_from_phase( np.linspace(0,2*np.pi*.5,48), 47 )
        assert( np.allclose(tst,.5) )

        tst = freq_from_phase( np.linspace(0,2*np.pi*2,48), 47 )
        assert( np.allclose(tst,2) )

    def test_phase_from_freq( self ):
        from ..spectra import phase_from_freq

        tol = 1e-6

        phs = phase_from_freq( np.ones((100,)), sample_rate=100)
        assert( phs.max()-np.pi < tol )
