
import unittest

import numpy as np

from ..sift import sift
from ..utils import abreu2010

class test_sifts(unittest.TestCase):

    def get_resid( self, x, x_bar ):
        ss_orig = np.power( x,2 ).sum()
        ss_resid = np.power(x-x_bar,2).sum()
        return (ss_orig-ss_resid)/ss_orig

    def check_diff( self, val, target, eta=1e-3):
        return np.abs( val-target ) < eta

    def setUp( self ):

        # Create core signal
        seconds = 5.1
        sample_rate = 2000
        f1 = 5
        f2 = 18
        time_vect = np.linspace(0,seconds,int(seconds*sample_rate))

        x = abreu2010( 2, .2, 0, sample_rate, seconds )
        self.x = x + np.cos( 2.3*np.pi*f2*time_vect ) + np.linspace(-.5,1,len(time_vect))
        self.imf = sift( self.x, interp_method='splrep' )


    def test_complete_decomposition( self ):
        """Test that IMFs are complete description of signal"""

        assert( np.allclose( self.x, self.imf.sum(axis=1)) is True )


    # Implement the four checks from https://doi.org/10.1016/j.ymssp.2007.11.028
    def test_sift_multiplied_by_constant( self ):
        """Test that sifting a scaled signal only changes the scaling of the IMFs"""
        x2 = self.imf[:,0] * 3
        imf2 = sift( x2, sd_thresh=1e-3,interp_method='splrep' )

        tst = self.check_diff( self.get_resid(self.imf[:,0]*3,imf2[:,0]), 1 )
        assert( tst )

    def test_sift_plus_constant( self ):
        """Test that sifting a signal plus a constant only changes the last IMF"""
        x3 = self.x + 2
        imf3 = sift( x3, interp_method='splrep' )

        tst = list()
        for ii in range(imf3.shape[1]):
            tmp = self.imf[:,ii]
            if ii == self.imf.shape[1]-1:
                tmp = tmp+2
            tst.append( self.check_diff( self.get_resid(tmp,imf3[:,ii]),1 ) )

        assert( all(tst) is True )

    def test_sift_of_imf( self ):
        """Test that sifting an IMF returns the IMF"""
        x4 = self.imf[:,0].copy()
        imf4 = sift( x4, interp_method='splrep' )

        tst = list()
        for ii in range(imf4.shape[1]):
            if ii == 0:
                target = 1
            else:
                target = 0
            tst.append( self.check_diff( self.get_resid(x4,imf4[:,ii]), target) )

        assert( all(tst) is True )

    def test_sift_of_reversed_signal( self ):
        """Test that sifting a reversed signal only reverses the IMFs"""
        x5 = self.x[::-1]
        imf5 = sift( x5, interp_method='splrep' )

        tst = self.check_diff( self.get_resid( self.imf[::-1,0], imf5[:,0] ),1)

        assert( tst )
