import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate as interp
from scipy import signal
from . import utils


if __name__ == '__main__':

    from .plotting import plot_imfs
    from .frequency_transforms import instantaneous_stats,hilberthuang

    sample_rate = 1000
    seconds = 10
    num_samples = seconds*sample_rate
    t = np.linspace(0,seconds,num_samples)
    #X = np.sin( 2*np.pi*4*t) + np.sin( 2*np.pi*14*t) + np.random.randn(num_samples,)*.02# + np.linspace(-1,1,num_samples)
    #X = np.r_[signal.chirp( t[:5000], 5, 5, 25 ), signal.chirp( t[5000:],25,10,5)]
    #X =  np.sin( 2*np.pi*75*t ) + .8*np.sin( 2*np.pi*73*t) + .8*np.sin( 2*np.pi*77*t )
    #X +=  np.random.randn(num_samples,)*.2

    #X = (np.sin( 2*np.pi*5*t )+1) ** 2 + np.random.randn(num_samples,)*.1

    #X += np.linspace(-1,1,num_samples)

    #X = np.loadtxt('/Users/andrew/Projects/emd/cetdl_data.txt')
    #X = X[:5000]
    #t = np.arange(X.shape[0])

    X = abreu2010( 10, .75, -np.pi/4.,sample_rate, seconds ) + np.random.randn( num_samples, )*.02

    imf = sift( X )
    imf2 = ensemble_sift2( X, t, nensembles=100, ensemble_noise=.1)
    #imf2 = sift_second_layer( imf )

    infr, inam = instantaneous_stats( imf, sample_rate, 'hilbert' )
    #infr2, inam2 = instantaneous_stats( imf2, sample_rate, 'hilbert' )

    #holo = holospectrum_am( infr, infr2, inam2, np.linspace(0,100,50),np.linspace(0,5,5) )

    plot_imfs(imf, t)


    hht = hilberthuang( infr, inam, np.linspace(0,100,100) )

    plt.figure()
    plt.contourf(t[:1500],np.linspace(0,100,101),hht[:1500,:].T )

    #plt.show()
