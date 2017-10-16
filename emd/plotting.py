
import numpy as np
import matplotlib.pyplot as plt

def plot_imfs( imfs, time_vect=None, scale_y=False ):

    nplots = imfs.shape[1] + 1
    if time_vect is None:
        time_vect = np.arange(imfs.shape[0])

    mx = np.abs(imfs).max()

    plt.figure()
    ax = plt.subplot(nplots,1,1)
    ax.plot( time_vect, imfs.sum(axis=1), 'k' )
    ax.tick_params( axis='x', labelbottom='off')

    for ii in range(1,nplots):
        ax = plt.subplot(nplots,1,ii+1)
        ax.plot( time_vect, imfs[:,ii-1],'k' )
        if scale_y:
            ax.set_ylim( -mx*1.2, mx*1.2 )

        if ii < nplots:
           ax.tick_params( axis='x', labelbottom='off')
