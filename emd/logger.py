import logging
import numpy as np
from functools import wraps

import logging
logger = logging.getLogger(__name__)

def set_up(level='DEBUG'):
    """
    LEVELS = [DEBUG, INFO, NOTIFY, WARN, ERROR, FATAL]
    """
    fmt = '[%(filename)s:%(lineno)s - %(funcName)20s()'

    fmt = '%(asctime)s emd.%(module)s:%(lineno)s - %(funcName)15s() : %(message)s'
    logging.basicConfig(level=getattr(logging,level),
                        format=fmt,
                        datefmt='%m-%d %H:%M:%S')
    logging.info('Logging Started on {0}'.format(level))

def set_level(level='INFO'):
    """
    LEVELS = [DEBUG, INFO, NOTIFY, WARN, ERROR, FATAL]
    """
    logging.getLogger().setLevel(getattr(logging,level))
    logging.info('Logging Level changed to {0}'.format(level))


##------------------------------------

## Decorator for logging sift function
def sift_logger(sift_name):
    # This first layer is a wrapper func to allow an argument to be passed in.
    # If we don't do this then we can't easily tell which function is being
    # decorated
    def add_logger(func):
        # This is the actual decorator
        @wraps(func)
        def wrapper(*args,**kwargs):
            logger.info('STARTED: {0}'.format(sift_name))

            if (sift_name is 'ensemble_sift') or \
               (sift_name is 'complete_ensemble_sift'):
                # Print number of ensembles if ensemble sift - this is a
                # positional arg not a kwarg
                logger.debug('Input data size: {0}'.format(args[0].shape))
                logger.debug('Computing {0} ensembles'.format(args[1]))
            else:
                logger.debug('Input data size: {0}'.format(args[0].shape))

            # Print main keyword arguments
            logger.debug('Input Sift Args: {0}'.format(kwargs))

            # Call function itself
            func_output = func(*args,**kwargs)

            # Print number of IMFs, catching other outputs if they're returned
            # as well
            if isinstance(func_output,np.ndarray):
                logger.debug('Returning {0} imfs'.format(func_output.shape[1]))
            else:
                logger.debug('Returning {0} imfs'.format(func_output[0].shape[1]))

            # Close function
            logger.info('COMPLETED: {0}'.format(sift_name))
            return func_output
        return wrapper
    return add_logger
