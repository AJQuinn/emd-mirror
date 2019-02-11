import logging
import numpy as np
from functools import wraps

def set_up(level='DEBUG'):
    """
    LEVELS = [DEBUG, INFO, NOTIFY, WARN, ERROR, FATAL]
    """
    logging.basicConfig(level=getattr(logging,level),
                        format='%(asctime)s %(name)s %(levelname)-8s %(message)s',
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
            logging.info('STARTED: {0}'.format(sift_name))

            if (sift_name is 'ensemble_sift') or \
               (sift_name is 'complete_ensemble_sift'):
                # Print number of ensembles if ensemble sift - this is a
                # positional arg not a kwarg
                logging.debug('Input data size: {0}'.format(args[0].shape))
                logging.debug('Computing {0} ensembles'.format(args[1]))
            else:
                logging.debug('Input data size: {0}'.format(args[0].shape))

            # Print main keyword arguments
            logging.debug('Input Sift Args: {0}'.format(kwargs))

            # Call function itself
            func_output = func(*args,**kwargs)

            # Print number of IMFs, catching other outputs if they're returned
            # as well
            if isinstance(func_output,np.ndarray):
                logging.debug('Returning {0} imfs'.format(func_output.shape[1]))
            else:
                logging.debug('Returning {0} imfs'.format(func_output[0].shape[1]))

            # Close function
            logging.info('COMPLETED: {0}'.format(sift_name))
            return func_output
        return wrapper
    return add_logger
