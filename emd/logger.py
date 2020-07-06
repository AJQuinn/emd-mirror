#!/usr/bin/python

# vim: set expandtab ts=4 sw=4:

import numpy as np
from functools import wraps

import logging
logger = logging.getLogger(__name__)


def set_up(level='DEBUG', filename=None, mode='both'):
    """
    LEVELS = [DEBUG, INFO, WARN, ERROR, FATAL]
    """
    fmt = '[%(filename)s:%(lineno)s - %(funcName)20s()'

    fmt = '%(asctime)s emd.%(module)s:%(lineno)s - %(funcName)20s() : %(message)s'

    if (mode == 'console') or (mode == 'both'):
        # Start logging to console
        logging.basicConfig(level=getattr(logging, level),
                            format=fmt,
                            datefmt='%m-%d %H:%M:%S')

    if (filename is not None) and (mode == 'both'):
        # Add file handler to existing console logger
        filelog = logging.FileHandler(filename)
        filelog.setLevel(level)
        filelog.setFormatter(logging.Formatter(fmt))
        # add the handler to the root logger
        logging.getLogger('').addHandler(filelog)

    if (filename is not None) and (mode == 'file'):
        # Start logging to file
        logging.basicConfig(level=getattr(logging, level),
                            format=fmt, filename=filename,
                            datefmt='%m-%d %H:%M:%S')

    logging.info('Logging Started on {0}'.format(level))
    logging.info('Logging mode \'{0}\''.format(mode))
    if (filename is not None) and (mode == 'file') or (mode == 'both'):
        logging.info('Logging to file: {0}'.format(filename))


def set_level(level='INFO'):
    """
    LEVELS = [DEBUG, INFO, WARN, ERROR, FATAL]
    """
    logging.getLogger().setLevel(getattr(logging, level))
    logging.info('Logging Level changed to {0}'.format(level))


def shut_down():
    """
    Function for disabling logging in EMD

    Note, this  doesn't remove the logger altogether, but sets the logging
    level extremely high so no output ever appears.
    """

    logging.info('EMD logging shutting down')
    logging.getLogger().setLevel(100)


# ------------------------------------

# Decorator for logging sift function
def sift_logger(sift_name):
    # This first layer is a wrapper func to allow an argument to be passed in.
    # If we don't do this then we can't easily tell which function is being
    # decorated
    def add_logger(func):
        # This is the actual decorator
        @wraps(func)
        def sift_logger(*args, **kwargs):
            logger.info('STARTED: {0}'.format(sift_name))

            if (sift_name == 'ensemble_sift') or \
               (sift_name == 'complete_ensemble_sift'):
                # Print number of ensembles if ensemble sift
                logger.debug('Input data size: {0}'.format(args[0].shape))
                if 'nensembles' in kwargs:
                    logger.debug('Computing {0} ensembles'.format(kwargs['nensembles']))
                else:
                    logger.debug('Computing 4 ensembles (default)')
            else:
                logger.debug('Input data size: {0}'.format(args[0].shape[0]))

            # Print main keyword arguments
            logger.debug('Input Sift Args: {0}'.format(kwargs))

            # Call function itself
            func_output = func(*args, **kwargs)

            # Print number of IMFs, catching other outputs if they're returned
            # as well
            if isinstance(func_output, np.ndarray):
                logger.debug('Returning {0} imfs'.format(func_output.shape[1]))
            else:
                logger.debug('Returning {0} imfs'.format(func_output[0].shape[1]))

            # Close function
            logger.info('COMPLETED: {0}'.format(sift_name))
            return func_output
        return sift_logger
    return add_logger
