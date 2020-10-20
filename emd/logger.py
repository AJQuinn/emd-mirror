#!/usr/bin/python

# vim: set expandtab ts=4 sw=4:

import sys
import yaml
import numpy as np
from functools import wraps
from .support import get_install_dir, get_installed_version

# Housekeeping for logging
import logging
import logging.config

# Add a single null handler until set-up is called, this is activated on import
# to __init__
logging.getLogger('emd').addHandler(logging.NullHandler())

# Initialise logging for this sub-module
logger = logging.getLogger(__name__)


default_config = """
version: 1
loggers:
  emd:
    level: DEBUG
    handlers: [console, file]
    propagate: false

handlers:
  console:
    class : logging.StreamHandler
    formatter: default
    level   : INFO
    stream  : ext://sys.stdout
  file:
    class : logging.handlers.RotatingFileHandler
    formatter: verbose
    filename: {log_file}
    backupCount: 3
    maxBytes: 102400

formatters:
  brief:
    format: '{prefix} %(message)s'
  default:
    format: '[%(asctime)s] {prefix} %(levelname)-8s %(funcName)20s : %(message)s'
    datefmt: '%H:%M:%S'
  verbose:
    format: '[%(asctime)s] {prefix} - %(levelname)s - emd.%(module)s:%(lineno)s - %(funcName)20s() : %(message)s'
    datefmt: '%Y-%m-%d %H:%M:%S'

disable_existing_loggers: true

"""


def set_up(prefix='', log_file=''):
    """
    Initialisation for the EMD module logger.

    Parameters
    ----------
    prefix : str
        Optional prefix to attach to logger output
    log_file : str
        Optional path to a log file to record logger output
    """

    # Format config with user options
    new_config = default_config.format(prefix=prefix, log_file=log_file)
    # Load config to dict
    new_config = yaml.load(new_config, Loader=yaml.FullLoader)

    # Remove log file from dict if not user requested
    if len(log_file) == 0:
        new_config['loggers']['emd']['handlers'] = ['console']
        del new_config['handlers']['file']

    # Configure logger with dict
    logging.config.dictConfig(new_config)

    # Print some info
    logger.info('EMD Logger Started')
    if len(log_file) > 0:
        logger.info('logging to file: {0}'.format(log_file))
    logger.debug('EMD v{0} installed in {1}'.format(get_installed_version(),
                                                    get_install_dir()))


def set_level(level, handler='console'):
    """Set new logging level for EMD module."""
    logger = logging.getLogger('emd')
    for handler in logger.handlers:
        if handler.get_name() == 'console':
            handler.setLevel(getattr(logging, level))


def get_level(handler='console'):
    """Return current logging level for EMD module."""
    logger = logging.getLogger('emd')
    for handler in logger.handlers:
        if handler.get_name() == 'console':
            return handler.level


def set_format(formatter='', handler_name='console', prefix=''):
    """Set new formatter EMD module logger."""
    logger = logging.getLogger('emd')
    new_config = yaml.load(default_config, Loader=yaml.FullLoader)
    try:
        fmtstr = new_config['formatters'][formatter]['format']
    except KeyError:
        logger.warning("EMD logger format type '{0}' not recognised".format(formatter))
        raise KeyError("EMD logger format type '{0}' not recognised".format(formatter))
    fmt = logging.Formatter(fmtstr.format(prefix=prefix))
    for handler in logger.handlers:
        if handler.get_name() == handler_name:
            handler.setFormatter(fmt)
    logger.info('EMD logger: handler {0} format changed to {1}'.format(handler_name, formatter))


def disable():
    """Turn off logging for the EMD module."""
    logger = logging.getLogger('emd')
    logger.info('EMD logging disabled')
    logging.disable(sys.maxsize)


def enable():
    """Turn on logging for the EMD module."""
    logger = logging.getLogger('emd')
    logging.disable(logging.NOTSET)
    logger.info('EMD logging enabled')


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


# Decorator for logging sift function
def wrap_verbose(func):
    # This is the actual decorator
    @wraps(func)
    def inner_verbose(*args, **kwargs):

        if 'verbose' in kwargs:
            tmp_level = kwargs['verbose']
            current_level = get_level()
            set_level(level=tmp_level)

        # Call function itself
        func_output = func(*args, **kwargs)

        if 'verbose' in kwargs:
            set_level(level=logging._levelToName[current_level])

        return func_output
    return inner_verbose
