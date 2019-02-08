import logging


def set_up(level='DEBUG'):
    """
    LEVELS = [DEBUG, INFO, NOTIFY, WARN, ERROR, FATAL]
    """

    logging.basicConfig(level=getattr(logging,level),
                        format='%(asctime)s %(name)s %(levelname)-8s %(message)s',
                        datefmt='%m-%d %H:%M:%S')

    logging.info('Logging Started\n')


