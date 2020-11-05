import unittest

import logging

from ..logger import set_up, set_level, get_level, is_active, set_format


class test_logger(unittest.TestCase):

    def test_0_setup(self):
        """ Need zero in test name to ensure it is run first"""

        logger = logging.getLogger('emd')

        # Check logger inactive
        assert(is_active() is False)

        # By default should have one handler
        assert(len(logger.handlers) == 1)
        # ...which is Null
        assert(isinstance(logger.handlers[0], logging.NullHandler))

        set_up()

        # Check logger now active
        assert(is_active())

        # By default should have one handler
        assert(len(logger.handlers) == 1)
        # ...which is a stream handler
        assert(isinstance(logger.handlers[0], logging.StreamHandler))

    def test_logger_level_change(self):

        set_level('WARNING')
        assert(get_level() == 30)

        set_level('DEBUG')
        assert(get_level() == 10)

    def test_logger_format_change(self):

        # Can't directly introspect current formatter so just check we don't
        # get an error
        try:
            set_format(formatter='brief', handler_name='console')
        except Exception:
            self.fail("emd.logger.set_format() raised an Exception")

    def test_log_to_file(self):
        logger = logging.getLogger('emd')

        import tempfile
        test_log = tempfile.NamedTemporaryFile(prefix="EMD_TestLogFile").name

        set_up(log_file=test_log)

        # By default should have two handlers
        assert(len(logger.handlers) == 2)
        # ...first is a stream handler
        assert(isinstance(logger.handlers[0], logging.StreamHandler))
        # ...second is a file handler
        assert(isinstance(logger.handlers[1], logging.FileHandler))
