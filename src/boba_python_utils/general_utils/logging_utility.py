import logging
import warnings
from os import path
from typing import Union

from numpy import iterable
from boba_python_utils.general_utils.console_util import (
    hprint_message, get_pairs_str_for_hprint_and_regular_print, eprint_message, hprint, eprint
)


def get_file_logger(
        name: str,
        log_dir_path: str = '.',
        logging_level=logging.DEBUG,
        log_format="%(asctime)s - %(process)d - %(name)s - %(levelname)s - %(message)s",
        append=False,
        file_ext='log'
):
    """
    Gets a file-based logger.

    Args:
        name: the name of the logger; also used as the main name for the log file.
        log_dir_path: the path to the directory containing the log file;
            if the log file does not exist, a new file will be created;
            if the log file already exists, then it is overwritten if `append` is `False`,
                or new log lines are appended to the existing file if `append` is set `True`.
        logging_level: provides the logging level; the default is the lowest level 'logging.DEBUG'.
        log_format: the format for each logging message;
            by default it includes time, process id, logger name, logging level and the message;
            check https://docs.python.org/3/library/logging.html#logrecord-attributes
                for more about logging format directives.
        append: True if appending new log lines to the log file;
            False if the existing log file should be overwritten.
        file_ext: the extension name for the log file.

    Returns:
        a file-based logger.

    """
    from IPython.utils.path import ensure_dir_exists
    logger = logging.getLogger(name)
    ensure_dir_exists(log_dir_path)
    handler = logging.FileHandler(
        path.join(
            log_dir_path,
            f'{name}.{file_ext if file_ext else "log"}'), 'a+' if append else 'w+'
    )
    handler.setFormatter(logging.Formatter(log_format))
    logger.addHandler(handler)
    logger.setLevel(logging_level)
    return logger


class LoggableBase:
    """
    A base class that wraps a logger and provides convenient common logging methods.
    """

    def __init__(
            self,
            logger: Union[logging.Logger, str] = None,
            logger_flush_interval=20,
            print_out=__debug__,
            color_print=True,
            *args,
            **kwargs
    ):
        """
        Args:
            logger: provides a logger instance or the path to a log file.
                if a string is provided, then we assume it is a file path
                and a file-based logger will be automatically created.
            logger_flush_interval: specifies the number of log lines between two flush operations.
            print_out: True if printing out the logging messages on the terminal; otherwise, False.
            color_print: True if to enable colorful print out on the terminal; otherwise, False.
        """
        if isinstance(logger, str):
            main_name, ext_name = path.splitext(path.basename(logger))
            self.logger = get_file_logger(
                name=main_name,
                log_dir_path=path.dirname(logger),
                file_ext=ext_name
            )
        else:
            self.logger = logger
        self._has_logger = logger is not None
        self._print_out = print_out
        self._color_print = color_print
        if color_print:
            self._debug_print = self._info_print = hprint
            self._debug_print_with_title = self._info_print_with_title = hprint_message
            self._error_print = eprint
            self._error_print_with_title = eprint_message
        else:
            self._debug_print = self._info_print = \
                self._error_print = self._debug_print_with_title = \
                self._info_print_with_title = self._error_print_with_title = print
        self._warn_print = warnings.warn  # warn print is always `warnings.warn`

        if self._has_logger:
            self._log_lines_count = 0
            self._logger_flush_interval = logger_flush_interval

    def log_flush(self):
        if self._has_logger and self._log_lines_count >= self._logger_flush_interval:
            for handler in self.logger.handlers:
                if hasattr(handler, 'flush'):
                    handler.flush()

    def _log_with_print(self, log_method, print_method, msg: str, print_out=False):
        log_method(msg)
        if (self._print_out or print_out) and msg:
            print_method(msg)
        self._log_lines_count += 1
        self.log_flush()

    def _log_titled_msg(self, log_method, print_method, title: str, content: str, print_out=False):
        if self._has_logger:
            log_method(f"{title}: {content}")
            self._log_lines_count += 1
            self.log_flush()

        if self._print_out or print_out:
            print_method(title, content)

    def _log_pairs(self, *pairs, log_method, pair_msg_gen_method, print_out=False):
        colored_msg, uncolored_msg = pair_msg_gen_method(*pairs)
        if self._has_logger:
            log_method(uncolored_msg)
            self._log_lines_count += 1
            self.log_flush()

        if self._print_out or print_out:
            print(colored_msg if self._color_print else uncolored_msg)

    def _log_multiple_with_print(self, log_method, print_method, *msgs, print_out=False):
        msg_count = len(msgs)
        if self._has_logger:
            for i in range(msg_count):
                msg = msgs[i]
                log_method(msg)
                if self._print_out and msg:
                    print_method(msg)
            self._log_lines_count += msg_count
            self.log_flush()
        elif self._print_out or print_out:
            for i in range(msg_count):
                msg = msgs[i]
                if msg:
                    print_method(msg)

    # region error
    def error(self, msg: str, print_out=True):
        """
        Logs one logging.ERROR level message.

        Args:
            msg: the message to log.
            print_out: True to print out the message on terminal.
        """
        if self._has_logger:
            self._log_with_print(self.logger.error, self._error_print, msg, print_out=print_out)

    def error_message(self, title, msg, print_out=True):
        self._log_titled_msg(
            log_method=self.logger.info,
            print_method=self._error_print_with_title,
            title=title,
            content=msg,
            print_out=print_out
        )

    def error_multiple(self, *msgs, print_out=True):
        """
        Logs multiple logging.ERROR level messages.
        msgs: the messages to log.
        """
        if self._has_logger:
            self._log_multiple_with_print(
                self.logger.error,
                self._error_print,
                *msgs,
                print_out=print_out
            )

    # endregion

    # region debug
    def debug(self, msg: str, print_out=False):
        """
        Logs one logging.DEBUG level message.

        Args:
            msg: the message to log.
            print_out: True to print out the message on terminal.
        """
        if self._has_logger:
            self._log_with_print(self.logger.debug, self._debug_print, msg, print_out=print_out)

    def debug_multiple(self, *msgs, print_out=False):
        """
        Logs multiple logging.DEBUG level messages.
        msgs: the messages to log.
        """
        if self._has_logger:
            self._log_multiple_with_print(self.logger.debug if self._has_logger else None, self._debug_print, *msgs, print_out=print_out)

    def debug_pairs(self, *pairs, print_out=False):
        self._log_pairs(
            *pairs,
            log_method=self.logger.debug if self._has_logger else None,
            pair_msg_gen_method=get_pairs_str_for_hprint_and_regular_print,
            print_out=print_out
        )

    def debug_message(self, title, msg, print_out=False):
        if self._has_logger:
            self._log_titled_msg(
                log_method=self.logger.debug if self._has_logger else None,
                print_method=self._debug_print_with_title,
                title=title,
                content=msg,
                print_out=print_out
            )

    # endregion

    # region warning
    def warning(self, msg: str, print_out=True, print_method=None):
        """
        Logs one logging.WARNING level message.
        msg: the message to log.
        """
        if self._has_logger:
            self._log_with_print(self.logger.warning, print_method if print_method else self._warn_print, msg, print_out=print_out)

    def warning_multiple(self, *msgs, print_out=True, print_method=None):
        """
        Logs multiple logging.WARNING level messages.
        msgs: the messages to log.
        """
        if self._has_logger:
            self._log_multiple_with_print(self.logger.warning if self._has_logger else None, print_method if print_method else self._warn_print, *msgs, print_out=print_out)

    # endregion

    # region info
    def info(self, msg: str, print_out=True, print_method=None):
        """
        Logs one logging.INFO level message.
        msg: the message to log.
        """
        if self._has_logger:
            self._log_with_print(self.logger.info if self._has_logger else None, print_method if print_method else self._info_print, msg, print_out=print_out)

    def info_multiple(self, *msgs, print_out=True, print_method=None):
        """
        Logs multiple logging.INFO level messages.
        msgs: the messages to log.
        """
        if self._has_logger:
            self._log_multiple_with_print(self.logger.info, print_method if print_method else self._info_print, *msgs, print_out=print_out)

    def info_message(self, title, msg, print_out=True):
        self._log_titled_msg(
            log_method=self.logger.info if self._has_logger else None,
            print_method=self._info_print_with_title,
            title=title,
            content=msg,
            print_out=print_out
        )

    def info_pairs(self, *pairs, print_out=True, pairs_str_gen_method=get_pairs_str_for_hprint_and_regular_print):
        self._log_pairs(*pairs, log_method=self.logger.info if self._has_logger else None, pair_msg_gen_method=pairs_str_gen_method, print_out=print_out)
    # endregion
