import sys
import logging
from time import gmtime, strftime
from colorlog import ColoredFormatter

def create_logger(name, silent=False, to_disk=False, log_file=None, prefix=None):
    """Logger wrapper
    by xiaodl
    """
    # setup logger
    log = logging.getLogger(name)
    log.setLevel(logging.DEBUG)
    formatter = ColoredFormatter(
        "%(asctime)s %(log_color)s%(levelname)-8s%(reset)s [%(blue)s%(message)s%(reset)s]",
        datefmt='%Y-%m-%d %I:%M:%S',
        reset=True,
        log_colors={
            'DEBUG':    'cyan',
            'INFO':     'green',
            'WARNING':  'yellow',
            'ERROR':    'red',
            'CRITICAL': 'red,bg_white',
        },
        secondary_log_colors={},
        style='%'
    )
    fformatter = logging.Formatter(
        "%(asctime)s [%(funcName)-12s] %(levelname)-8s [%(message)s]",
        datefmt='%Y-%m-%d %I:%M:%S',
        style='%'
    )
    if not silent:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        log.addHandler(ch)
    if to_disk:
        prefix = prefix if prefix is not None else 'my_log'
        log_file = log_file if log_file is not None else strftime('{}-%Y-%m-%d-%H-%M-%S.log'.format(prefix), gmtime())
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fformatter)
        log.addHandler(fh)
    # disable elmo info
    log.propagate = False
    return log
