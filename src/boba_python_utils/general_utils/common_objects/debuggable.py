import logging
from abc import ABC

from attr import attrs, attrib

@attrs(slots=False)
class Debuggable(ABC):
    """
    Attributes:
        debug_mode: True to indicate the module should run in the debug mode;
            there will be extra error checking and logging in the debug mode.
        logger: the logger object; can provide a customized logger;
            otherwise a default logger will be used.
    """
    _debug_mode = attrib(type=bool, default=False, kw_only=True)
    _logger = attrib(type=logging.Logger, default=None, kw_only=True)
