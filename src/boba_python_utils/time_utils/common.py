from datetime import datetime, timedelta
from time import time
from typing import Union, Mapping, List, Tuple
import re
from boba_python_utils.common_utils import solve_obj
from time import sleep
import random


def random_sleep(min_sleep, max_sleep):
    sleep(random.uniform(min_sleep, max_sleep))


def timestamp(scale=100) -> str:
    return str(int(time() * scale))


def solve_datetime(
        _dt_obj: Union[datetime, str, Mapping, List, Tuple],
        datetime_str_format: str = '%m/%d/%Y',
        delta: Union[int, Mapping, List, Tuple, timedelta] = None,
) -> datetime:
    """
    Solves the input object as a `datetime` object.
        For example, a string '10/30/2021' (with `datetime_str_format` '%m/%d/%Y'),
        or a dictionary `{"year": 2021, "month": 10, "day":30}`, or a tuple `(2021, 10, 30)`,
        will be solved as `datetime(year=2021, month=10, day=30)`.
    Args:
        _dt_obj: the input object to solve as a `datetime`; current support
            1. a string (format defined by the other parameter `datetime_str_format`);
            2. an iterable (e.g. tuple, list) that can be used as
        datetime_str_format: specifies a string format to solve a string object as a `datetime`.
        delta: adds this timedelta to the created `datetime` object;
            can specify an integer, which is `timedelta` in days;
            can specify a dictionary `{"days":7, "hours":3}`
                with keys being the parameter names of the `timedelta` object.

    Returns: the created `datetime` object solved from the input and optionally the time delta.

    """
    _dt_obj = solve_obj(
        _dt_obj, obj_type=datetime, str2obj=datetime.strptime, str_format=datetime_str_format
    )
    if delta is None:
        return _dt_obj
    else:
        return _dt_obj + solve_obj(delta, obj_type=timedelta)


def solve_date_time_format_by_granularity(
        granularity: str,
        date_format: str = '%Y%m%d',
        time_format: str = '%H%M%S'
) -> Tuple[str, str]:
    """
    This function solves the date and time format based on the given granularity.
    Args:
        granularity: Can be one of "year", "month", "day", "hour", "minute", "second".
        date_format: If provided, will be used as the base date format.
        time_format: If provided, will be used as the base time format.
    Returns:
        Tuple[str, str]: The solved date and time format.
    Examples:
        >>> solve_date_time_format_by_granularity('year', '%Y-%m-%d', '%H:%M:%S')
        ('%Y', '')
        >>> solve_date_time_format_by_granularity('month', '%Y-%m-%d', '%H:%M:%S')
        ('%Y-%m', '')
        >>> solve_date_time_format_by_granularity('day', '%Y-%m-%d', '%H:%M:%S')
        ('%Y-%m-%d', '')
        >>> solve_date_time_format_by_granularity('hour', '%Y-%m-%d', '%H:%M:%S')
        ('%Y-%m-%d', '%H')
        >>> solve_date_time_format_by_granularity('minute', '%Y-%m-%d', '%H:%M:%S')
        ('%Y-%m-%d', '%H:%M')
        >>> solve_date_time_format_by_granularity('second', '%Y-%m-%d', '%H:%M:%S')
        ('%Y-%m-%d', '%H:%M:%S')
        >>> solve_date_time_format_by_granularity('day', '%d/%m/%Y', '%H-%M-%S')
        ('%d/%m/%Y', '')
        >>> solve_date_time_format_by_granularity('month', '%d/%m/%Y', '%H-%M-%S')
        ('%m/%Y', '')
        >>> solve_date_time_format_by_granularity('year', '%d/%m/%Y', '%H-%M-%S')
        ('%Y', '')
        >>> solve_date_time_format_by_granularity('minute', '%d/%m/%Y', '%M/%H/%S')
        ('%d/%m/%Y', '%M/%H')
    """

    # Define the formats for date and time elements
    formats = ['year', 'month', 'day', 'hour', 'minute', 'second']
    format_codes = {
        'year': '%Y',
        'month': '%m',
        'day': '%d',
        'hour': '%H',
        'minute': '%M',
        'second': '%S'
    }

    # Identify the separators by replacing format elements with empty string
    date_separator = date_format
    time_separator = time_format
    for code in format_codes.values():
        date_separator = date_separator.replace(code, '')
        time_separator = time_separator.replace(code, '')

    # Initialize the new format strings
    new_date_format = date_format
    new_time_format = time_format

    # Iterate over the formats in order
    for format in formats:
        if format_codes[format] in date_format and formats.index(format) > formats.index(granularity):
            new_date_format = new_date_format.replace(format_codes[format], '')
        if format_codes[format] in time_format and formats.index(format) > formats.index(granularity):
            new_time_format = new_time_format.replace(format_codes[format], '')

    # Remove extra separators
    new_date_format = new_date_format.strip(date_separator).replace(date_separator * 2, date_separator)
    new_time_format = new_time_format.strip(time_separator).replace(time_separator * 2, time_separator)

    return new_date_format, new_time_format
