import re
import unicodedata
from typing import List, Mapping, Callable, Any, Union, Iterable

from boba_python_utils.common_utils import process
from boba_python_utils.string_utils.common import strip_


def remove_accents(s: str) -> str:
    nfkd_form = unicodedata.normalize('NFKD', s)
    only_ascii = nfkd_form.encode('ASCII', 'ignore')
    return only_ascii.decode('utf-8')


def remove_trailing_bracketed_strings(s: str) -> str:
    """
    Remove all trailing bracketed contents from the input string.

    This function removes all trailing bracketed contents from the input string using a single
    regular expression pattern. It can handle multiple consecutive bracketed contents.

    Args:
        s (str): The input string from which to remove trailing bracketed contents.

    Returns:
        str: The input string with all trailing bracketed contents removed.

    Examples:
        >>> remove_trailing_bracketed_strings("Example string (2021) (v1.0) (beta)")
        'Example string'
        >>> remove_trailing_bracketed_strings("Another example (123) (ABC)")
        'Another example'
        >>> remove_trailing_bracketed_strings("Another example (123)")
        'Another example'
        >>> remove_trailing_bracketed_strings("No brackets here")
        'No brackets here'
    """
    year_pattern = r'(\s\([^()]*\))+$'
    return re.sub(year_pattern, '', s).strip()


def extract_trailing_bracketed_strings(s: str) -> List[str]:
    """
    Extract all trailing bracketed contents from the input string.

    This function extracts all trailing bracketed contents from the input string using a single
    regular expression pattern. It can handle multiple consecutive bracketed contents.

    Args:
        s: The input string from which to extract trailing bracketed contents.

    Returns: A list of extracted trailing bracketed contents, in the order they appear.

    Examples:
        >>> extract_trailing_bracketed_strings("Example string (2021) (v1.0) (beta)")
        ['2021', 'v1.0', 'beta']
        >>> extract_trailing_bracketed_strings("Another example (123) (ABC)")
        ['123', 'ABC']
        >>> extract_trailing_bracketed_strings("No brackets here")
        []
    """
    pattern = r'(?<=\s)\([^()]*\)$'
    bracketed_strings = []
    s = s.strip()

    while re.search(pattern, s):
        match = re.search(pattern, s)
        bracketed_string = match.group().strip()[1:-1]
        if bracketed_string:
            bracketed_strings.append(bracketed_string)
        start, end = match.span()
        s = s[:start].rstrip()

    return list(reversed(bracketed_strings))


# region common string processing

def process_string(s: str, processors: Mapping = None, **kwargs) -> Union[str, Any]:
    """
    Apply string processing functions to the input string.

    Args:
        s (str): The input string to be processed.
        processors (Mapping, optional): A dictionary mapping processor names to functions. Defaults to None.
        **kwargs: Keyword arguments where the key is the processor name and the value is either True, False,
            a dictionary of processor arguments, or directly the arguments.

    Returns:
        Union[str, Any]: The processed string or result of processing.

    Example:
        >>> s = "   Hello, World!   "
        >>> process_string(s, strip=True)
        'Hello, World!'
        >>> process_string(s, strip=True, lower=True)
        'hello, world!'
        >>> s = "   Hello, World!   xxx"
        >>> process_string(s, rstrip=' x')
        '   Hello, World!'
        >>> process_string(s, rstrip=' x', extract_between={'search1': 'll', 'search2': 'or'})
        'o, W'
        >>> process_string(s, isdigit=True)
        False
        >>> s = "zgchen-pod-stjdp                                                        1/1     Running                  0               9d"
        >>> process_string(s, split=True)
        ['zgchen-pod-stjdp', '1/1', 'Running', '0', '9d']
        >>> fields=['name', 'ready', 'status', 'restarts', 'age']
        >>> process_string(s, split=True, zip=[fields, '#output'], dict=True)
        {'name': 'zgchen-pod-stjdp', 'ready': '1/1', 'status': 'Running', 'restarts': '0', 'age': '9d'}
    """
    import boba_python_utils.string_utils as str_utils
    return process(
        obj=s,
        modules=[str, str_utils],
        processors=processors,
        **kwargs
    )



def process_lines(
        line_iter: Union[str, Iterable[str]],
        processors: Mapping = None,
        ignore_empty_lines: bool = True,
        ignore_empty_output: bool = True,
        **kwargs
) -> Union[str, Any]:
    for line in line_iter.split('\n') if isinstance(line_iter, str) else line_iter:
        if not line and ignore_empty_lines:
            continue
        item = process_string(line, processors=processors, **kwargs)
        if not item and ignore_empty_output:
            continue
        yield item
# endregion
