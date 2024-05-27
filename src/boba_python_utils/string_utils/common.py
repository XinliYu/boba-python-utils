from enum import Enum
from typing import Iterable, Optional, Union, Tuple

from boba_python_utils.common_utils.iter_helper import iter__
from boba_python_utils.common_utils.typing_helper import solve_nested_singleton_tuple_list


class OccurrenceOptions(str, Enum):
    First = 'first'
    Last = 'last'
    All = 'all'


def startswith_any(s: str, targets: Union[str, Iterable[str]]):
    """
    Checks if a string `s` starts with any of the target substrings specified
    in the `targets` iterable.
    """
    return any(not target or s.startswith(target) for target in iter__(targets))


def endswith_any(s: str, targets: Union[str, Iterable[str]]):
    """
    Checks if a string `s` ends with any of the target substrings specified
    in the `targets` iterable.
    """
    return any(not target or s.endswith(target) for target in iter__(targets))


def contains_any(s: str, targets: Iterable[str], ignore_empty: bool = True) -> bool:
    """
    Checks if a string `s` contains any of the target substrings specified in the `targets` iterable.

    If the `ignore_empty` parameter is set to True (which is the default),
    the function will ignore any empty or None values in the targets iterable
    when checking for substrings. If `ignore_empty` is set to False,
    the function will include empty or None values in the check.

    The function returns True if any of the target substrings are found in the string s,
    and False otherwise.

    """
    if ignore_empty:
        return any(target in s for target in iter__(targets) if target)
    else:
        return any(target in s for target in iter__(targets) if target is not None)


def contains_all(s: str, targets: Iterable[str]) -> bool:
    return all(target in s for target in iter__(targets) if target)


def contains_any_all(s: str, targets: Iterable[Union[Iterable[str], str]]):
    return any(
        (contains_all(s, target)) for target in iter__(targets) if target
    )


def find_all(s: str, substr: str) -> Iterable[int]:
    i = s.find(substr)
    while i != -1:
        yield i
        i = s.find(substr, i + 1)


def join_(*strs, sep: str = '', ignore_none_or_empty=True) -> str:
    strs = solve_nested_singleton_tuple_list(strs)
    if len(strs) == 1:
        if strs[0] is not None:
            return f'{strs[0]}'
        else:
            return ''
    else:
        if ignore_none_or_empty:
            return sep.join((
                f'{x}' for x in strs
                if x is not None and x != ''
            ))
        else:
            return sep.join((
                ('' if x is None else f'{x}') for x in strs
            ))


def count_uppercase(s):
    return sum(1 for c in s if c.isupper())


def count_lowercase(s):
    return sum(1 for c in s if c.islower())


def cut(s: str, cut_before=None, cut_after=None):
    if cut_before is not None:
        try:
            s = s[(s.index(cut_before) + 1):]
        except ValueError:
            pass
    if cut_after is not None:
        try:
            s = s[: s.rindex(cut_after)]
        except ValueError:
            pass
    return s


def strip_(s: str, lstrip: Union[str, bool] = True, rstrip: Union[str, bool] = True) -> str:
    """
    This function `strip_` is a flexible string manipulation function that allows selective removal of leading (left) and/or
    trailing (right) characters from a given string `s`. The user can specify which sides to strip and which characters to remove.

    Args:
        s (str): The input string from which characters will be removed.
        lstrip (Union[str, bool]): A boolean flag indicating whether to remove leading characters from the left side of the string.
            If a string is provided, it removes characters specified in the string.
        rstrip (Union[str, bool]): A boolean flag indicating whether to remove trailing characters from the right side of the string.
            If a string is provided, it removes characters specified in the string.

    Returns:
        str: The modified string after removing the specified characters from the selected sides.

    Examples:
        >>> input_str = "   Hello, World!   "
        >>> strip_(input_str, lstrip=True, rstrip=True)
        'Hello, World!'

        >>> input_str = "   Hello, World!   "
        >>> strip_(input_str, lstrip=True, rstrip=False)
        'Hello, World!   '

        >>> input_str = "xxHello, World!xx"
        >>> strip_(input_str, lstrip='x', rstrip='x')
        'Hello, World!'
    """
    if lstrip is True:
        s = s.lstrip()
    elif isinstance(lstrip, str) and len(lstrip) > 0:
        s = s.lstrip(lstrip)

    if rstrip is True:
        s = s.rstrip()
    elif isinstance(rstrip, str) and len(rstrip) > 0:
        s = s.rstrip(rstrip)

    return s


class SearchFallbackOptions(str, Enum):
    EOS = 'eos',
    Empty = 'empty'
    RaiseError = 'error'


def index_(
        s: str,
        search: Union[str, Iterable[str]],
        start: int = 0,
        return_at_first_match: bool = False,
        return_end: bool = False,
        search_fallback_option: Union[str, SearchFallbackOptions] = SearchFallbackOptions.RaiseError
):
    """
    Finds the index of the first occurrence of a given substring or an ordered sequence of substrings in a string.
    The search can optionally return the end index of the found substring and handle cases where the substring is not found using different fallback options.

    Args:
        s: The string to search within.
        search: The substring or an iterable of substrings to find in `s`. If an iterable is provided,
                the function searches for each substring sequentially.
        start: The starting index in `s` from which to begin the search. Defaults to 0.
        return_end: If True, returns a tuple of the start index and the end index of the found substring.
                    If False, only the start index is returned. Defaults to False.
        search_fallback_option: Specifies the behavior when the search substring is not found. It can be:
                                - SearchFallbackOptions.EOS: Return the end of the string as the fallback index.
                                - SearchFallbackOptions.Empty: Return an empty string or (-1, -1) depending on `return_end`.
                                - SearchFallbackOptions.RaiseError: Raise a ValueError. This is the default behavior.

    Returns:
        The index of the first occurrence of `search` in `s` or a tuple (start index, end index) if `return_end` is True.
        The return value depends on the `search_fallback_option` if the substring is not found.

    Raises:
        ValueError: If `search` is not found in `s` and `search_fallback_option` is SearchFallbackOptions.RaiseError.

    Examples:
        Single Substring Search:
        >>> index_("hello world", "world")
        6

        Sequential Substrings Search:
        >>> index_("find the needle in the haystack", ["the", "needle"], return_end=True)
        (9, 15)

        Fallback Options:
        >>> index_("hello world", "bye", search_fallback_option='eos')
        11

        >>> index_("hello world", "bye", search_fallback_option=SearchFallbackOptions.Empty)
        -1

        Handling Iterables:
        >>> index_("looking for multiple words in a sentence", ["multiple", "words"], return_end=True)
        (21, 26)

        >>> index_("phrase with missing parts", ["missing", "parts"], search_fallback_option=SearchFallbackOptions.EOS, return_end=True)
        (20, 25)

        >>> index_("no such substrings", ["no", "such", "substrings"], search_fallback_option=SearchFallbackOptions.Empty)
        8
    """

    index_method = s.index if search_fallback_option == SearchFallbackOptions.RaiseError.value else s.find
    if isinstance(search, str):
        if return_end:
            start = index_method(search, start)
            if start == -1:
                if search_fallback_option == SearchFallbackOptions.EOS.value:
                    end = start = len(s)
                else:
                    end = start
            else:
                end = start + len(search)
            return start, end
        else:
            start = index_method(search, start)
            if search_fallback_option == SearchFallbackOptions.EOS.value:
                return len(s)
            else:
                return start
    else:
        end = None
        if return_at_first_match:
            _start = -1
            for substr in search:
                _start = s.find(substr, start)
                if _start != -1:
                    end = _start + len(substr)
                    break
            start = _start
        else:
            for substr in search:
                if end is not None:
                    start = end
                if isinstance(substr, str):
                    start = index_method(substr, start)
                    end = start + len(substr)
                else:
                    start, end = index_(s, substr, start, return_end=True, search_fallback_option=SearchFallbackOptions.Empty)

                if start == -1:
                    break

        if start == -1:
            if return_end:
                if search_fallback_option == SearchFallbackOptions.EOS.value:
                    return len(s), len(s)
                else:
                    return start, start
            else:
                if search_fallback_option == SearchFallbackOptions.EOS.value:
                    return len(s)
                else:
                    return start

        if return_end:
            return start, end
        else:
            return start


def index_pair(
        s: str,
        search1: str,
        search2: str,
        start: int = 0,
        search_fallback_option: Union[str, SearchFallbackOptions] = SearchFallbackOptions.RaiseError
) -> Tuple[int, int]:
    """
    Finds the indices in the string `s` marking the end of the first occurrence of `search1` and the start of the
    subsequent occurrence of `search2`. The function provides flexible handling for cases where `search2` is not found.

    Args:
        s: The string to search within.
        search1: The substring whose end marks the starting index of the result.
        search2: The substring whose start marks the ending index of the result.
        start: The index in `s` to start the search from. Defaults to 0.
        search_fallback_option: Determines the behavior when either `search1` or `search2` is not found:
            - SearchFallbackOptions.EOS: If `search1` or `search2` is not found, return the end of the string (`len(s)`)
              as the respective index.
            - SearchFallbackOptions.Empty: If `search1` or `search2` is not found, return the start index or the last valid
              search position as the respective index.
            - SearchFallbackOptions.RaiseError: Raise a ValueError if either `search1` or `search2` is not found. This is
              the default behavior.
    Returns:
        A tuple of two integers (start, end) representing the indices in `s`. The start index is at the end of the
        first occurrence of `search1`, and the end index is at the start of the subsequent occurrence of `search2`.

    Raises:
        ValueError: If either `search1` is not found in `s`, or `search2` is not found and `search_fallback_option`
                    is set to raise an error.

    Examples:
        >>> s = "This string is a sample string for testing."
        >>> index_pair(s, "is", "sample")
        (4, 17)

        >>> index_pair(s, "sample", "string", 10)
        (23, 24)

        >>> index_pair(s, "This", "not found", search_fallback_option=SearchFallbackOptions.EOS)
        (4, 43)

        >>> index_pair(s, "not there", "string", search_fallback_option=SearchFallbackOptions.Empty)
        (43, 43)

        >>> index_pair(s, "is", "not found", search_fallback_option=SearchFallbackOptions.RaiseError)
        Traceback (most recent call last):
            ...
        ValueError: substring not found
    """
    try:
        _, start = index_(s, search1, start, return_end=True)
    except ValueError as e:
        if search_fallback_option == SearchFallbackOptions.RaiseError.RaiseError.value:
            raise e
        else:
            start = len(s)
            return start, start

    try:
        end = index_(s, search2, start)
    except ValueError as e:
        if search_fallback_option == SearchFallbackOptions.RaiseError.EOS.value:
            end = len(s)
        elif search_fallback_option == SearchFallbackOptions.RaiseError.Empty.value:
            end = start
        else:
            raise e
    return start, end


def extract_between(
        s: str,
        search1,
        search2,
        start: int = 0,
        search_fallback_option: Union[str, SearchFallbackOptions] = SearchFallbackOptions.RaiseError
) -> str:
    """
    Extracts a substring from a given string, located between two specified substrings.

    This function identifies the segments of the string `s` that occur after `search1` and before `search2`,
    then returns the substring located between these segments. If `search2` is not found and
    `eos_fallback_for_search2` is true, it extracts until the end of the string `s`.

    Args:
        s: The string to extract from.
        search1: The substring after which extraction should start.
        search2: The substring before which extraction should end.
        start: The index to start the search from (defaults to 0).
        search_fallback_option: Determines the behavior when either `search1` or `search2` is not found:
            - SearchFallbackOptions.EOS: If `search1` or `search2` is not found, return the end of the string (`len(s)`)
              as the respective index.
            - SearchFallbackOptions.Empty: If `search1` or `search2` is not found, return the start index or the last valid
              search position as the respective index.
            - SearchFallbackOptions.RaiseError: Raise a ValueError if either `search1` or `search2` is not found. This is
              the default behavior.

    Returns:
        The extracted substring between `search1` and `search2`.

    Raises:
        ValueError: If either `search1` is not found in `s`, or `search2` is not found and `search_fallback_option`
                    is set to raise an error.


    Examples:
        >>> s = "This is a sample string for testing."
        >>> extract_between(s, "This is", "string")
        ' a sample '

        >>> extract_between(s, "sample", "testing", search_fallback_option=SearchFallbackOptions.EOS)
        ' string for '

        >>> extract_between(s, "not found", "string")
        Traceback (most recent call last):
            ...
        ValueError: substring not found

        >>> extract_between(s, "This is", "not found", search_fallback_option='empty')
        ''
    """
    start_index, end_index = index_pair(s, search1, search2, start, search_fallback_option)
    return s[start_index:end_index]
