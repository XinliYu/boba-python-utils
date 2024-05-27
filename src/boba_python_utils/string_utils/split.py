import re
from typing import Tuple, Optional, Union, Iterable, List

from boba_python_utils.common_utils.typing_helper import str2val_, make_list
from boba_python_utils.string_utils.common import cut, strip_, index_, index_pair, SearchFallbackOptions
from boba_python_utils.string_utils.regex import _get_whole_word_pattern


def bisplit(s: str, sep: str) -> Tuple[str, Optional[str]]:
    """
    Splits the input string `s` into two parts by the separator `sep`.
    Search of `sep` starts from the left side of `s`.
    If the separator does not exist in `s`, None will be returned as the second split.

    """
    splits = s.split(sep, maxsplit=1)
    return (splits[0], None) if len(splits) == 1 else splits


def birsplit(s: str, sep: str) -> Tuple[str, Optional[str]]:
    """
    Splits the input string `s` into two parts by the separator `sep`.
    Search of `sep` starts from the right side of `s`.
    If the separator does not exist in `s`, None will be returned as the second split.

    """
    splits = s.rsplit(sep, maxsplit=1)
    return (splits[0], None) if len(splits) == 1 else splits


def split_(
        s: str,
        sep: Union[str, Iterable[str]] = None,
        maxsplit: int = -1,
        n: int = None,
        minsplit=-1,
        pad=None,
        tile_padding: bool = True,
        remove_empty_split: bool = False,
        parse: bool = False,
        lstrip: bool = False,
        rstrip: bool = False,
        cut_before=None,
        cut_after=None,
        return_ori_num_splits=False,
        split_by_whole_words=False,
        use_space_as_word_boundary=True,
):
    """
    Advanced string split function with rich options.

    Args:
        s: The string to be split.
        sep: The delimiter(s) according to which to split the string.
        maxsplit: Maximum number of splits to do.
        n: The desired number of elements in the result.
        minsplit: Minimum number of splits required.
        pad: Element used to pad the result.
        tile_padding: Whether to tile the padding or not.
        remove_empty_split: Whether to remove empty strings from the result.
        parse: Whether to parse each split element.
        lstrip: Whether to strip leading whitespace from each split.
        rstrip: Whether to strip trailing whitespace from each split.
        cut_before: Substring before which everything is removed.
        cut_after: Substring after which everything is removed.
        return_ori_num_splits: Whether to return the original number of splits.
        split_by_whole_words: Whether to split by whole words.
        use_space_as_word_boundary: Use spaces as word boundaries when splitting.

    Returns:
        A list of split strings, and optionally the original number of splits.

    Examples:
        Enforce Result Size
        -------------------
        >>> split_('a, b, c', sep=',', n=2, lstrip=True, rstrip=True)
        ['a', 'b']

        Padding Missing Elements
        ------------------------
        >>> split_('a', sep=',', n=2, lstrip=True, rstrip=True)
        ['a', None]

        Removing Empty Splits
        ---------------------
        >>> split_('a, b, ,,, c, d, ,, e', sep=',', remove_empty_split=True, lstrip=True, rstrip=True)
        ['a', 'b', 'c', 'd', 'e']

        Parsing Values
        --------------
        >>> split_('1,2,3,4', sep=',', parse=True)
        [1, 2, 3, 4]

        Complex Parsing
        ---------------
        >>> split_('1\t\t2\t[3,4,5,6]\t{"a":7, "b":8}', sep='\t', parse=True, remove_empty_split=True)
        [1, 2, [3, 4, 5, 6], {'a': 7, 'b': 8}]
    """

    if n is not None and n < minsplit:
        raise ValueError("`n` must be greater or equal to `minsplit`")

    s = cut(s, cut_before, cut_after)

    if isinstance(sep, (list, tuple)) and len(sep) == 1:
        sep = sep[0]
    if sep is None or isinstance(sep, str):
        if split_by_whole_words:
            sep = _get_whole_word_pattern(sep, use_space_as_word_boundary)
            splits = re.split(sep, s) if maxsplit <= 0 else re.split(sep, s, maxsplit=maxsplit)
        else:
            splits = s.split(sep) if maxsplit <= 0 else s.split(sep, maxsplit=maxsplit)
    else:
        sep = '|'.join(f'(?:{re.escape(_sep)})' for _sep in sep)
        if split_by_whole_words:
            sep = _get_whole_word_pattern(sep, use_space_as_word_boundary)
        splits = re.split(sep, s) if maxsplit <= 0 else re.split(sep, s, maxsplit=maxsplit)

    splits = [x for x in splits if x is not None]
    num_splits = len(splits)
    if num_splits <= (minsplit + 1):
        raise ValueError(f"string '{s}' does not have enough splits by '{sep}'")

    def _process(x: str):
        x = strip_(x, lstrip=lstrip, rstrip=rstrip)
        if x:
            if parse:
                val, succ = str2val_(x, success_label=True)
                return val if succ else x
            else:
                return x
        else:
            return x

    if parse:
        splits = [_process(x) for x in splits]
    elif lstrip or rstrip:
        splits = [strip_(x, lstrip=lstrip, rstrip=rstrip) for x in splits]
    if remove_empty_split:
        splits = [x for x in splits if x]

    if n:
        if num_splits < n:
            if tile_padding:
                splits += [pad] * (n - len(splits))
            else:
                splits = splits + make_list(pad)
                if len(splits) != n:
                    raise ValueError(
                        f"number of splits is expected to be {n} after padding; got {len(splits)}"
                    )
        elif num_splits > n:
            splits = splits[:n]
    return (splits, num_splits) if return_ori_num_splits else splits


def csv_line_split(line: str, separator: str = ',') -> List[str]:
    """
    Split a single line of a CSV-like string into a list of values, handling quoted values.

    Args:
        line (str): The CSV-like string to split.
        separator (str, optional): The separator used to split the line. Defaults to ','.

    Returns:
        List[str]: A list of values representing the cells in the CSV line.

    Example:
        >>> csv_line = 'John|Doe|30|"New York, NY"'
        >>> csv_line_split(csv_line, separator='|')
        ['John', 'Doe', '30', 'New York, NY']
    """
    parts = []
    in_quote = False
    current_part = ''

    for char in line:
        if char == '"':
            in_quote = not in_quote
        elif char == separator and not in_quote:
            parts.append(current_part)
            current_part = ''
        else:
            current_part += char

    parts.append(current_part)

    return parts


def split_with_escape_and_quotes(
        s: str,
        delimiter: str,
        escape: Optional[str] = '\\',
        quotes: Optional[Tuple[str, str]] = ('[', ']'),
        keep_quotes: bool = True,
        keep_escape: bool = False,
        max_split: Optional[int] = None
):
    """
    Function to split a string by a delimiter, but ignoring delimiters that are escaped or inside quotes.

    Args:
        s: The string to split.
        delimiter: The delimiter character to split by.
        escape: The escape character. If None, no escape character is used. Defaults to '\\'.
        quotes: The opening and closing quote characters. If None, no quote characters are used. Defaults to ('[', ']').
        keep_quotes: Whether to keep the quote characters in the split parts. Defaults to True.
        keep_escape: Whether to keep the escape characters in the split parts. Defaults to False.
        max_split: Maximum number of splits to perform. If None, there is no limit. Defaults to None.

    Returns:
        The list of substrings after splitting the input string.

    Examples:
        >>> split_with_escape_and_quotes('abc.def.[ghi.jkl].mno', '.')
        ['abc', 'def', '[ghi.jkl]', 'mno']

        >>> split_with_escape_and_quotes('abc.def.[ghi.jkl].mno', '.', max_split=2)
        ['abc', 'def', '[ghi.jkl].mno']

        >>> split_with_escape_and_quotes('[abc.def].[ghi.jkl].mno', '.', max_split=1)
        ['[abc.def]', '[ghi.jkl].mno']

        >>> split_with_escape_and_quotes('abc.def.[ghi\\.jkl].mno', '.')
        ['abc', 'def', '[ghi.jkl]', 'mno']

        >>> split_with_escape_and_quotes('abc.def.\\[ghi.jkl].mno', '.')
        ['abc', 'def', '[ghi', 'jkl].mno']

        >>> split_with_escape_and_quotes('abc.def.[ghi\\.jkl].mno', '.', escape='')
        ['abc', 'def', '[ghi\\\\.jkl]', 'mno']

        >>> split_with_escape_and_quotes('abc.def\\.ghi\\.jkl.mno', '.', quotes=None)
        ['abc', 'def.ghi.jkl', 'mno']

        >>> split_with_escape_and_quotes('abc.def.ghi.jkl.mno', '.', escape=None, quotes=None)
        ['abc', 'def', 'ghi', 'jkl', 'mno']

        >>> split_with_escape_and_quotes('abc.def.[ghi.jkl].mno', '.', escape=None)
        ['abc', 'def', '[ghi.jkl]', 'mno']

        >>> split_with_escape_and_quotes('abc.def.ghi.jkl.mno', '.', quotes=None)
        ['abc', 'def', 'ghi', 'jkl', 'mno']

        >>> split_with_escape_and_quotes('abc;def;"ghi;jkl";mno', ';', quotes=('"', '"'))
        ['abc', 'def', '"ghi;jkl"', 'mno']

        >>> split_with_escape_and_quotes('abc;def;"ghi\\;jkl";mno', ';', quotes=('"', '"'))
        ['abc', 'def', '"ghi;jkl"', 'mno']

        >>> split_with_escape_and_quotes('abc;def;"ghi;jkl";mno', ';', escape=None, quotes=('"', '"'))
        ['abc', 'def', '"ghi;jkl"', 'mno']

        >>> split_with_escape_and_quotes('abc;def;"ghi;jkl;mno"', ';', quotes=('"', '"'))
        ['abc', 'def', '"ghi;jkl;mno"']
    """
    result = []
    current = []
    escaped = False
    quoted = False
    split_count = 0

    for char in s:
        if escape and char == escape and not escaped:  # Check if escape is not None and not empty
            escaped = True
            if keep_escape:
                current.append(char)
        elif quotes and char in quotes and not escaped:  # Check if quotes is not None and not empty
            quoted = not quoted
            if keep_quotes:
                current.append(char)
        elif char == delimiter and not quoted and not escaped:
            if max_split is not None and split_count >= max_split:
                current.append(char)
            else:
                result.append("".join(current))
                current = []
                split_count += 1
        else:
            current.append(char)
            escaped = False
    result.append("".join(current))
    return result


class SplitOptions:
    def __init__(
            self,
            separator,
            must_exist_or_default_value: Union[bool, str] = False,
            search_option: SearchFallbackOptions = SearchFallbackOptions.RaiseError
    ):
        self.separator = separator
        self.must_exist_or_default_value = must_exist_or_default_value
        self.search_option = search_option


def split_multiple(
        s: str,
        separators: List[
            Union[
                Union[str, Iterable[str]],
                Tuple[Union[str, Iterable[str]], Union[bool, str]]
            ]
        ],
        return_first: bool = True,
        return_last: bool = True,
        lstrip: bool = False,
        rstrip: bool = False,
        skip_empty_split: bool = False,
        start: int = 0,
        flexible_separators: bool = False
):
    """
    Splits a string based on multiple separators, yielding each split segment of the string.
    Optionally allows checking if separators must exist in the string and skipping their absence.

    Args:
        s: The string to be split.
        separators: A list of string separators or tuples used to split the input string.
             A tuple can be (separator, must_exist) where `must_exist` is True or False;
             and `must_exist` being True means a ValueError is raised if the `separator` is not found.
             A tuple can be (separator, default) where `default` is a string (e.g. an empty string);
             and `default` is yielded if the `separator` is not found.
        return_first: If True, includes the first segment before any separator. Defaults to True.
        return_last: If True, includes the last segment after the last separator. Defaults to True.
        lstrip: If True, strips leading whitespace from each split segment. Defaults to False.
        rstrip: If True, strips trailing whitespace from each split segment. Defaults to False.
        skip_empty_split: If True, skips empty strings in the split result. Defaults to False.
        start: The starting index from which the string will be split. Defaults to 0.
        flexible_separators: When True, allows for the separators to be processed flexibly. This means if multiple
                             separators are given, the function will not strictly follow the order in the separators list
                             but will process based on the occurrence of the separator in the string. If multiple separators
                             are found, the one occurring first in the string will be used. Defaults to False.
    Yields:
        str: The next split segment of the string.

    Examples:

        # >>> list(split_multiple("apple-banana-orange", ["-", "-"]))
        # ['apple', 'banana', 'orange']
        #
        # >>> list(split_multiple("one, two, three; four, five", [(", ", False), ("; ", True)], return_first=False))
        # ['two, three', 'four, five']
        #
        # >>> list(split_multiple("start-middle-end", [("-", True)], return_first=False, return_last=False))
        # []
        #
        # >>> list(split_multiple("no separators here", [("#", False), ("$", False)]))
        # ['no separators here']
        #
        # >>> list(split_multiple("  spaced  -words - separated ", ["-"], lstrip=True))
        # ['spaced  ', 'words - separated ']
        #
        # >>> list(split_multiple("empty;;;;splits;;here", [";;", ";;"], skip_empty_split=True))
        # ['empty', 'splits;;here']
        #
        # >>> list(split_multiple("multiple-separators  ;in;string", [";", "-"], rstrip=True, return_last=False))
        # ['multiple-separators']
        #
        # >>> list(split_multiple("string with missing separator", [("-", True)]))
        # Traceback (most recent call last):
        # ...
        # ValueError: separator '-' cannot be found in the input string "string with missing separator"

        # >>> list(split_multiple(
        # ...    'device endpointId is 49, name is "washer", appliance type is washer, use its instance Washer.WashCycle with friendly names ["cycle", "wash cycle"] to set one of {"WashCycle.Normal": "normal", "WashCycle.Delicates": "delicates"}, actions are SmartHome.adjustMode, SmartHome.getConnectivity, SmartHome.getMode, SmartHome.getPower, SmartHome.setMode, SmartHome.turnOff and SmartHome.turnOn',
        # ...    [
        # ...        SplitOptions(('device endpointId is', ', '), True),
        # ...        SplitOptions(('name is', ', '), ''),
        # ...        SplitOptions(('appliance type is', ', '), ''),
        # ...        SplitOptions(('with friendly names [', ']'), ''),
        # ...        SplitOptions(('actions are', ['and SmartHome', ', ']), search_option=SearchFallbackOptions.EOS)
        # ...    ],
        # ...    lstrip=True, rstrip=True, return_first=False, return_last=False)
        # ... )
        ['49', '"washer"', 'washer', '"cycle", "wash cycle"', 'SmartHome.adjustMode, SmartHome.getConnectivity, SmartHome.getMode, SmartHome.getPower, SmartHome.setMode, SmartHome.turnOff and SmartHome.turnOn']

        # >>> list(split_multiple(
        # ...    'device endpointId is 49, name is "washer", appliance type is washer, use its instance Washer.WashCycle with friendly names ["cycle", "wash cycle"] to set one of {"WashCycle.Normal": "normal", "WashCycle.Delicates": "delicates"}, actions are SmartHome.adjustMode, SmartHome.getConnectivity, SmartHome.getMode, SmartHome.getPower, SmartHome.setMode, SmartHome.turnOff and SmartHome.turnOn',
        # ...    [
        # ...        (('with friendly names [', ']'), ''),
        # ...        (('device endpointId is', ', '), True),
        # ...        (('name is', ', '), ''),
        # ...        (('appliance type is', ', '), ''),
        # ...        (('actions are', ['and SmartHome', ', ']), '$')
        # ...    ],
        # ...    lstrip=True, rstrip=True, return_first=False, return_last=False, flexible_separators=True)
        # ... )
        # ['49', '"washer"', 'washer', '"cycle", "wash cycle"', 'SmartHome.adjustMode, SmartHome.getConnectivity, SmartHome.getMode, SmartHome.getPower, SmartHome.setMode, SmartHome.turnOff and SmartHome.turnOn']

        >>> splits = list(split_multiple(
        ...    "Human: at sunrise wake me up and play my morning playlist\\nThought:\\nStep 1. One human turn in the conversation. Human starts by \\"at sunrise\\", which is a condition with a specific future time, hence one-time condition for the future, indicating automation. Then human asks to \\"wake me up\\" and \\"play my morning playlist\\"; the two asks have semantic connection as they are both what human needs in the morning, and based on natural understanding of the language they should be both associated with the condition.\\nStep 2. From Step 1 analysis, we recognize a single automation request with two action phrases \\"wake me up\\" and \\"play my morning playlist\\" with the condition phrase \\"at sunrise\\" together; the action phrases in this automation do not depend on each other and can execute independently.\\nStep 3. Decision: {\\"automation_requests\\": [{\\"action_phrases\\": [{\\"index\\": 0, \\"action_phrase\\": \\"wake me up\\", \\"dependency\\": []}, {\\"index\\": 1, \\"action_phrase\\": \\"play my morning playlist\\", \\"dependency\\": []}]",
        ...    [
        ...        ['Human:'],
        ...        ['Thought:'],
        ...        ['Step 1.', 'Step 1:'],
        ...        ['Step 2.', 'Step 2:'],
        ...        ['Step 3.', 'Step 3:'],
        ...    ],
        ...    lstrip=True, rstrip=True, return_first=False, skip_empty_split=False, return_last=True)
        ... )
        >>> len(splits)
        5
        >>> print(splits)
        ['at sunrise wake me up and play my morning playlist', '', 'One human turn in the conversation. Human starts by "at sunrise", which is a condition with a specific future time, hence one-time condition for the future, indicating automation. Then human asks to "wake me up" and "play my morning playlist"; the two asks have semantic connection as they are both what human needs in the morning, and based on natural understanding of the language they should be both associated with the condition.', 'From Step 1 analysis, we recognize a single automation request with two action phrases "wake me up" and "play my morning playlist" with the condition phrase "at sunrise" together; the action phrases in this automation do not depend on each other and can execute independently.', 'Decision: {"automation_requests": [{"action_phrases": [{"index": 0, "action_phrase": "wake me up", "dependency": []}, {"index": 1, "action_phrase": "play my morning playlist", "dependency": []}]']

        >>> splits = list(split_multiple(
        ...    "Human: i want to order a birthday cake from the local bakery for this saturday can you take care of it\\nThought:\\nStep 1: The human has one request in this turn. The human wants to \\"order a birthday cake from the local bakery for this Saturday\\". The phrase \\"can you take care of it\\" is a request for the Assistant to complete the action. There are no conditions or co-references that need resolution.\\nStep 2: Based on Step 1, we recognize one unit regular request, which is \\"order a birthday cake from the local bakery for this Saturday\\". This request does not have any dependencies.\\nStep 3: Decision: {\\"regular_requests\\": [{\\"index\\": 0, \\"action_phrase\\": \\"order a birthday cake from the local bakery for this Saturday\\", \\"dependency\\": []}]}",
        ...    [
        ...        ['Human:'],
        ...        ['Thought:'],
        ...        ['Step 1.', 'Step 1:'],
        ...        ['Step 2.', 'Step 2:'],
        ...        ['Step 3.', 'Step 3:'],
        ...    ],
        ...    lstrip=True, rstrip=True, return_first=False, skip_empty_split=False, return_last=True)
        ... )
        >>> len(splits)
        5
        >>> print(splits)
        ['i want to order a birthday cake from the local bakery for this saturday can you take care of it', '', 'The human has one request in this turn. The human wants to "order a birthday cake from the local bakery for this Saturday". The phrase "can you take care of it" is a request for the Assistant to complete the action. There are no conditions or co-references that need resolution.', 'Based on Step 1, we recognize one unit regular request, which is "order a birthday cake from the local bakery for this Saturday". This request does not have any dependencies.', 'Decision: {"regular_requests": [{"index": 0, "action_phrase": "order a birthday cake from the local bakery for this Saturday", "dependency": []}]}']
    """

    def strip_(segment, lstrip, rstrip):
        """Helper function to conditionally strip a string segment."""
        if lstrip:
            segment = segment.lstrip()
        if rstrip:
            segment = segment.rstrip()
        return segment

    is_first = True
    for i in range(len(separators)):
        must_exist_or_default_value = False
        search_option = SearchFallbackOptions.RaiseError

        if flexible_separators:
            separator = None
            separator0_idx = len(s)
            j = -1
            _any_sep_must_exist_flag = False
            _must_exist_seperator = None
            for _j, _separator in enumerate(separators[i:]):
                _sep_must_exist_flag_or_default_yield = False
                if isinstance(_separator, tuple) and len(_separator) == 2:
                    _separator, _sep_must_exist_flag_or_default_yield = _separator
                    if _sep_must_exist_flag_or_default_yield is True and _any_sep_must_exist_flag is False:
                        _any_sep_must_exist_flag = True
                        _must_exist_seperator = _separator
                is_pair_based_split = isinstance(_separator, tuple)
                if is_pair_based_split:
                    separator0 = _separator[0]
                else:
                    separator0 = _separator
                try:
                    _separator0_idx = index_(s, separator0, start)
                except ValueError:
                    continue
                if _separator0_idx < separator0_idx:
                    separator0_idx = _separator0_idx
                    separator = _separator
                    must_exist_or_default_value = _sep_must_exist_flag_or_default_yield
                    j = (i + _j)
            if j == -1:
                if _any_sep_must_exist_flag:
                    raise ValueError(f"separator '{separator}' cannot be found in the input string \"{s}\"")
                else:
                    return
            if i != j:
                separators[i], separators[j] = separators[j], separators[i]
        else:
            separator = separators[i]
            if isinstance(separator, SplitOptions):
                must_exist_or_default_value = separator.must_exist_or_default_value
                search_option = separator.search_option
                separator = separator.separator
        try:
            return_at_first_match = not isinstance(separator, tuple)
            _start, end = index_(
                s,
                separator,
                start,
                return_end=True,
                return_at_first_match=return_at_first_match,
                search_fallback_option=search_option
            )
            if return_first or (not is_first):
                split = strip_(s[start:_start], lstrip, rstrip)
                if (not skip_empty_split) or split:
                    yield split
            is_first = False
            start = end
        except ValueError:
            if must_exist_or_default_value is True:
                raise ValueError(f"separator '{separator}' cannot be found in the input string \"{s}\"")
            elif must_exist_or_default_value is not False:
                is_first = False
                yield must_exist_or_default_value
                continue
            elif search_option == SearchFallbackOptions.EOS:
                start = len(s)
                continue
            else:
                continue

    if return_last:
        split = strip_(s[start:], lstrip, rstrip)
        if (not skip_empty_split) or split:
            yield split
