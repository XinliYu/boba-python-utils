from typing import Optional

from boba_python_utils.string_utils.common import OccurrenceOptions
from boba_python_utils.string_utils.regex import sub_last, sub_first, sub


def get_human_int_str(
        num: int,
        num_digits: int = 3,
        magnitude_letters=('K', 'M', 'B', 'T')
) -> str:
    """
    Gets a concise human-readable string to represent an integer, usually large integer.
    The representation has a float number part with at most `num_digits` digits,
        followed by a magnitude letttr like K, M, B, T;
        for example, '1.82M'.

    Examples:
        >>> get_human_int_str(1000000)
        '1M'
        >>> get_human_int_str(1621)
        '1.62K'
        >>> get_human_int_str(28234231, num_digits=4, magnitude_letters=('k', 'm', 'b', 't'))
        '28.23m'
    """
    num = int(num)
    num = float(f'{{:.{num_digits}g}}'.format(num))
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return '{}{}'.format(
        '{:f}'.format(num).rstrip('0').rstrip('.'), ['', *magnitude_letters][magnitude]
    )


def adjust_num_in_str(
        string: str,
        adjust: int = 1,
        occurrence: OccurrenceOptions = OccurrenceOptions.Last
):
    """
    Adjusts numbers in the string by the quantity specified in argument `adjust`.

    Args:
        string: the string.
        adjust: the adjustment to the number(s).
        occurrence: indicates whether we adjust
            the first number, last number or all numbers in the string.

    Returns: the string with numbers in it adjusted.

    Examples:
        >>> adjust_num_in_str('v1.1')
        'v1.2'
        >>> adjust_num_in_str('v1.1', occurrence=OccurrenceOptions.First)
        'v2.1'
        >>> adjust_num_in_str('v1.1', occurrence=OccurrenceOptions.All)
        'v2.2'
        >>> adjust_num_in_str('v1.1', adjust=-1)
        'v1.0'
        >>> adjust_num_in_str('v001')
        'v002'
        >>> adjust_num_in_str('v999')
        'v1000'

    """
    if occurrence == occurrence.Last:
        return sub_last(
            pattern='[0-9]+',
            repl=lambda x: '{{:0{}d}}'.format(len(x.group(0))).format(int(x.group(0)) + adjust),
            string=string
        )
    elif occurrence == occurrence.First:
        return sub_first(
            pattern='[0-9]+',
            repl=lambda x: '{{:0{}d}}'.format(len(x.group(0))).format(int(x.group(0)) + adjust),
            string=string
        )
    else:
        return sub(
            pattern='[0-9]+',
            repl=lambda x: '{{:0{}d}}'.format(len(x.group(0))).format(int(x.group(0)) + adjust),
            string=string
        )


def increment_num_in_str(s: str, occurrence: OccurrenceOptions = OccurrenceOptions.Last):
    """
    See `adjust_num_in_str`.
    """
    return adjust_num_in_str(s, adjust=1, occurrence=occurrence)


def get_domain_from_name(name: str, domain_separator: str = '.') -> Optional[str]:
    """
    Extracts the domain (the substring before the first occurrence of the domain separator) from a given name.

    If the name contains the domain separator, this function returns the part of the name before the separator.
    If the separator is not found, the function returns None. This function is particularly useful for parsing
    names that follow a domain-based notation (e.g., domain-specific language terms, hierarchical identifiers).

    Args:
        name: The string from which to extract the domain.
        domain_separator: The character used to separate the domain from the rest of the string. Defaults to '.'.

    Returns:
        The domain extracted from the name if the separator is present; otherwise, None.

    Examples:
        >>> get_domain_from_name('subdomain.domain.com')
        'subdomain'

        >>> get_domain_from_name('domain.com', domain_separator='.')
        'domain'

        >>> get_domain_from_name('singleword')

        >>> get_domain_from_name('namespace::classname', domain_separator='::')
        'namespace'
    """
    if name:
        function_name_splits = name.split(domain_separator, maxsplit=1)
        if len(function_name_splits) == 2:
            return function_name_splits[0]
