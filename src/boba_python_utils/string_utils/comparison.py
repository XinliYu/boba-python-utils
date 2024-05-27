from enum import Enum
from typing import Tuple, Optional

from attr import attrs, attrib
import re


# def string_check(s: str, pattern: str):
#     if pattern[0] == '!':
#         negative = True
#         pattern = pattern[1:]
#     else:
#         negative = False
#
#     if pattern[0] in '^$*@':
#         indicator = pattern[0]
#         pattern = pattern[1:]
#         if indicator == '^':
#             result = (s.startswith(pattern))
#         elif indicator == '$':
#             result = (s.endswith(pattern))
#         elif indicator == '*':
#             result = (pattern in s)
#         else:
#             result = (re.fullmatch(pattern, s) is not None)
#     else:
#         result = (s == pattern)
#     return result != negative


class CompareMethod(str, Enum):
    ExactMatch = 'exact_match'
    Contains = 'contains'
    StartsWith = 'starts_with'
    EndsWith = 'ends_with'
    LowerLexicalOrder = '<'
    HigherLexicalOrder = '>'


@attrs(slots=True)
class CompareOption:
    compare_method = attrib(type=CompareMethod, default=CompareMethod.ExactMatch)
    is_regular_expression = attrib(type=bool, default=False)
    case_sensitive = attrib(type=bool, default=True)
    ignore_null = attrib(type=bool, default=False)
    negation = attrib(type=bool, default=False)
    other_options = attrib(type=str, default=None)


def solve_compare_option(
        s: str,
        contains_indicator='*',
        starts_with_indicator='^',
        ends_with_indicator='$',
        lower_lexical_order_indicator='<',
        higher_lexical_order_indicator='>',
        regular_expression_indicator='@',
        negation_indicators='!~',
        case_insensitive_indicator='/',
        ignore_null_indicator='?',
        space_as_option_break=True,
        option_at_start=True,
        option_at_end=False,
        other_option_indicators=None,
        return_none_if_no_option_available: bool = False
) -> Tuple[Optional[CompareOption], str]:
    """
    Solves a string comparison directive. See :func:`string_check`.
    """
    negation = ignore_null = is_regular_expression = False
    case_sensitive = True
    compare_method = CompareMethod.ExactMatch
    other_options = []

    def _get_options():
        nonlocal i, compare_method, is_regular_expression, case_sensitive, ignore_null, negation
        for i in idxes:
            c = s[i]
            if negation_indicators and c in negation_indicators:
                negation = (not negation)
            elif contains_indicator and c == contains_indicator:
                compare_method = CompareMethod.Contains
            elif starts_with_indicator and c == starts_with_indicator:
                compare_method = CompareMethod.StartsWith
            elif ends_with_indicator and c == ends_with_indicator:
                compare_method = CompareMethod.EndsWith
            elif lower_lexical_order_indicator and c == lower_lexical_order_indicator:
                compare_method = CompareMethod.LowerLexicalOrder
            elif higher_lexical_order_indicator and c == higher_lexical_order_indicator:
                compare_method = CompareMethod.HigherLexicalOrder
            elif regular_expression_indicator and c == regular_expression_indicator:
                is_regular_expression = True
            elif case_insensitive_indicator and c == case_insensitive_indicator:
                case_sensitive = False
            elif ignore_null_indicator and c == ignore_null_indicator:
                ignore_null = True
            elif other_option_indicators and c in other_option_indicators:
                other_options.append(c)
            else:
                break

    has_option_at_end = has_option_at_start = False
    if option_at_end:
        i = len(s) - 1
        idxes = reversed(range(len(s)))
        _get_options()
        s = s[:(i + 1)]
        has_option_at_end = (i != len(s) - 1)
        if space_as_option_break:
            s = s.rstrip()
    if option_at_start:
        i = 0
        idxes = range(len(s))
        _get_options()
        s = s[i:]
        has_option_at_start = (i != 0)
        if space_as_option_break:
            s = s.lstrip()

    if not (has_option_at_start or has_option_at_end) and return_none_if_no_option_available:
        return None, s
    else:
        return CompareOption(
            compare_method=compare_method,
            is_regular_expression=is_regular_expression,
            case_sensitive=case_sensitive,
            ignore_null=ignore_null,
            negation=negation,
            other_options=(''.join(other_options) if other_options else None)
        ), s


def string_compare(src: str, trg: str, option: CompareOption) -> bool:
    if src is None:
        return option.ignore_null

    if option.is_regular_expression:
        if option.compare_method == CompareMethod.ExactMatch:
            result = (
                    re.fullmatch(trg, src, (0 if option.case_sensitive else re.IGNORECASE))
                    is not None
            )
        else:
            if option.compare_method == CompareMethod.StartsWith:
                trg = f'^{trg}'
            elif option.compare_method == CompareMethod.EndsWith:
                trg = f'{trg}$'
            else:
                raise ValueError(
                    f"regular expression does not support method {option.compare_method}"
                )
            result = (
                # ! must use `search` rather than `match`,
                # because `re.match` only matches from the beginning of the string
                    re.search(trg, src, (0 if option.case_sensitive else re.IGNORECASE))
                    is not None
            )
    else:
        if not option.case_sensitive:
            src = src.lower()
            trg = trg.lower()
        if option.compare_method == CompareMethod.ExactMatch:
            result = (src == trg)
        elif option.compare_method == CompareMethod.Contains:
            result = (trg in src)
        elif option.compare_method == CompareMethod.StartsWith:
            result = (src.startswith(trg))
        elif option.compare_method == CompareMethod.EndsWith:
            result = (src.endswith(trg))
        elif option.compare_method == CompareMethod.LowerLexicalOrder:
            result = (src < trg)
        elif option.compare_method == CompareMethod.HigherLexicalOrder:
            result = (src > trg)
        else:
            result = (src == trg)

    return result != option.negation


def string_check(s: str, pattern: str, **kwargs) -> bool:
    """
    Quickly check if the string `s` matches the given `pattern`.

    The `pattern` uses a string directive.
    A string comparison directive is a fast way to specify a condition a string must satisfy.
    The currently supported directives include
    1) '* substr', the string must contain the specified substring;
    2) '^ substr', the string must start with the specified substring;
    3) '$ substr', the string must end with the specified substring;
    4) '@ regex', the string must match the specified regular expression; can combine with '^', '$',
        for example, '@^' means the start of the string must match the regular expression;
    5) '! directive', negation of another directive, e.g. '!*substr' means the string must not
        contain the specified substring, or '!@$ regex' mens the end of the string must not match
        the specified pattern.

    A space between the directive characters and the string is recommended,
    but optional in most cases. For example, we can specify '* substr' or '*substr'.

    See Also :func:`solve_compare_option` and :func:`string_compare`.

    Examples:
        >>> assert string_check('1456', '1456')  # exact match
        >>> assert string_check('123456', '*12')  # contains substring '12'
        >>> assert string_check('123456', '* 12')  # contains substring '12', can add a space
        >>> assert string_check('123456', '^ 12')  # starts with substring '12'
        >>> assert string_check('123456', '$ 56')  # ends with substring '56'
        >>> assert string_check('123456', '!* ab')  # not contains with substring 'ab'
        >>> assert string_check('123456', '@ [0-9]+')  # matches regular expression '[0-9]+'
        >>> assert string_check('ab123456', '@^ [a-z]+')  # start of string matches regular expression '[a-z]+'
        >>> assert string_check('123456ab', '@$ [a-z]+')  # end of string matches regular expression '[a-z]+'
        >>> assert string_check('123456', '!@$ [a-z]+')  # end of string does not match regular expression '[a-z]+'
    
    Args:
        s: the string to check.
        pattern: a pattern in the format of a string directive.
        kwargs: the options for the string directive, passed to :func:`solve_compare_option`.

    Returns: a Boolean value indicating if the string `s` satisfies the specified `pattern`.

    """
    option, pattern = solve_compare_option(pattern, **kwargs)
    return string_compare(src=s, trg=pattern, option=option)
