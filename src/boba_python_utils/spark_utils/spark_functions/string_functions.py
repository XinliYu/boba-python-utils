from functools import partial
from typing import Union, Iterable, Tuple, Mapping, Optional, Callable
from pyspark.sql.column import Column
from pyspark.sql.functions import regexp_replace, lower, lit, length, when, levenshtein, greatest, udf, expr
from pyspark.sql.types import IntegerType, ArrayType, StringType

from boba_python_utils.common_utils.iter_helper import iter__
from boba_python_utils.common_utils.typing_helper import solve_key_value_pairs
from boba_python_utils.general_utils.nlp_utility.punctuations import remove_acronym_periods_and_spaces_udf
from boba_python_utils.general_utils.nlp_utility.string_sanitization import (
    string_sanitize,
    StringSanitizationOptions,
    StringSanitizationConfig,
    _lower,
    _sort_tokens,
    _replace_method,
    _remove_spaces,
    fuzz,
    remove_common_tokens
)
from boba_python_utils.spark_utils.spark_functions.nlp_functions.punctuations import remove_punctuation_except_for_hyphen_udf
from boba_python_utils.spark_utils.spark_functions.common import col_, or_, and_
from boba_python_utils.spark_utils.typing import NameOrColumn
from boba_python_utils.string_utils import remove_common_prefix_suffix
from boba_python_utils.string_utils.common import count_uppercase as _count_uppercase, count_lowercase as _count_lowercase
from boba_python_utils.string_utils.comparison import CompareOption, CompareMethod, solve_compare_option
from boba_python_utils.string_utils.string_sanitization import remove_accents as _remove_accents

# region string normalization

remove_accents = udf(_remove_accents, StringType())

# endregion


def regexp_remove(col: NameOrColumn, removal) -> Column:
    return regexp_replace(col_(col), removal, '')


def regexp_remove_many(col: NameOrColumn, *removals) -> Column:
    col = col_(col)
    for pattern in removals:
        col = regexp_replace(col, pattern, '')
    return col


def regexp_replace_many(col: NameOrColumn, replaces) -> Column:
    """
    Performs one or more `regexp_replace` operations on the input column.
    The `replaces` is a sequence of pairs of pattern and replacement.

    """
    col = col_(col)
    for pattern, replacement in solve_key_value_pairs(replaces):
        col = regexp_replace(col, pattern, replacement)
    return col


def contains_any(col: NameOrColumn, targets) -> Column:
    """
    Checks if a string column contains any of the target,
    where each target can be another string column, or a string literal.
    """
    col = col_(col)
    return or_(
        col.contains(target) for target in iter__(targets)
    )


def contains_all(col: NameOrColumn, targets) -> Column:
    """
    Checks if a string column contains all the targets,
    where each target can be another string column, or a string literal.
    """
    col = col_(col)
    return and_(
        col.contains(target) for target in iter__(targets)
    )


def contains_any_all(col: NameOrColumn, targets) -> Column:
    """
    Checks if a string column contains any of the target,
    where each target can be one or more string columns or literals,
    and `col` is considered containing one target
    if `col` contains all string columns or literals in the target.


    """
    col = col_(col)
    return or_(
        contains_all(col, target) for target in iter__(targets)
    )


def replace_all(
        s: str,
        replacements: Union[Iterable[Tuple[str, str]], Mapping[str, str]],
        replace_when_target_not_in_string: bool = False
):
    """
    Replaces all occurrences of the first element in each tuple with the second element in a string.

    Args:
        s: The string in which to perform the replacements.
        replacements: An iterable of 2-tuples, where the first element is the string to be
            replaced, and the second element is the replacement string; or a mapping,
            where the key is the string to be replaced, and the value is the replacement string.
        replace_when_target_not_in_string: If True, only performs the replacement if the replacement
            string is not in `s`, or if it is in the original string but also
            appears in the string to be replaced.

    Returns:
        str: The string with all replacements made.

    Examples:
        >>> replace_all("This is a test.", [("is", "was"), ("test", "trial")])
        'Thwas was a trial.'

        >>> replace_all("This was a test.", [("is", "was"), ("test", "trial")], True)
        'This was a trial.'

        >>> replace_all("This is a test.", [("is", "was"), ("was", "is")])
        'This is a test.'

        >>> replace_all("This is a test.", [])
        'This is a test.'
    """
    for x, y in solve_key_value_pairs(replacements):
        if (not replace_when_target_not_in_string) or (y not in s) or (y in x):
            s = s.replace(x, y)
    return s


count_uppercase = udf(_count_uppercase, IntegerType())
count_lowercase = udf(_count_lowercase, IntegerType())


def string_compare(
        col: NameOrColumn,
        pattern: str,
        option: CompareOption
) -> Column:
    col = col_(col)
    if option.is_regular_expression:
        if option.compare_method == CompareMethod.ExactMatch:
            pattern = f'^{pattern}$'
        elif option.compare_method == CompareMethod.StartsWith:
            pattern = f'^{pattern}'
        elif option.compare_method == CompareMethod.EndsWith:
            pattern = f'{pattern}$'
        elif option.compare_method == CompareMethod.Contains:
            pattern = f'.*(?:{pattern}).*'
        if not option.case_sensitive:
            pattern = f'(?i){pattern}'
        result = col.rlike(pattern)
    else:
        if not option.case_sensitive:
            col = lower(col)
            pattern = lower(pattern)
        if option.compare_method == CompareMethod.ExactMatch:
            result = (col == pattern)
        elif option.compare_method == CompareMethod.Contains:
            result = (col.contains(pattern))
        elif option.compare_method == CompareMethod.StartsWith:
            result = (col.startswith(pattern))
        elif option.compare_method == CompareMethod.EndsWith:
            result = (col.endswith(pattern))
        elif option.compare_method == CompareMethod.LowerLexicalOrder:
            result = (col < pattern)
        elif option.compare_method == CompareMethod.HigherLexicalOrder:
            result = (col > pattern)
        else:
            result = (col == pattern)

    if option.negation:
        result = (~result)
    if option.ignore_null:
        result = (result.isNull() | result)
    return result


def string_check(col: Union[str, Column], pattern: str, **kwargs) -> Column:
    option, pattern = solve_compare_option(pattern, **kwargs)
    return string_compare(col=col, pattern=pattern, option=option)


def levenshtein_normalized(left, right):
    left = col_(left)
    right = col_(right)
    return levenshtein(left, right) / (greatest(length(left), length(right)))


def levenshtein_similarity(left, right):
    return 1 - levenshtein_normalized(left, right)


def multi_col_udf(
        *cols,
        func,
        return_type=None,
        results_as_separated_cols=True,
        **kwargs
):
    result_col = udf(
        partial(
            func,
            **kwargs
        ),
        returnType=ArrayType(return_type or StringType())
    )(*cols)

    if results_as_separated_cols:
        return [
            result_col.getItem(i)
            for i in range(
                results_as_separated_cols
                if isinstance(results_as_separated_cols, int) else len(cols)
            )
        ]
    else:
        return result_col


def remove_common_prefix_suffix_udf(
        *strings,
        tokenizer: Optional[Union[Callable, str]] = None,
        prefixes: Iterable[str] = None,
        suffixes: Iterable[str] = None,
        remove_prefix: bool = True,
        remove_suffix: bool = False,
        results_as_separated_cols=True
):
    return multi_col_udf(
        *strings,
        func=remove_common_prefix_suffix,
        results_as_separated_cols=results_as_separated_cols,
        tokenizer=tokenizer,
        prefixes=prefixes,
        suffixes=suffixes,
        remove_prefix=remove_prefix,
        remove_suffix=remove_suffix
    )


def remove_common_tokens_udf(
        *strings,
        tokenizer: Optional[Union[Callable, str]] = None,
        results_as_separated_cols=True
):
    return multi_col_udf(
        *strings,
        func=remove_common_tokens,
        results_as_separated_cols=results_as_separated_cols,
        tokenizer=tokenizer
    )


def string_sanitize_udf(
        *strings,
        config: Union[Iterable[StringSanitizationOptions], StringSanitizationConfig],
        tokenizer=None,
        **kwargs
) -> Union[Column, Iterable[Column]]:
    return string_sanitize(
        *strings,
        config=config,
        tokenizer=tokenizer,
        remove_acronym_periods_and_spaces_method=remove_acronym_periods_and_spaces_udf,
        remove_case_method=_lower,
        remove_common_prefix_suffix_method=remove_common_prefix_suffix_udf,
        remove_punctuation_except_for_hyphen_method=remove_punctuation_except_for_hyphen_udf,
        remove_spaces_method=_remove_spaces,
        replace_method=_replace_method,
        sort_tokens_method=_sort_tokens,
        make_fuzzy_method=udf(fuzz),
        remove_common_tokens_method=remove_common_tokens_udf,
        return_intermediate_results_before_actions=None,
        **kwargs
    )


# region prefix/suffix

def remove_suffix(col, suffix: Union[str, Column]):
    col = col_(col)
    substr_len = (lit(len(suffix)) if isinstance(suffix, str) else length(suffix))
    return when(
        col.endswith(suffix),
        col.substr(
            lit(0),
            (length(col) - substr_len)
        ),
    ).otherwise(col)


# endregion


@udf
def substring_after(s, substr):
    idx = s.find(substr)
    if idx != -1:
        return s[(idx + len(substr)):]


@udf
def substring_after_caseless(s, substr):
    idx = s.lower().find(substr.lower())
    if idx != -1:
        return s[(idx + len(substr)):]


def locate_(substr_colname: str, str_colname: str, pos: int = 0):
    return expr(f'locate({substr_colname}, {str_colname}, {pos}) - 1')
