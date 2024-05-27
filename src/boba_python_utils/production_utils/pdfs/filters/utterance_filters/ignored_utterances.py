from boba_python_utils.spark_utils.spark_functions.common import or_, col_
from boba_python_utils.spark_utils.spark_functions.string_functions import string_check
from boba_python_utils.spark_utils.typing import NameOrColumn
from boba_python_utils.string_utils.comparison import string_check as string_check_str
from boba_python_utils.production_utils.common._constants.path import SupportedLocales
from boba_python_utils.production_utils.pdfs.filters._configs.utterance_filter import (
    IGNORED_UTTERANCES_FOR_QUERIES,
    IGNORED_UTTERANCES_FOR_QUERIES_AND_REWRITES,
    IGNORED_UTTERANCE_PATTERNS_FOR_QUERIES,
    IGNORED_UTTERANCE_PATTERNS_FOR_QUERIES_AND_REWRITES,
    IGNORED_UTTERANCES_FOR_REWRITES,
    IGNORED_UTTERANCE_PATTERNS_FOR_REWRITES,
    IGNORED_UTTERANCES_FOR_QUERIES_BY_LOCALE,
    IGNORED_UTTERANCES_FOR_REWRITES_BY_LOCALE, HIGH_RISK_UTTERANCE_PREFIX_INCOMPLETE, MORE_UTTERANCE_PREFIX_INCOMPLETE
)


def is_incomplete_utterance(utterance):
    return utterance in HIGH_RISK_UTTERANCE_PREFIX_INCOMPLETE or \
           utterance in MORE_UTTERANCE_PREFIX_INCOMPLETE


def is_incomplete_utterance_udf(utterance_col):
    return (
            col_(utterance_col).isin(set(HIGH_RISK_UTTERANCE_PREFIX_INCOMPLETE)) |
            col_(utterance_col).isin(set(MORE_UTTERANCE_PREFIX_INCOMPLETE))
    )


def is_ignored_utterance_for_query(utterance):
    """
    Examples:
         >>> assert not is_ignored_utterance_for_query('asshole i have feedback')

    """
    return (
            utterance in IGNORED_UTTERANCES_FOR_QUERIES or
            utterance in IGNORED_UTTERANCES_FOR_QUERIES_AND_REWRITES or
            any(
                string_check_str(utterance, pattern)
                for pattern in IGNORED_UTTERANCE_PATTERNS_FOR_QUERIES
            ) or
            any(
                string_check_str(utterance, pattern)
                for pattern in IGNORED_UTTERANCE_PATTERNS_FOR_QUERIES_AND_REWRITES
            )
    )


def is_ignored_utterance_for_rewrite(utterance):
    """
    Examples:
        >>> assert is_ignored_utterance_for_rewrite('you are an asshole')
        >>> assert is_ignored_utterance_for_rewrite('fucking bitch')

    """
    return (
            utterance in IGNORED_UTTERANCES_FOR_REWRITES or
            utterance in IGNORED_UTTERANCES_FOR_QUERIES_AND_REWRITES or
            any(
                string_check_str(utterance, pattern)
                for pattern in IGNORED_UTTERANCE_PATTERNS_FOR_REWRITES
            ) or
            any(
                string_check_str(utterance, pattern)
                for pattern in IGNORED_UTTERANCE_PATTERNS_FOR_QUERIES_AND_REWRITES
            )
    )


def is_ignored_utterance_pair(query_utterance, rewrite_utterance):
    return (
            is_ignored_utterance_for_query(query_utterance) or
            is_ignored_utterance_for_rewrite(rewrite_utterance)
    )


def is_ignored_utterance_for_query_udf(
        query_utterance_col: NameOrColumn,
        locale: SupportedLocales = None
):
    return col_(query_utterance_col).isNotNull() & or_(
        col_(query_utterance_col).isin(set(IGNORED_UTTERANCES_FOR_QUERIES)),
        col_(query_utterance_col).isin(set(IGNORED_UTTERANCES_FOR_QUERIES_AND_REWRITES)),
        or_(
            string_check(query_utterance_col, pattern)
            for pattern in IGNORED_UTTERANCE_PATTERNS_FOR_QUERIES
        ),
        or_(
            string_check(query_utterance_col, pattern)
            for pattern in IGNORED_UTTERANCE_PATTERNS_FOR_QUERIES_AND_REWRITES
        ),
        (
            col_(query_utterance_col).isin(IGNORED_UTTERANCES_FOR_QUERIES_BY_LOCALE[locale])
            if locale and locale in IGNORED_UTTERANCES_FOR_QUERIES_BY_LOCALE
            else None
        )
    )


def is_ignored_utterance_for_rewrite_udf(
        rewrite_utterance_col: NameOrColumn,
        locale: SupportedLocales = None
):
    return col_(rewrite_utterance_col).isNotNull() & or_(
        col_(rewrite_utterance_col).isin(set(IGNORED_UTTERANCES_FOR_REWRITES)),
        col_(rewrite_utterance_col).isin(set(IGNORED_UTTERANCES_FOR_QUERIES_AND_REWRITES)),
        or_(
            string_check(rewrite_utterance_col, pattern)
            for pattern in IGNORED_UTTERANCE_PATTERNS_FOR_REWRITES
        ),
        or_(
            string_check(rewrite_utterance_col, pattern)
            for pattern in IGNORED_UTTERANCE_PATTERNS_FOR_QUERIES_AND_REWRITES
        ),
        (
            col_(rewrite_utterance_col).isin(IGNORED_UTTERANCES_FOR_REWRITES_BY_LOCALE[locale])
            if locale and locale in IGNORED_UTTERANCES_FOR_REWRITES_BY_LOCALE
            else None
        )
    )


def is_ignored_utterance_pair_udf(
        query_utterance_col,
        rewrite_utterance_col,
        locale: SupportedLocales = None
):
    return (
            is_ignored_utterance_for_query_udf(query_utterance_col, locale=locale) |
            is_ignored_utterance_for_rewrite_udf(rewrite_utterance_col, locale=locale)
    )
