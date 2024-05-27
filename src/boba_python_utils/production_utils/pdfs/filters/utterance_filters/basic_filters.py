from enum import Enum

import boba_python_utils.spark_utils.spark_functions as F
from boba_python_utils.production_utils.pdfs import constants as c

"""
Define basic utterance filters. These filters are only dependent on the utterance texts,
without references to filter lists, or using advanced textual features like edit distance.
"""


class QueryRewriteBasicFilters(str, Enum):
    DifferentQueryRewriteUtterance = 'diff_qr'
    RewriteNotSubstrOfQuery = 'rewrite_not_substr'
    QueryHasMultipleTokens = 'query_multi_tokens'
    RewriteHasMultipleTokens = 'rewrite_multi_tokens'


def query_rewrite_basic_filter(
        *filters: QueryRewriteBasicFilters,
        query_utterance_colname: str = c.KEY_REQUEST_FIRST,
        rewrite_utterance_basic_colname: str = c.KEY_REQUEST_SECOND
):
    conds = []
    for _filter in set(filters):
        if _filter == QueryRewriteBasicFilters.DifferentQueryRewriteUtterance:
            conds.append(
                (F.col(query_utterance_colname) != F.col(rewrite_utterance_basic_colname))
            )
        elif _filter == QueryRewriteBasicFilters.RewriteNotSubstrOfQuery:
            conds.append(
                (~F.col(query_utterance_colname).contains(F.col(rewrite_utterance_basic_colname)))
            )
        elif _filter == QueryRewriteBasicFilters.QueryHasMultipleTokens:
            conds.append(F.col(query_utterance_colname).contains(' '))
        elif _filter == QueryRewriteBasicFilters.RewriteHasMultipleTokens:
            conds.append(F.col(rewrite_utterance_basic_colname).contains(' '))
    return F.and_(*conds)


def utterance_length_filter(
        utterance_colname: str,
        min_length: int,
        max_length: int
):
    return (
            (F.length(utterance_colname) >= min_length) &
            (F.length(utterance_colname) <= max_length)
    )
