from enum import Enum
from typing import Union

import pyspark.sql.functions as F
from pyspark.sql import Column

import boba_python_utils.production_utils.pdfs.constants as c


class GlobalImpressionLevels(int, Enum):
    Low = 10
    Medium = 50
    High = 150
    Ultra = 500


class DefectReductionFilterLevels(str, Enum):
    Tiny = 'tiny'
    Low = 'low'
    Medium = 'medium'
    High = 'high'
    Ultra = 'ultra'


def pair_popularity_filter(
        filter_level: Union[str, DefectReductionFilterLevels],
        pair_popularity_colname=c.KEY_PAIR_POPULARITY,
        customer_pair_count_colname=c.KEY_CUSTOMER_PAIR_COUNT
):
    if filter_level == DefectReductionFilterLevels.Low:
        # low popularity threshold - presence in at least two customers' history, 
        # or one customer used it at least twice
        return (F.col(pair_popularity_colname) <= 1) & (F.col(customer_pair_count_colname) <= 1)
    elif filter_level == DefectReductionFilterLevels.Medium:
        # medium popularity threshold - presence in at least four customers' history
        return F.col(pair_popularity_colname) >= 4
    elif filter_level == DefectReductionFilterLevels.High:
        # high popularity threshold - presence in at least ten customers' history
        return F.col(pair_popularity_colname) >= 10

    raise ValueError(f"the filter level '{filter_level}' is not supported")


def global_defect_reduction_filter(
        filter_level: Union[str, DefectReductionFilterLevels],
        query_utterance_avg_defect_colname=c.KEY_GLOBAL_AVG_DEFECT_FIRST,
        rewrite_utterance_count_colname=c.KEY_GLOBAL_COUNT_SECOND,
        rewrite_utterance_global_avg_defect_colname=c.KEY_GLOBAL_AVG_DEFECT_SECOND,
        rewrite_utterance_customer_avg_defect_colname=c.KEY_CUSTOMER_AVG_DEFECT_SECOND
) -> Column:
    if filter_level == DefectReductionFilterLevels.Low:
        return ((
                        F.col(query_utterance_avg_defect_colname) -
                        F.col(rewrite_utterance_global_avg_defect_colname)
                ) > 0.05)
    if filter_level == DefectReductionFilterLevels.Medium:
        return ((
                        F.col(query_utterance_avg_defect_colname) -
                        F.col(rewrite_utterance_global_avg_defect_colname)
                ) > 0.10)
    if filter_level == DefectReductionFilterLevels.High:
        return ((
                        F.col(query_utterance_avg_defect_colname) -
                        F.col(rewrite_utterance_global_avg_defect_colname)
                ) > 0.20)
    if filter_level == DefectReductionFilterLevels.Ultra:
        return (
                (
                        (
                            (F.col(query_utterance_avg_defect_colname) -
                             F.col(rewrite_utterance_global_avg_defect_colname))
                        ) > 0.50
                ) &
                (F.col(rewrite_utterance_count_colname) > int(GlobalImpressionLevels.Medium)) &
                (F.col(rewrite_utterance_customer_avg_defect_colname) <= 0.2)
        )

    raise ValueError(f"the filter level '{filter_level}' is not supported")


def global_defect_reduction_pairwise_filter(
        filter_level: Union[str, DefectReductionFilterLevels],
        query_utterance_avg_defect_colname=c.KEY_GLOBAL_PAIR_AVG_DEFECT_FIRST,
        rewrite_utterance_avg_defect_colname=c.KEY_GLOBAL_PAIR_AVG_DEFECT_SECOND,
        pair_popularity_colname=c.KEY_PAIR_POPULARITY,
        customer_pair_count_colname=c.KEY_CUSTOMER_PAIR_COUNT
) -> Column:
    if filter_level == DefectReductionFilterLevels.Low:
        return (
                (
                    pair_popularity_filter(
                        DefectReductionFilterLevels.Low,
                        pair_popularity_colname=pair_popularity_colname,
                        customer_pair_count_colname=customer_pair_count_colname
                    )
                ) |
                (
                        (
                                F.col(query_utterance_avg_defect_colname) -
                                F.col(rewrite_utterance_avg_defect_colname)
                        ) <= 0.05
                )
        )
    if filter_level == DefectReductionFilterLevels.Medium:
        return (
                pair_popularity_filter(
                    DefectReductionFilterLevels.Medium,
                    pair_popularity_colname=pair_popularity_colname,
                    customer_pair_count_colname=customer_pair_count_colname
                ) &
                (
                        (
                                F.col(query_utterance_avg_defect_colname) -
                                F.col(rewrite_utterance_avg_defect_colname)
                        ) > 0.10
                )
        )

    raise ValueError(f"the filter level '{filter_level}' is not supported")


def global_defect_reduction_potential_filter(
        filter_level: Union[str, DefectReductionFilterLevels],
        query_utterance_avg_defect_colname=c.KEY_GLOBAL_AVG_DEFECT_FIRST,
        rewrite_utterance_avg_defect_colname=c.KEY_GLOBAL_AVG_DEFECT_SECOND
) -> Column:
    if filter_level == DefectReductionFilterLevels.Tiny:
        return (
                (F.col(query_utterance_avg_defect_colname) <= 0.10) |
                (F.col(rewrite_utterance_avg_defect_colname) >= 0.99)
        )
    raise ValueError(f"the filter level '{filter_level}' is not supported")


def global_rewrite_defect_reduction_potential_filter(
        filter_level: Union[str, DefectReductionFilterLevels],
        rewrite_avg_defect_colname=c.KEY_GLOBAL_AVG_DEFECT,
) -> Column:
    if filter_level == DefectReductionFilterLevels.Tiny:
        return F.col(rewrite_avg_defect_colname) > 0.95
    if filter_level == DefectReductionFilterLevels.Low:
        return F.col(rewrite_avg_defect_colname) > 0.75
    if filter_level == DefectReductionFilterLevels.Medium:
        return F.col(rewrite_avg_defect_colname) <= 0.55
    if filter_level == DefectReductionFilterLevels.High:
        return F.col(rewrite_avg_defect_colname) <= 0.35
    if filter_level == DefectReductionFilterLevels.Ultra:
        return F.col(rewrite_avg_defect_colname) <= 0.2

    raise ValueError(f"the filter level '{filter_level}' is not supported")


def global_confident_rewrite_filter(
        filter_level: Union[str, DefectReductionFilterLevels],
        rewrite_count_colname=c.KEY_GLOBAL_COUNT,
        global_rewrite_avg_defect_colname=c.KEY_GLOBAL_AVG_DEFECT
) -> Column:
    if filter_level == DefectReductionFilterLevels.High:
        return (
                (F.col(rewrite_count_colname) >= int(GlobalImpressionLevels.Medium)) &
                global_rewrite_defect_reduction_potential_filter(
                    filter_level=DefectReductionFilterLevels.Ultra,
                    rewrite_avg_defect_colname=global_rewrite_avg_defect_colname
                )
        )
    raise ValueError(f"the filter level '{filter_level}' is not supported")
