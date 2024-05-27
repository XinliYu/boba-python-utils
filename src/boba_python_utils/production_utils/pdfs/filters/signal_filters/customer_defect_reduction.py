from enum import Enum
from typing import Union

import pyspark.sql.functions as F
from pyspark.sql import Column

import boba_python_utils.production_utils.pdfs.constants as c
from boba_python_utils.production_utils.pdfs.filters.signal_filters.global_defect_reduction \
    import DefectReductionFilterLevels, global_rewrite_defect_reduction_potential_filter


class CustomerImpressionLevels(int, Enum):
    Low = 1
    Medium = 2
    High = 3
    Ultra = 4


def customer_defect_reduction_potential_filter(
        filter_level: Union[str, DefectReductionFilterLevels],
        query_utterance_count_colname=c.KEY_CUSTOMER_COUNT_FIRST,
        query_utterance_avg_defect_colname=c.KEY_CUSTOMER_AVG_DEFECT_FIRST
) -> Column:
    if filter_level == DefectReductionFilterLevels.Tiny:
        return F.col(query_utterance_avg_defect_colname) <= 0.05
    if filter_level == DefectReductionFilterLevels.Low:
        return (
                (
                        (F.col(query_utterance_count_colname) >= int(CustomerImpressionLevels.Medium)) &
                        (F.col(query_utterance_avg_defect_colname) <= 0.15)
                ) |
                (
                        (F.col(query_utterance_count_colname) >= int(CustomerImpressionLevels.High)) &
                        (F.col(query_utterance_avg_defect_colname) <= 0.2)
                )
        )
    if filter_level == DefectReductionFilterLevels.High:
        return (
                (F.col(query_utterance_count_colname) >= int(CustomerImpressionLevels.Medium)) &
                (F.col(query_utterance_avg_defect_colname) > 0.30)
        )
    raise ValueError(f"the filter level '{filter_level}' is not supported")


def customer_min_defect_reduction_filter(
        filter_level: Union[str, DefectReductionFilterLevels],
        query_utterance_avg_defect_colname=c.KEY_CUSTOMER_AVG_DEFECT_FIRST,
        rewrite_utterance_count_colname=c.KEY_CUSTOMER_COUNT_SECOND,
        rewrite_utterance_avg_defect_colname=c.KEY_CUSTOMER_AVG_DEFECT_SECOND
) -> Column:
    if filter_level == DefectReductionFilterLevels.Low:
        return (
                F.col(query_utterance_avg_defect_colname) <
                F.col(rewrite_utterance_avg_defect_colname)
        )
    if filter_level == DefectReductionFilterLevels.Medium:
        return ((
                        F.col(query_utterance_avg_defect_colname) -
                        F.col(rewrite_utterance_avg_defect_colname)
                ) > 0.05)
    if filter_level == DefectReductionFilterLevels.High:
        return ((
                        F.col(query_utterance_avg_defect_colname) -
                        F.col(rewrite_utterance_avg_defect_colname)
                ) > 0.1)
    if filter_level == DefectReductionFilterLevels.Ultra:
        return (
                (
                        (
                                F.col(c.KEY_CUSTOMER_AVG_DEFECT_FIRST) -
                                F.col(c.KEY_CUSTOMER_AVG_DEFECT_SECOND)
                        ) > 0.15
                ) &
                customer_defect_reduction_guardrail_filter(
                    DefectReductionFilterLevels.Low,
                    query_utterance_avg_defect_colname=query_utterance_avg_defect_colname,
                    rewrite_utterance_count_colname=rewrite_utterance_count_colname,
                    rewrite_utterance_avg_defect_colname=rewrite_utterance_avg_defect_colname
                )
        )

    raise ValueError(f"the filter level '{filter_level}' is not supported")


def customer_defect_reduction_pairwise_filter(
        filter_level: Union[str, DefectReductionFilterLevels],
        query_utterance_avg_defect_colname=c.KEY_CUSTOMER_PAIR_AVG_DEFECT_FIRST,
        rewrite_utterance_avg_defect_colname=c.KEY_CUSTOMER_PAIR_AVG_DEFECT_SECOND,
        pair_count_colname=c.KEY_CUSTOMER_PAIR_COUNT,
) -> Column:
    if filter_level == DefectReductionFilterLevels.Low:
        return (
                F.col(query_utterance_avg_defect_colname) <=
                F.col(rewrite_utterance_avg_defect_colname)
        )
    if filter_level == DefectReductionFilterLevels.Medium:
        return (
                (F.col(pair_count_colname) >= int(CustomerImpressionLevels.Medium)) &
                (
                        (
                                F.col(query_utterance_avg_defect_colname) -
                                F.col(rewrite_utterance_avg_defect_colname)
                        ) > 0.05
                )
        )
    if filter_level == DefectReductionFilterLevels.High:
        return (
                (F.col(pair_count_colname) >= int(CustomerImpressionLevels.Medium)) &
                (
                        (
                                F.col(query_utterance_avg_defect_colname) -
                                F.col(rewrite_utterance_avg_defect_colname)
                        ) > 0.10
                )
        )

    raise ValueError(f"the filter level '{filter_level}' is not supported")


def customer_defect_reduction_guardrail_filter(
        filter_level: Union[str, DefectReductionFilterLevels],
        query_utterance_avg_defect_colname=c.KEY_CUSTOMER_AVG_DEFECT_FIRST,
        rewrite_utterance_count_colname=c.KEY_CUSTOMER_COUNT_SECOND,
        rewrite_utterance_avg_defect_colname=c.KEY_CUSTOMER_AVG_DEFECT_SECOND
) -> Column:
    if filter_level == DefectReductionFilterLevels.Low:
        # rewrite has low defect <=0.1, and the harm should be low
        return (
                (F.col(rewrite_utterance_count_colname) >= int(CustomerImpressionLevels.Medium)) &
                (F.col(rewrite_utterance_avg_defect_colname) <= 0.1)
        )
    elif filter_level == DefectReductionFilterLevels.Medium:
        # rewrite has high defect >0.5, and there should be risk of harm
        return F.col(rewrite_utterance_avg_defect_colname) > 0.5
    elif filter_level == DefectReductionFilterLevels.High:
        # rewrite is observed to cause regression
        return (
                F.col(query_utterance_avg_defect_colname) -
                F.col(rewrite_utterance_avg_defect_colname) <= -0.1
        )
    elif filter_level == DefectReductionFilterLevels.Ultra:
        # rewrite is observed to have nearly no success
        return F.col(rewrite_utterance_avg_defect_colname) >= 0.99

    raise ValueError(f"the filter level '{filter_level}' is not supported")


def customer_defect_reduction_pairwise_guardrail_filter(
        filter_level: Union[str, DefectReductionFilterLevels],
        pair_count_colname=c.KEY_CUSTOMER_PAIR_COUNT,
        query_utterance_avg_defect_colname=c.KEY_CUSTOMER_PAIR_AVG_DEFECT_FIRST,
        rewrite_utterance_avg_defect_colname=c.KEY_CUSTOMER_PAIR_AVG_DEFECT_SECOND
) -> Column:
    if filter_level == DefectReductionFilterLevels.High:
        return (
                (F.col(pair_count_colname) >= int(CustomerImpressionLevels.Medium)) &
                (
                        F.col(query_utterance_avg_defect_colname) <
                        F.col(rewrite_utterance_avg_defect_colname)
                )
        )
    raise ValueError(f"the filter level '{filter_level}' is not supported")


def customer_query_defect_reduction_potential_filter(
        filter_level: Union[str, DefectReductionFilterLevels],
        avg_defect_colname=c.KEY_CUSTOMER_AVG_DEFECT,
) -> Column:
    if filter_level == DefectReductionFilterLevels.Tiny:
        return F.col(avg_defect_colname) <= 0.10
    if filter_level == DefectReductionFilterLevels.Low:
        return F.col(avg_defect_colname) <= 0.18
    if filter_level == DefectReductionFilterLevels.Medium:
        return F.col(avg_defect_colname) > 0.30
    if filter_level == DefectReductionFilterLevels.High:
        return F.col(avg_defect_colname) > 0.45
    if filter_level == DefectReductionFilterLevels.Ultra:
        return F.col(avg_defect_colname) > 0.60

    raise ValueError(f"the filter level '{filter_level}' is not supported")


def customer_rewrite_defect_reduction_potential_filter(
        filter_level: Union[str, DefectReductionFilterLevels],
        rewrite_avg_defect_colname=c.KEY_CUSTOMER_AVG_DEFECT,
) -> Column:
    if filter_level == DefectReductionFilterLevels.Tiny:
        return F.col(rewrite_avg_defect_colname) > 0.8
    if filter_level == DefectReductionFilterLevels.Low:
        return F.col(rewrite_avg_defect_colname) > 0.7
    if filter_level == DefectReductionFilterLevels.Medium:
        return F.col(rewrite_avg_defect_colname) <= 0.5
    if filter_level == DefectReductionFilterLevels.High:
        return F.col(rewrite_avg_defect_colname) <= 0.3
    if filter_level == DefectReductionFilterLevels.Ultra:
        return F.col(rewrite_avg_defect_colname) <= 0.2

    raise ValueError(f"the filter level '{filter_level}' is not supported")


def customer_confident_rewrite_filter(
        filter_level: Union[str, DefectReductionFilterLevels],
        rewrite_count_colname=c.KEY_CUSTOMER_COUNT,
        customer_rewrite_avg_defect_colname=c.KEY_CUSTOMER_AVG_DEFECT,
        global_rewrite_avg_defect_colname=c.KEY_CUSTOMER_AVG_DEFECT
) -> Column:
    if filter_level == DefectReductionFilterLevels.High:
        return (
                (F.col(rewrite_count_colname) >= int(CustomerImpressionLevels.High)) &
                customer_rewrite_defect_reduction_potential_filter(
                    filter_level=DefectReductionFilterLevels.Ultra,
                    rewrite_avg_defect_colname=customer_rewrite_avg_defect_colname
                ) &
                global_rewrite_defect_reduction_potential_filter(
                    filter_level=DefectReductionFilterLevels.High,
                    rewrite_avg_defect_colname=global_rewrite_avg_defect_colname
                )
        )
    raise ValueError(f"the filter level '{filter_level}' is not supported")
