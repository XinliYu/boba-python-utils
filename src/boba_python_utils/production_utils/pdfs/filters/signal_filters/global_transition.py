from typing import Union

from pyspark.sql import Column, functions as F

from boba_python_utils.production_utils.pdfs import constants as c
from boba_python_utils.production_utils.pdfs.filters.signal_filters.global_defect_reduction import DefectReductionFilterLevels


def global_transition_count_filter(
        filter_level: Union[str, DefectReductionFilterLevels],
        pair_count_colname=c.KEY_GLOBAL_PAIR_COUNT
) -> Column:
    if filter_level == DefectReductionFilterLevels.Tiny:
        return F.col(pair_count_colname) <= 1
    if filter_level == DefectReductionFilterLevels.Low:
        return F.col(pair_count_colname) < 4
    if filter_level == DefectReductionFilterLevels.Medium:
        return F.col(pair_count_colname) >= 20
    if filter_level == DefectReductionFilterLevels.High:
        return F.col(pair_count_colname) >= 30
    raise ValueError(f"the filter level '{filter_level}' is not supported")
