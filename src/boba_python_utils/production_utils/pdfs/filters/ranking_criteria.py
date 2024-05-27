from typing import Tuple

from pyspark.sql import Column

from boba_python_utils.spark_utils.spark_functions.common import col_
from boba_python_utils.spark_utils.typing import NameOrColumn


def get_customer_rank_criteria(
        count_col: NameOrColumn, defect_col: NameOrColumn
) -> Tuple[Column, ...]:
    count_col = col_(count_col)
    defect_col = col_(defect_col)
    return (
        (count_col * (1 - defect_col)).desc(),
        count_col.desc(),
        defect_col
    )


def get_global_rank_criteria(
        count_col: NameOrColumn, defect_col: NameOrColumn
) -> Tuple[Column, ...]:
    count_col = col_(count_col)
    defect_col = col_(defect_col)
    return (
        (count_col * (1 - defect_col)).desc(),
        defect_col,
        count_col.desc(),
    )
