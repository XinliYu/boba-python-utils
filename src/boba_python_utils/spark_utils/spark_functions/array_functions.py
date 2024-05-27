from functools import partial
from typing import Union, Optional

from pyspark.sql import DataFrame
from pyspark.sql.column import Column
from pyspark.sql.functions import when, array_union, udf, array_contains, lit, aggregate, expr, flatten, collect_list, array_distinct, slice
from pyspark.sql.types import ArrayType
import random
from boba_python_utils.spark_utils.common import get_coltype, get_spark_type
from boba_python_utils.spark_utils.spark_functions.common import col_
from boba_python_utils.spark_utils.typing import NameOrColumn


def array_contains_(col: NameOrColumn, value):
    return col_(col).isNotNull() & array_contains(col, value)


def array_union_(col1: NameOrColumn, col2: NameOrColumn):
    return when(
        col_(col1).isNull(), col_(col2)
    ).otherwise(
        when(
            col_(col2).isNull(), col_(col1)
        ).otherwise(
            array_union(col1, col2)
        )
    )


def array_sum(arr_col: Union[str, Column], init_value=0.0, field_name=None, final=None):
    init_value_type = get_spark_type(init_value)
    if field_name:
        return aggregate(
            arr_col,
            lit(init_value),
            lambda acc, x: acc + x[field_name].cast(init_value_type),
            final
        )
    else:
        return aggregate(
            arr_col,
            lit(init_value),
            lambda acc, x: acc + x.cast(init_value_type),
            final
        )


def _array_shuffle(arr):
    random.shuffle(arr)
    return arr


def array_shuffle(arr_col: NameOrColumn, df: DataFrame):
    arr_coltype = get_coltype(df, arr_col)
    return udf(_array_shuffle, returnType=arr_coltype)(arr_col)


def _array_shuffle_with_sample(arr, sample_size: int, seed: int = None):
    if seed is not None:
        random.seed(seed)
    random.shuffle(arr)
    if len(arr) > sample_size:
        return arr[:sample_size]
    else:
        return arr


def array_shuffle_with_sample(arr_col: NameOrColumn, sample_size: int, df: DataFrame, seed: int = None):
    arr_coltype = get_coltype(df, arr_col)
    return udf(
        partial(_array_shuffle_with_sample, sample_size=sample_size, seed=seed),
        returnType=arr_coltype
    )(arr_col)


def array_concat(
        arr_col: NameOrColumn,
        distinct: bool = False,
        concat_max_size: Optional[int] = None
) -> Column:
    if concat_max_size:
        arr_col = flatten(slice(collect_list(arr_col), 1, concat_max_size))
        if distinct:
            arr_col = slice(array_distinct(arr_col), 1, concat_max_size)
    else:
        arr_col = flatten(collect_list(arr_col))
        if distinct:
            arr_col = array_distinct(arr_col)

    return arr_col


def _array_concat_shuffle_and_sample(arr_col_collected: NameOrColumn, sample_size: int, distinct: bool = False, seed: int = None):
    if distinct:
        arr_col_collected_flattened = list(set(sum(arr_col_collected, [])))
    else:
        arr_col_collected_flattened = sum(arr_col_collected, [])
    return _array_shuffle_with_sample(arr_col_collected_flattened, sample_size=sample_size, seed=seed)


def array_concat_shuffle_and_sample(arr_col: NameOrColumn, sample_size: int, df: DataFrame, distinct: bool = False, seed: int = None):
    arr_coltype = get_coltype(df, arr_col)
    return udf(
        partial(_array_concat_shuffle_and_sample, sample_size=sample_size, distinct=distinct, seed=seed),
        returnType=arr_coltype
    )(collect_list(arr_col))
