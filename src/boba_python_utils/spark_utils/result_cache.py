import uuid
from os import path
from typing import Callable, Iterable

from pyspark.sql import DataFrame, SparkSession

from boba_python_utils.common_utils.typing_helper import is_none_or_empty_str
from boba_python_utils.spark_utils.common import CacheOptions, repartition
from boba_python_utils.spark_utils.data_loading import solve_input, cache__
from boba_python_utils.spark_utils.data_writing import write_df
from boba_python_utils.spark_utils.join_and_filter import (
    filter_by_inner_join_on_columns, exclude_by_anti_join_on_columns
)
from boba_python_utils.spark_utils.aggregation import union


def process_data_with_result_cache(
        df_input: DataFrame,
        process: Callable,
        result_cache,
        result_cache_key_colnames: Iterable[str],
        spark: SparkSession,
        num_result_cache_files: int = 1200,
        cache_option: CacheOptions = CacheOptions.IMMEDIATE
):
    result_cache_key_colnames = list(result_cache_key_colnames)

    df_result_from_cache = None
    _result_cache = result_cache
    if not is_none_or_empty_str(result_cache):
        if isinstance(result_cache, str):
            _result_cache = path.join(result_cache, '*')

    try:
        df_cache = solve_input(
            _result_cache,
            repartition=result_cache_key_colnames,
            spark=spark
        )
    except Exception as err:
        if isinstance(result_cache, str):
            # in case the path is not available
            df_cache = None
        else:
            raise err

    if df_cache is not None:
        df_result_from_cache, size_df_result_from_cache = cache__(
            filter_by_inner_join_on_columns(
                df_cache,
                df_input,
                result_cache_key_colnames,
            ),
            name='df_result_from_cache',
            return_count=True,
            cache_option=cache_option
        )
        if size_df_result_from_cache != 0:
            df_input, size_df_input_new = cache__(
                exclude_by_anti_join_on_columns(
                    df_input,
                    df_cache,
                    result_cache_key_colnames
                ),
                name='df_input (not in cache)',
                unpersist=df_input,
                return_count=True,
                cache_option=cache_option
            )

            if size_df_input_new == 0:
                df_input = None

    if df_input is None:
        if df_result_from_cache is None:
            raise ValueError("input is empty")
        df_result = df_result_from_cache
    else:
        df_result = process(df_input)
        if isinstance(result_cache, str):
            write_df(
                df_result,
                path.join(result_cache, str(uuid.uuid4())),
                num_files=num_result_cache_files
            )

        if df_result_from_cache is not None:
            df_result = cache__(
                repartition(
                    union(
                        df_result,
                        df_result_from_cache
                    ),
                    None,
                    *result_cache_key_colnames,
                    spark=spark
                ),
                name='df_result (union with df_result_from_cache)',
                unpersist=(
                    df_result,
                    df_result_from_cache
                ),
                cache_option=cache_option
            )

    return df_result
