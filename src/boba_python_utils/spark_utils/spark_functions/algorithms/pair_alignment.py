from functools import partial
from typing import Iterable, Any, Mapping, Union, Callable

from pyspark.sql import Column
from pyspark.sql.functions import udf
from pyspark.sql.types import StructField, StringType, FloatType, ArrayType, StructType

from boba_python_utils.common_utils.function_helper import compose
from boba_python_utils.general_utils.nlp_utility.metrics.edit_distance import EditDistanceOptions
from boba_python_utils.general_utils.nlp_utility.string_sanitization import StringSanitizationOptions, StringSanitizationConfig

from boba_python_utils.general_utils.algorithms.pair_alignment import iter_aligned_pairs_by_edit_distance_with_distinct_source


def align_pairs_of_distinct_source_by_edit_distance_udf(
        arr_col1, arr_col2,
        result_source_colname,
        result_target_colname,
        result_score_colname,
        source_str_getter: Callable[[Any], str] = None,
        target_str_getter: Callable[[Any], str] = None,
        source_date_type=None,
        target_data_type=None,
        enabled_target_items: Union[Iterable, Mapping[Any, int]] = None,
        enabled_source_items: Iterable = None,
        include_all_tie_target_items: bool = True,
        reversed_sort_by_scores: bool = False,
        identity_score=None,
        # region edit distance config
        sanitization_config: Union[Iterable[StringSanitizationOptions], StringSanitizationConfig] = None,
        edit_distance_consider_sorted_tokens: Union[bool, Callable] = min,
        edit_distance_options: EditDistanceOptions = None,
        **edit_distance_kwargs,
        # endregion
) -> Column:
    if source_str_getter is not None and source_date_type is None:
        raise ValueError()
    if target_str_getter is not None and target_data_type is None:
        raise ValueError()

    return_align_score = bool(result_score_colname)
    return_fields = [
        StructField(name=result_source_colname, dataType=(source_date_type or StringType())),
        StructField(name=result_target_colname, dataType=(target_data_type or StringType())),
    ]
    if return_align_score:
        return_fields = [
            StructField(name=result_score_colname, dataType=FloatType()),
            *return_fields
        ]

    return udf(
        compose(
            list,
            partial(
                iter_aligned_pairs_by_edit_distance_with_distinct_source,
                x_str_getter=source_str_getter,
                y_str_getter=target_str_getter,
                enabled_target_items=enabled_target_items,
                enabled_source_items=enabled_source_items,
                include_all_tie_target_items=include_all_tie_target_items,
                reversed_sort_by_scores=reversed_sort_by_scores,
                identity_score=identity_score,
                sanitization_config=sanitization_config,
                edit_distance_consider_sorted_tokens=edit_distance_consider_sorted_tokens,
                edit_distance_options=edit_distance_options,
                **edit_distance_kwargs
            )
        ),
        returnType=ArrayType(StructType(return_fields))
    )(arr_col1, arr_col2)
