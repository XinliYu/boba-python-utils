from functools import partial
from typing import Union, Iterable, Callable

import pyspark.sql.functions as F
from pyspark.sql.column import Column
from pyspark.sql.types import IntegerType, FloatType

from boba_python_utils.general_utils.nlp_utility.metrics.edit_distance import edit_distance, EditDistanceOptions
from boba_python_utils.general_utils.nlp_utility.string_sanitization import StringSanitizationOptions


def edit_distance_udf(
        str1: Union[str, Column],
        str2: Union[str, Column],
        return_ratio=True,
        consider_sorted_tokens: Union[bool, Callable] = False,
        consider_same_num_tokens: Union[bool, Callable] = False,
        options: EditDistanceOptions = None,
        sanitization_config: Iterable[StringSanitizationOptions] = None,
        tokenizer=None,
        **kwargs
):
    return F.udf(
        partial(
            edit_distance,
            return_ratio=return_ratio,
            consider_sorted_tokens=consider_sorted_tokens,
            consider_same_num_tokens=consider_same_num_tokens,
            options=options,
            sanitization_config=sanitization_config,
            tokenizer=tokenizer,
            **kwargs
        ),
        returnType=FloatType() if return_ratio else IntegerType()
    )(str1, str2)
