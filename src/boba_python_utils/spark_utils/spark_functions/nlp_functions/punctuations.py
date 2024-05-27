from functools import partial

import pyspark.sql.functions as F
from pyspark.sql.column import Column
from pyspark.sql.types import StringType
import re

from boba_python_utils.general_utils.nlp_utility.punctuations import (
    remove_punctuation as _remove_punctuation,
    remove_punctuation_except_for_hyphen as _remove_punctuation_except_for_hyphen,
    remove_punctuation_except_for_hyphen_and_underscore as _remove_punctuation_except_for_hyphen_and_underscore
)

REGEX_MATCH_ALL_PUNCTUATION_WITH_EXCEPTION_SPARK = r'(?=[\x00-\x7F])[^\w\s{}]'


def remove_punctuation(text, exception=None) -> Column:
    return F.trim(
        F.regexp_replace(
            F.regexp_replace(
                text,
                REGEX_MATCH_ALL_PUNCTUATION_WITH_EXCEPTION_SPARK.format(
                    re.escape(exception)
                    if exception
                    else ''
                ),
                ''
            ),
            r'\s+',
            replacement=' '
        )
    )


def remove_punctuation_except_for_hyphen(text) -> Column:
    return remove_punctuation(text, exception='-')


def remove_punctuation_except_for_hyphen_and_underscore(text) -> Column:
    return remove_punctuation(text, exception='-_')


def remove_punctuation_udf(text, exception=None):
    return F.udf(
        partial(_remove_punctuation, exception=exception), returnType=StringType()
    )(text)


remove_punctuation_except_for_hyphen_udf = F.udf(
    _remove_punctuation_except_for_hyphen, returnType=StringType()
)

remove_punctuation_except_for_hyphen_and_underscore_udf = F.udf(
    _remove_punctuation_except_for_hyphen_and_underscore, returnType=StringType()
)
