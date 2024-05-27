from pyspark.sql import DataFrame

from boba_python_utils.common_utils.iter_helper import iter__
from boba_python_utils.spark_utils.data_transform import with_columns
from boba_python_utils.spark_utils.spark_functions.string_functions import regexp_replace_many
from boba_python_utils.production_utils._nlu.hypothesis_parsing import get_domain_from_hypothesis, get_intent_from_hypothesis
from boba_python_utils.production_utils.pdfs.filters._configs.utterance_filter import (
    UTTERANCE_REPLACE, PRE_DEFINED_REWRITES
)
import boba_python_utils.spark_utils.spark_functions as F


def sanitize_rewrite_udf(rewrite_utterance, replaces=UTTERANCE_REPLACE):
    return regexp_replace_many(rewrite_utterance, replaces)


def with_pre_defined_rewrites(
        df: DataFrame,
        query_colname,
        rewrite_colname,
        rewrite_nlu_hypothesis_colname,
        rewrite_domain_colname,
        rewrite_intent_colname,
        reset_colnames,
        reset_value,
        enable_cond,
        predefined_rewrites=PRE_DEFINED_REWRITES
):
    for query, rewrite, rewrite_nlu_hypothesis in predefined_rewrites:
        _enable_cond = F.and_(
            F.col(query_colname) == query,
            F.col_(enable_cond)
        )
        df = with_columns(
            df,
            {
                rewrite_colname: F.when(
                    _enable_cond,
                    F.lit(rewrite)
                ).otherwise(F.col(rewrite_colname)),
                rewrite_nlu_hypothesis_colname: F.when(
                    _enable_cond,
                    F.lit(rewrite_nlu_hypothesis)
                ).otherwise(F.col(rewrite_nlu_hypothesis_colname)),
                rewrite_domain_colname: F.when(
                    _enable_cond,
                    F.lit(
                        get_domain_from_hypothesis(
                            rewrite_nlu_hypothesis,
                            third_party_skill_as_one_domain=False
                        )
                    )
                ).otherwise(F.col(rewrite_domain_colname)),
                rewrite_intent_colname: F.when(
                    _enable_cond,
                    F.lit(
                        get_intent_from_hypothesis(rewrite_nlu_hypothesis)
                    )
                ).otherwise(F.col(rewrite_intent_colname)),
                **{
                    colname: F.when(
                        _enable_cond,
                        F.lit(reset_value)
                    ).otherwise(F.col(colname))
                    for colname in iter__(reset_colnames)
                }
            }
        )
    return df
