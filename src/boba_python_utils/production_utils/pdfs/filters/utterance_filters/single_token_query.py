from pyspark.sql import Column

from boba_python_utils.general_utils.nlp_utility.string_sanitization import StringSanitizationOptions
from boba_python_utils.spark_utils.spark_functions import edit_distance_udf
from boba_python_utils.spark_utils.spark_functions.common import or_, col_
from boba_python_utils.production_utils.pdfs.filters._configs.utterance_filter import RISK_SINGLE_TOKEN_NLU_HYPOTHESIS_SUBSTRINGS


def is_risk_single_token_utterance_udf(
        query_utterance_colname,
        rewrite_utterance_colname,
        query_nlu_hypothesis_colname,
        risk_hypothesis_sub_strs=RISK_SINGLE_TOKEN_NLU_HYPOTHESIS_SUBSTRINGS
) -> Column:
    return (
            (~col_(query_utterance_colname).contains(' ')) &
            or_(
                *(
                    col_(query_nlu_hypothesis_colname).contains(x)
                    for x in risk_hypothesis_sub_strs
                )
            ) &
            (
                    edit_distance_udf(
                        query_utterance_colname, rewrite_utterance_colname,
                        consider_sorted_tokens=True,
                        sanitization_config=[
                            StringSanitizationOptions.REMOVE_ACRONYMS_PERIODS_AND_SPACES,
                            StringSanitizationOptions.REMOVE_PUNCTUATIONS_EXCEPT_FOR_HYPHEN,
                            StringSanitizationOptions.MAKE_FUZZY,
                            StringSanitizationOptions.REMOVE_SPACES
                        ],
                        return_ratio=True
                    ) <= 0.5
            )
    )
