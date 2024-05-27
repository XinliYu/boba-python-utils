from functools import partial
from typing import Iterable, Mapping

from pyspark.sql.functions import udf
from pyspark.sql.types import BooleanType

from boba_python_utils.string_utils.common import contains_any_all
from boba_python_utils.production_utils.pdfs.filters._configs.intent_filter import (
    RISK_INTENT_SPECIFIC_REQUEST_KEYWORDS
)


def has_risk_keywords_under_intent(
        intent: str,
        utterance: str,
        risk_intent_keywords: Mapping[str, Iterable[str]] = RISK_INTENT_SPECIFIC_REQUEST_KEYWORDS
) -> bool:
    """
    Given an NLU intent, checks if the utterance contains certain keywords
    associated with the intent.

    Examples:
        >>> has_risk_keywords_under_intent('QAIntent', 'how to spell granny')
        True

    """
    if intent:
        for risk_intent, keywords in risk_intent_keywords.items():
            if risk_intent in intent:
                if contains_any_all(utterance, keywords):
                    return True
    return False


def has_risk_keywords_under_intent_udf(
        intent_colname,
        utterance_colname,
        risk_intent_keywords: Mapping[str, Iterable[str]] = RISK_INTENT_SPECIFIC_REQUEST_KEYWORDS
):
    return udf(
        partial(
            has_risk_keywords_under_intent,
            risk_intent_keywords=risk_intent_keywords
        ),
        returnType=BooleanType()
    )(intent_colname, utterance_colname)
