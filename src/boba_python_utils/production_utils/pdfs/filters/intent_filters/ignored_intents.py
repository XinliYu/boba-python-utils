from typing import Iterable

from boba_python_utils.spark_utils.spark_functions.common import col_
from boba_python_utils.spark_utils.typing import NameOrColumn
from boba_python_utils.production_utils.pdfs.filters._configs.intent_filter import IGNORED_INTENTS_FOR_REPHRASE


def is_ignored_intent(
        query_intent: str,
        ignored_intents: Iterable[str] = IGNORED_INTENTS_FOR_REPHRASE
):
    """
    Checks if an NLU intent is in the "ignore" list.

    Examples:
         >>> assert is_ignored_intent('StopIntent')

    """
    return query_intent in ignored_intents


def is_ignored_intent_udf(
        query_intent_col: NameOrColumn,
        ignored_intents: Iterable[str] = IGNORED_INTENTS_FOR_REPHRASE
):
    if isinstance(ignored_intents, set):
        return col_(query_intent_col).isin(ignored_intents)
    else:
        return col_(query_intent_col).isin(set(ignored_intents))
