from functools import partial

from pyspark.sql.functions import udf
from pyspark.sql.types import BooleanType

from boba_python_utils.spark_utils.spark_functions.common import or_, col_
from boba_python_utils.production_utils.common._constants.path import SupportedLocales
from boba_python_utils.production_utils.pdfs.filters._configs.intent_filter import INVALID_INTENT_PAIRS, INVALID_INTENT_PAIRS_BY_LOCALE, INVALID_INTENT_TRANSITION


def has_opposite_intents(
        request_intent,
        rewrite_intent,
        opposite_intents=INVALID_INTENT_PAIRS,
        opposite_intents_by_locale=INVALID_INTENT_PAIRS_BY_LOCALE,
        locale: SupportedLocales = None
):
    return (
            ((request_intent, rewrite_intent) in opposite_intents) or
            ((rewrite_intent, request_intent) in opposite_intents) or
            (
                # ! the following 'bool' conversion is needed for Spark udf wrap;
                # ! otherwise, None will be returned rather than False when `locale`
                # ! or `opposite_intents_by_locale` are not specified
                    bool(locale) and
                    bool(opposite_intents_by_locale) and
                    locale in opposite_intents_by_locale and
                    (
                            (request_intent, rewrite_intent) in opposite_intents_by_locale[locale] or
                            (rewrite_intent, request_intent) in opposite_intents_by_locale[locale]
                    )
            )
    )


def has_opposite_intent_udf(
        query_nlu_intent_colname,
        rewrite_nlu_intent_colname,
        opposite_intents=INVALID_INTENT_PAIRS,
        opposite_intents_by_locale=INVALID_INTENT_PAIRS_BY_LOCALE,
        locale: SupportedLocales = None,
        native_spark_compute=False,
        exception=None
):
    if native_spark_compute:
        return or_(
            *(
                (
                        (
                                (col_(query_nlu_intent_colname) == intent1) &
                                (col_(rewrite_nlu_intent_colname) == intent2)
                        ) |
                        (
                                (col_(query_nlu_intent_colname) == intent2) &
                                (col_(rewrite_nlu_intent_colname) == intent1)
                        )
                )
                for intent1, intent2 in opposite_intents
            ),
            *(
                (
                    (
                            (col_(query_nlu_intent_colname) == intent1) &
                            (col_(rewrite_nlu_intent_colname) == intent2)
                    ) |
                    (
                            (col_(query_nlu_intent_colname) == intent2) &
                            (col_(rewrite_nlu_intent_colname) == intent1)
                    )
                    for intent1, intent2 in opposite_intents_by_locale[locale]
                )
                if locale and opposite_intents_by_locale and locale in opposite_intents_by_locale
                else ()
            )
        )
    else:
        result = (
            udf(
                partial(
                    has_opposite_intents,
                    opposite_intents=opposite_intents,
                    opposite_intents_by_locale=opposite_intents_by_locale,
                    locale=locale
                ),
                returnType=BooleanType()
            )(
                query_nlu_intent_colname, rewrite_nlu_intent_colname
            )
        )

    if exception is not None:
        result = result & (~exception)
    return result


def has_invalid_intent_transition_udf(
        query_nlu_intent_colname,
        rewrite_nlu_intent_colname,
        invalid_intent_transitions=INVALID_INTENT_TRANSITION,
):
    return or_(
        *(
            (
                    (col_(query_nlu_intent_colname) == intent1) &
                    (col_(rewrite_nlu_intent_colname) == intent2)
            )
            for intent1, intent2 in invalid_intent_transitions
        )
    )

# if __name__ == '__main__':
#     assert has_opposite_intents(
#         request_intent='AddToPlaylistIntent',
#         rewrite_intent='PlayMusicIntent',
#         locale=SupportedLocales.EN_US
#     )
#     assert has_opposite_intents(
#         request_intent='AddToPlaylistIntent',
#         rewrite_intent='PlayMusicIntent',
#         locale=SupportedLocales.EN_GB
#     )
#     assert not has_opposite_intents(
#         request_intent='ReadBookIntent',
#         rewrite_intent='NavigateBooksIntent',
#         locale=SupportedLocales.EN_US
#     )
#     assert has_opposite_intents(
#         request_intent='ReadBookIntent',
#         rewrite_intent='NavigateBooksIntent',
#         locale=SupportedLocales.EN_GB
#     )
#     assert not has_opposite_intents(
#         request_intent='ReadBookIntent',
#         rewrite_intent='NavigateBooksIntent',
#         opposite_intents_by_locale=None,
#         locale=SupportedLocales.EN_GB
#     )
#     assert not has_opposite_intents(
#         request_intent='ReadBookIntent',
#         rewrite_intent='NavigateBooksIntent',
#         opposite_intents_by_locale=None,
#         locale=None
#     )
#     assert has_opposite_intents(
#         request_intent='SetMusicNotificationIntent',
#         rewrite_intent='PlayMusicIntent',
#         opposite_intents=RISKY_INTENT_PAIRS,
#         opposite_intents_by_locale=None
#     )
