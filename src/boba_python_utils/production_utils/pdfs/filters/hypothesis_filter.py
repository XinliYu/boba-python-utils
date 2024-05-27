from pyspark.sql import Column

from boba_python_utils.common_utils.iter_helper import iter__
from boba_python_utils.general_utils.nlp_utility.string_sanitization import StringSanitizationOptions
from boba_python_utils.spark_utils.spark_functions import edit_distance_udf, string_check
from boba_python_utils.spark_utils.spark_functions.common import col_, and_, or_
from boba_python_utils.string_utils.comparison import string_check as string_check_str
from boba_python_utils.production_utils.pdfs.filters._configs.hypothesis_filter import *


def is_ignored_nlu_hypothesis_for_query(nlu_hypothesis: str, locale: SupportedLocales = None):
    """
    Examples:
        >>> assert not is_ignored_nlu_hypothesis_for_query('Books|ReadBookIntent|GenreName:bedtime story')
        >>> assert is_ignored_nlu_hypothesis_for_query('Books|ReadBookIntent|GenreName:bedtime story', locale=SupportedLocales.EN_GB)
    """
    return (
            nlu_hypothesis in IGNORED_HYPOTHESIS_FOR_QUERIES or
            (
                    bool(locale) and
                    locale in IGNORED_HYPOTHESIS_FOR_QUERIES_BY_LOCALE and
                    IGNORED_HYPOTHESIS_FOR_QUERIES_BY_LOCALE[locale]
            )
    )


def is_ignored_nlu_hypothesis_for_rewrite(nlu_hypothesis: str, locale: SupportedLocales = None):
    """
    Examples:
        >>> assert not is_ignored_nlu_hypothesis_for_query('Game|LaunchGameIntent|MediaType:game')
        >>> assert is_ignored_nlu_hypothesis_for_query('Game|LaunchGameIntent|MediaType:game', locale=SupportedLocales.EN_US)
    """
    return (
            nlu_hypothesis in IGNORED_HYPOTHESIS_FOR_REWRITES or
            (
                    bool(locale) and
                    locale in IGNORED_HYPOTHESIS_FOR_REWRITES_BY_LOCALE and
                    IGNORED_HYPOTHESIS_FOR_REWRITES_BY_LOCALE[locale]
            )
    )


def is_ignored_nlu_hypothesis_for_query_udf(
        nlu_hypothesis_col,
        locale: SupportedLocales = None
) -> Column:
    cond = col_(nlu_hypothesis_col).isin(IGNORED_HYPOTHESIS_FOR_QUERIES)
    if locale and locale in IGNORED_HYPOTHESIS_FOR_QUERIES_BY_LOCALE:
        cond = cond | col_(nlu_hypothesis_col).isin(IGNORED_HYPOTHESIS_FOR_QUERIES_BY_LOCALE[locale])
    return cond


def is_ignored_nlu_hypothesis_for_rewrite_udf(
        nlu_hypothesis,
        locale: SupportedLocales = None
) -> Column:
    cond = col_(nlu_hypothesis).isin(IGNORED_HYPOTHESIS_FOR_REWRITES)
    if locale and locale in IGNORED_HYPOTHESIS_FOR_REWRITES_BY_LOCALE:
        cond = cond | col_(nlu_hypothesis).isin(IGNORED_HYPOTHESIS_FOR_REWRITES_BY_LOCALE[locale])
    return cond


def has_risk_hypothesis_change(
        query_utterance,
        rewrite_utterance,
        query_hypothesis,
        rewrite_hypothesis,
        risk_hypothesis_drop=RISK_HYPOTHESIS_DROP,
        risk_hypothesis_addition=RISK_HYPOTHESIS_ADDITION,
        risk_hypothesis_change=RISK_HYPOTHESIS_CHANGE,
        allows_rewrite_contains_query=True
):
    return (query_hypothesis != rewrite_hypothesis) and any(
        (
            (
                    any(
                        all(
                            (
                                *(
                                    (
                                        any(

                                            _x in query_hypothesis
                                            for _x in iter__(x)

                                        )
                                    ) for x in _risk_hypothesis_drop
                                ),
                                *(
                                    (
                                        not any(

                                            _x in rewrite_hypothesis
                                            for _x in iter__(x)

                                        )
                                    ) for x in _risk_hypothesis_drop
                                )
                            )
                        )
                        for _risk_hypothesis_drop in iter__(risk_hypothesis_drop)
                    ) and
                    (query_utterance not in rewrite_utterance)
            ),
            any(
                all(
                    (
                        *(
                            (
                                not any(

                                    _x in query_hypothesis
                                    for _x in iter__(x)
                                )
                            ) for x in _risk_hypothesis_add
                        ),
                        *(
                            (
                                any(
                                    _x in rewrite_hypothesis
                                    for _x in iter__(x)
                                )
                            ) for x in _risk_hypothesis_add
                        )
                    )
                )
                for _risk_hypothesis_add in iter__(risk_hypothesis_addition)
            ),
            (
                    any(
                        all(
                            (
                                *(
                                    (
                                        any(
                                            string_check_str(query_hypothesis, _x)
                                            for _x in iter__(x)
                                        )
                                    ) for x in _risk_hypothesis_change_query_cond
                                ),
                                *(
                                    (
                                        any(
                                            (
                                                    (not string_check_str(query_hypothesis, _x)) and
                                                    string_check_str(rewrite_hypothesis, _x)
                                            )
                                            for _x in iter__(x)
                                        )
                                    ) for x in _risk_hypothesis_change_rewrite_cond
                                )
                            )
                        )
                        for _risk_hypothesis_change_query_cond, _risk_hypothesis_change_rewrite_cond
                        in iter__(risk_hypothesis_change)
                    ) and
                    (
                            (not allows_rewrite_contains_query) or
                            (query_utterance not in rewrite_utterance)
                    )
            )
        )
    )


def has_risk_hypothesis_change_udf(
        query_utterance_colname,
        rewrite_utterance_colname,
        query_nlu_hypothesis_colname,
        rewrite_nlu_hypothesis_colname,
        risk_hypothesis_drop=RISK_HYPOTHESIS_DROP,
        risk_hypothesis_addition=RISK_HYPOTHESIS_ADDITION,
        risk_hypothesis_change=RISK_HYPOTHESIS_CHANGE,
        allows_rewrite_contains_query=True
):
    return (col_(query_nlu_hypothesis_colname) != col_(rewrite_nlu_hypothesis_colname)) & or_(
        (
            and_(
                or_(
                    and_(
                        *(
                            (
                                or_(

                                    col_(query_nlu_hypothesis_colname).contains(_x)
                                    for _x in iter__(x)

                                )
                            ) for x in _risk_hypothesis_drop
                        ),
                        *(
                            (
                                ~or_(

                                    col_(rewrite_nlu_hypothesis_colname).contains(_x)
                                    for _x in iter__(x)

                                )
                            ) for x in _risk_hypothesis_drop
                        )
                    )
                    for _risk_hypothesis_drop in iter__(risk_hypothesis_drop)
                ),
                (
                    (~col_(rewrite_utterance_colname).contains(col_(query_utterance_colname)))
                ) if allows_rewrite_contains_query else None
            )
        ),
        or_(
            and_(
                *(
                    (
                        ~or_(

                            col_(query_nlu_hypothesis_colname).contains(_x)
                            for _x in iter__(x)

                        )
                    ) for x in _risk_hypothesis_add
                ),
                *(
                    (
                        or_(

                            col_(rewrite_nlu_hypothesis_colname).contains(_x)
                            for _x in iter__(x)

                        )
                    ) for x in _risk_hypothesis_add
                )
            )
            for _risk_hypothesis_add in iter__(risk_hypothesis_addition)

        ),
        (
            and_(
                or_(
                    and_(
                        *(
                            (
                                or_(
                                    string_check(query_nlu_hypothesis_colname, _x)
                                    for _x in iter__(x)
                                )
                            ) for x in _risk_hypothesis_change_query_cond
                        ),
                        *(
                            (
                                or_(
                                    (
                                            (~string_check(query_nlu_hypothesis_colname, _x)) &
                                            string_check(rewrite_nlu_hypothesis_colname, _x)
                                    )
                                    for _x in iter__(x)
                                )
                            ) for x in _risk_hypothesis_change_rewrite_cond
                        )
                    )
                    for _risk_hypothesis_change_query_cond, _risk_hypothesis_change_rewrite_cond
                    in iter__(risk_hypothesis_change)
                ),
                (
                    (~col_(rewrite_utterance_colname).contains(col_(query_utterance_colname)))
                ) if allows_rewrite_contains_query else None
            )
        )
    )


def hypothesis_change_filter_waiver_udf(
        query_utterance_colname,
        rewrite_utterance_colname,
        query_nlu_hypothesis_colname,
        rewrite_nlu_hypothesis_colname
):
    return (
            (
                    col_(query_nlu_hypothesis_colname).contains('CallIntent') &
                    col_(rewrite_nlu_hypothesis_colname).contains('NavigateIntent') &
                    col_(rewrite_nlu_hypothesis_colname).contains('Direction')
            ) |
            (
                    col_(query_nlu_hypothesis_colname).contains('QAIntent') &
                    col_(rewrite_nlu_hypothesis_colname).contains('LaunchNativeAppIntent') &
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
                            ) > 0.5
                    )
            )
    )


if __name__ == '__main__':
    assert (
        has_risk_hypothesis_change(
            'put a timer for',
            'put a timer for six ten',
            'Notifications|SetNotificationIntent|OnType:timer',
            'Notifications|SetNotificationIntent|OnType:timer|Time:six ten',
            allows_rewrite_contains_query=False
        )
    )
    assert (
        not has_risk_hypothesis_change(
            'at eight fifteen',
            'set alarm at eight fifteen every workday',
            'Notifications|SetNotificationIntent|Duration:eight fifteen',
            'Notifications|SetNotificationIntent|Date:workday|OnType:alarm|RecurringTrigger:every|Time:eight fifteen'
        )
    )
    assert (
        has_risk_hypothesis_change(
            'play spotify',
            'play spotify everywhere',
            'Music|PlayMusicIntent|ServiceName:spotify',
            'Music|PlayMusicIntent|Device:everywhere|ServiceName:spotify'
        )
    )
    assert (
        has_risk_hypothesis_change(
            'play enemy',
            'open music quiz',
            'Music|PlayMusicIntent|SongName:enemy',
            'GeneralMedia|LaunchNativeAppIntent|AppName:music quiz'
        )
    )
    assert (
        has_risk_hypothesis_change(
            'what should a male wear today',
            'what should i wear today',
            'Closet|GetApparelRecommendationIntent|Date:today|GenderStylePreference:male',
            'Closet|GetApparelRecommendationIntent|Date:today'
        )
    )

    assert (
        has_risk_hypothesis_change(
            'what should i wear to church today',
            'what should i wear today',
            'Closet|GetApparelRecommendationIntent|Date:today|EventName:church',
            'Closet|GetApparelRecommendationIntent|Date:today'
        )
    )

    assert (
        has_risk_hypothesis_change(
            'do you ever think about me',
            "when you don't hear from me do you think about me",
            'Knowledge|QAIntent|Question:do you ever think about me',
            "Help|TopicSpecifiedFeedbackIntent|TopicFeedbackContent:when you don't hear from me do you think about me"
        )
    )

    assert (
        has_risk_hypothesis_change(
            'do you ever think about me',
            "when you don't hear from me do you think about me",
            'Knowledge|QAIntent|Question:do you ever think about me',
            "Help|TopicSpecifiedFeedbackIntent|TopicFeedbackContent:when you don't hear from me do you think about me"
        )
    )

    assert (
        has_risk_hypothesis_change(
            'betsy',
            "not you alexa",
            'Knowledge|QAIntent|Question:betsy',
            "Help|TopicSpecifiedFeedbackIntent|TopicFeedbackContent:not you alexa"
        )
    )
