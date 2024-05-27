from functools import partial
from typing import Iterable, Union, Callable

from pyspark.sql.functions import udf
from pyspark.sql.types import BooleanType

from boba_python_utils.common_utils.iter_helper import iter__
from boba_python_utils.general_utils.nlp_utility.common import Languages
from boba_python_utils.general_utils.nlp_utility.metrics.edit_distance import edit_distance
from boba_python_utils.general_utils.nlp_utility.string_sanitization import StringSanitizationOptions, string_sanitize
from boba_python_utils.production_utils._nlu.hypothesis_parsing import SEP_NLU_HYPOTHESIS_PARTS, get_hypothesis_splits, get_slots_dict_from_hypothesis_splits, get_slots_dict_from_hypothesis
from boba_python_utils.production_utils._nlu.slot_value_change import has_slot_change_or_drop
from boba_python_utils.production_utils._nlu.target_slots import is_for_comparison_slot_type
from boba_python_utils.production_utils.pdfs.filters._configs.slot_change_filter import RISK_SLOT_DROP_WHEN_ALL_OTHER_SLOTS_CARRY_OVER, get_risk_slot_change_configs, get_conflict_slot_change_configs, RISKY_SLOT_ADDITION, RISKY_SLOT_DROP, RiskSlotChangeConfig, DEFAULT_SANITIZATION_CONFIG_FOR_RISK_SLOT_CHANGE


def has_conflict_slot_change(
        query_utterance: str,
        rewrite_utterance: str,
        query_hypothesis: str,
        rewrite_hypothesis: str,
        language: Languages = Languages.English,
        risk_slot_change_configs: Union[Iterable[RiskSlotChangeConfig], Callable] = get_risk_slot_change_configs,
        **kwargs
) -> bool:
    if callable(risk_slot_change_configs):
        risk_slot_change_configs = risk_slot_change_configs(language)

    if not (query_hypothesis and rewrite_hypothesis):
        return False

    has_utterance = query_utterance and rewrite_utterance
    if has_utterance:
        if (
                SEP_NLU_HYPOTHESIS_PARTS not in query_hypothesis and
                SEP_NLU_HYPOTHESIS_PARTS not in rewrite_hypothesis and
                SEP_NLU_HYPOTHESIS_PARTS in query_utterance and
                SEP_NLU_HYPOTHESIS_PARTS in rewrite_utterance
        ):
            raise ValueError(
                f"Likely wrong input argument order; "
                f"got query_utterance '{query_utterance}' ",
                f"candidate_utterance '{rewrite_utterance}', ",
                f"query_hypothesis '{query_hypothesis}', ",
                f"candidate_hypothesis '{rewrite_hypothesis}', ",
            )

    return any(
        (
                (not has_utterance) or
                (_config.utterance_similarity_threshold is None) or
                (
                    not (
                            _config.enable_utterance_similarity_for_selected_slot_types is None or
                            any(
                                x in query_hypothesis
                                for x in _config.enable_utterance_similarity_for_selected_slot_types
                            )
                    )
                ) or
                (
                        edit_distance(
                            str1=query_utterance,
                            str2=rewrite_utterance,
                            consider_sorted_tokens=True,
                            fuzzy=False,
                            return_ratio=True
                        ) <= _config.utterance_similarity_threshold
                )
        ) and
        (
                has_slot_change_or_drop(
                    query_hypothesis,
                    rewrite_hypothesis,
                    query_utterance,
                    rewrite_utterance,
                    slot_types=_config.slot_types,
                    slot_change_options=_config.slot_change_options,
                    sanitization_config=(
                            _config.sanitization_config or
                            DEFAULT_SANITIZATION_CONFIG_FOR_RISK_SLOT_CHANGE
                    ),
                    edit_distance_consider_sorted_tokens=_config.edit_distance_consider_sorted_tokens,
                    edit_distance_consider_same_num_tokens=_config.edit_distance_consider_same_num_tokens,
                    edit_distance_options=_config.edit_distance_options,
                    **kwargs
                ) > 1 - _config.slot_value_similarity_threshold
        )
        for _config in iter__(risk_slot_change_configs, atom_types=(RiskSlotChangeConfig,))
    )


def has_conflict_slot_change_udf(
        query_utterance_colname,
        rewrite_utterance_colname,
        query_nlu_hypothesis_colname,
        rewrite_nlu_hypothesis_colname,
        slot_change_configs=get_risk_slot_change_configs
):
    return udf(
        partial(has_conflict_slot_change, risk_slot_change_configs=slot_change_configs),
        returnType=BooleanType()
    )(
        query_utterance_colname,
        rewrite_utterance_colname,
        query_nlu_hypothesis_colname,
        rewrite_nlu_hypothesis_colname
    )


def _has_slot_addition_or_drop(
        addition,
        utterance: str,
        query_hypothesis: str,
        rewrite_hypothesis: str,
        slots: Iterable[str]
):
    def _slot_filter(slot_type):
        return any(x in slot_type for x in slots)

    slots1, slots2 = (
        get_slots_dict_from_hypothesis(
            query_hypothesis,
            return_empty_dict_if_no_slots=True,
            slot_type_filter=_slot_filter
        ),
        get_slots_dict_from_hypothesis(
            rewrite_hypothesis,
            return_empty_dict_if_no_slots=True,
            slot_type_filter=_slot_filter
        )
    )

    if addition:
        for slot_type, slot_value in slots2.items():
            if (
                    slot_type not in slots1 and
                    query_hypothesis and
                    (slot_value not in query_hypothesis) and
                    (slot_value not in utterance)
            ):
                return True
    else:
        for slot_type, slot_value in slots1.items():
            if (
                    slot_type not in slots2 and
                    rewrite_hypothesis and
                    (slot_value not in rewrite_hypothesis) and
                    (slot_value not in utterance)
            ):
                return True
    return False


def has_slot_addition(
        query_utterance: str,
        query_hypothesis: str,
        rewrite_hypothesis: str,
        slots: Iterable[str] = RISKY_SLOT_ADDITION
):
    """

    Args:
        query_utterance:
        query_hypothesis:
        rewrite_hypothesis:
        slots:

    Returns:

    Examples:
        >>> has_slot_addition(
        ...    query_utterance = "new timer",
        ...    query_hypothesis = "Knowledge|QAIntent|Question:new timer",
        ...    rewrite_hypothesis = "Notifications|SetNotificationIntent|Duration:eight minutes|OnType:timer",
        ...    slots = RISKY_SLOT_ADDITION
        ... )
        True
        >>> has_slot_addition(
        ...    query_utterance = "eight minutes",
        ...    query_hypothesis = "Knowledge|QAIntent|Question:eight minutes",
        ...    rewrite_hypothesis = "Notifications|SetNotificationIntent|Duration:eight minutes|OnType:timer",
        ...    slots = RISKY_SLOT_ADDITION
        ... )
        False
    """

    return _has_slot_addition_or_drop(
        addition=True,
        utterance=query_utterance,
        query_hypothesis=query_hypothesis,
        rewrite_hypothesis=rewrite_hypothesis,
        slots=slots
    )


def has_slot_addition_udf(
        query_utterance_colname: str,
        query_nlu_hypothesis_colname: str,
        rewrite_nlu_hypothesis_colname: str,
        slots: Iterable[str] = RISKY_SLOT_ADDITION
):
    return udf(
        partial(
            has_slot_addition,
            slots=slots
        ),
        returnType=BooleanType()
    )(
        query_utterance_colname, query_nlu_hypothesis_colname, rewrite_nlu_hypothesis_colname
    )


def has_slot_drop(
        rewrite_utterance: str,
        query_hypothesis: str,
        rewrite_hypothesis: str,
        slots: Iterable[str] = RISKY_SLOT_DROP
):
    """

    Args:
        rewrite_utterance:
        query_hypothesis:
        rewrite_hypothesis:
        slots:

    Returns:

    Examples:
        >>> has_slot_drop(
        ...    rewrite_utterance = "new timer",
        ...    query_hypothesis = "Notifications|SetNotificationIntent|Duration:eight minutes|OnType:timer",
        ...    rewrite_hypothesis = "Knowledge|QAIntent|Question:new timer",
        ...    slots = RISKY_SLOT_DROP
        ... )
        True
        >>> has_slot_drop(
        ...    rewrite_utterance = "eight",
        ...    query_hypothesis = "Notifications|SetNotificationIntent|Duration:eight minutes|OnType:timer",
        ...    rewrite_hypothesis = "Knowledge|QAIntent|Question:eight",
        ...    slots = RISKY_SLOT_DROP
        ... )
        True
        >>> has_slot_drop(
        ...    rewrite_utterance = "eight minutes",
        ...    query_hypothesis = "Notifications|SetNotificationIntent|Duration:eight minutes|OnType:timer",
        ...    rewrite_hypothesis = "Knowledge|QAIntent|Question:eight minutes",
        ...    slots = RISKY_SLOT_DROP
        ... )
        False
    """

    return _has_slot_addition_or_drop(
        addition=False,
        utterance=rewrite_utterance,
        query_hypothesis=query_hypothesis,
        rewrite_hypothesis=rewrite_hypothesis,
        slots=slots
    )


def has_slot_drop_udf(
        rewrite_utterance_colname: str,
        query_nlu_hypothesis_colname: str,
        rewrite_nlu_hypothesis_colname: str,
        slots: Iterable[str] = RISKY_SLOT_DROP
):
    return udf(
        partial(
            has_slot_drop,
            slots=slots
        ),
        returnType=BooleanType()
    )(
        rewrite_utterance_colname, query_nlu_hypothesis_colname, rewrite_nlu_hypothesis_colname
    )


def has_all_target_slot_dropped(
        rewrite_utterance: str,
        query_hypothesis: str,
        rewrite_hypothesis: str
):
    if (not query_hypothesis) or (not rewrite_hypothesis):
        return None

    hyp_splits1, hyp_splits2 = (
        get_hypothesis_splits(query_hypothesis),
        get_hypothesis_splits(rewrite_hypothesis)
    )

    if len(hyp_splits1) <= 2:
        return False

    slots1, slots2 = (
        get_slots_dict_from_hypothesis_splits(hyp_splits1, return_empty_dict_if_no_slots=True),
        get_slots_dict_from_hypothesis_splits(hyp_splits2, return_empty_dict_if_no_slots=True)
    )

    target_slot_types1 = set(filter(is_for_comparison_slot_type, slots1))

    if (
            target_slot_types1 and
            (
                    not rewrite_utterance or
                    all(
                        slots1[slot_type] not in rewrite_utterance
                        for slot_type in target_slot_types1
                    )
            )
    ):
        if not slots2:
            return True

        target_slot_types2 = set(filter(is_for_comparison_slot_type, slots2))
        return not target_slot_types2

    return False


def has_one_slot_drop_when_all_other_slots_carry_over(
        rewrite_utterance: str,
        query_hypothesis: str,
        rewrite_hypothesis: str,
        risk_slot_types: Iterable[str] = RISK_SLOT_DROP_WHEN_ALL_OTHER_SLOTS_CARRY_OVER
):
    if (not query_hypothesis) or (not rewrite_hypothesis):
        return None

    hyp_splits1, hyp_splits2 = (
        get_hypothesis_splits(query_hypothesis),
        get_hypothesis_splits(rewrite_hypothesis)
    )

    if len(hyp_splits1) <= 3:
        return False

    slots1, slots2 = (
        get_slots_dict_from_hypothesis_splits(hyp_splits1, return_empty_dict_if_no_slots=True),
        get_slots_dict_from_hypothesis_splits(hyp_splits2, return_empty_dict_if_no_slots=True)
    )

    target_slot_types1 = set(filter(is_for_comparison_slot_type, slots1))
    if len(target_slot_types1) <= 1:
        return False

    target_slot_types2 = set(filter(is_for_comparison_slot_type, slots2))
    if not target_slot_types2.issubset(target_slot_types1):
        return False

    overlap_target_slot_types = (
            target_slot_types1 & target_slot_types2
    )
    if len(target_slot_types1) - len(overlap_target_slot_types) != 1:
        return False

    missing_slot_type = None
    for slot_type in target_slot_types1:
        if (
                slot_type in risk_slot_types and
                slot_type not in overlap_target_slot_types and
                (
                        (not rewrite_utterance) or
                        slots1[slot_type] not in rewrite_utterance
                )
        ):
            missing_slot_type = slot_type
            break

    if not missing_slot_type:
        return False

    slot_values2 = set(slots2.values())
    slot_values2_processed = set(
        string_sanitize(
            slot_value2,
            config=[
                StringSanitizationOptions.REMOVE_ACRONYMS_PERIODS_AND_SPACES,
                StringSanitizationOptions.REMOVE_PUNCTUATIONS_EXCEPT_FOR_HYPHEN
            ]
        )
        for slot_value2 in slot_values2
        if slot_value2
    )

    for slot_type in target_slot_types1:
        if slot_type == missing_slot_type:
            continue
        slot_value1 = slots1[slot_type]
        if (
                slot_value1 and
                slot_value1 not in rewrite_utterance and
                slot_value1 not in slot_values2
        ):
            slot_value1 = string_sanitize(
                slot_value1,
                config=[
                    StringSanitizationOptions.REMOVE_ACRONYMS_PERIODS_AND_SPACES,
                    StringSanitizationOptions.REMOVE_PUNCTUATIONS_EXCEPT_FOR_HYPHEN
                ]
            )
            if slot_value1 not in slot_values2_processed:
                return False
    return True


def has_one_slot_drop_when_all_other_slots_carry_over_udf(
        rewrite_utterance_colname: str,
        query_nlu_hypothesis_colname: str,
        rewrite_nlu_hypothesis_colname: str,
        risk_slot_types: Iterable[str] = RISK_SLOT_DROP_WHEN_ALL_OTHER_SLOTS_CARRY_OVER
):
    return udf(
        partial(
            has_one_slot_drop_when_all_other_slots_carry_over,
            risk_slot_types=risk_slot_types
        ),
        returnType=BooleanType()
    )(
        rewrite_utterance_colname,
        query_nlu_hypothesis_colname,
        rewrite_nlu_hypothesis_colname
    )


def has_all_target_slot_dropped_udf(
        rewrite_utterance_colname: str,
        query_nlu_hypothesis_colname: str,
        rewrite_nlu_hypothesis_colname
):
    return udf(has_all_target_slot_dropped, returnType=BooleanType())(
        rewrite_utterance_colname, query_nlu_hypothesis_colname, rewrite_nlu_hypothesis_colname
    )


if __name__ == '__main__':
    assert (
        has_conflict_slot_change(
            query_utterance="turn on j. j.",
            rewrite_utterance="turn on baby shark",
            query_hypothesis="Music|PlayMusicIntent|ArtistName:j. j.",
            rewrite_hypothesis="Music|PlayMusicIntent|SongName:baby shark",
            risk_slot_change_configs=get_conflict_slot_change_configs
        )
    )
    assert (
        has_conflict_slot_change(
            query_utterance="how many countries are located in africa",
            rewrite_utterance="on average how many calories are in spaghetti",
            query_hypothesis="Knowledge|QAIntent|Question:how many countries are located in africa",
            rewrite_hypothesis="Knowledge|QAIntent|Question:on average how many calories are in spaghetti",
            risk_slot_change_configs=get_conflict_slot_change_configs
        )
    )
    assert (
        has_conflict_slot_change(
            query_utterance="play w. k. c. l.",
            rewrite_utterance="play w. q. c. s.",
            query_hypothesis="Music|PlayStationIntent|CallSign:w. k. c. l.",
            rewrite_hypothesis="Music|PlayStationIntent|CallSign:w. q. c. s.",
            risk_slot_change_configs=get_conflict_slot_change_configs
        )
    )
    assert (
        has_conflict_slot_change(
            query_utterance="play w. k. c. l.",
            rewrite_utterance="play w. q. c. s.",
            query_hypothesis="Music|PlayStationIntent|CallSign:w. k. c. l.",
            rewrite_hypothesis="Music|PlayStationIntent|CallSign:w. q. c. s.",
            risk_slot_change_configs=get_conflict_slot_change_configs
        )
    )
    assert (
        has_conflict_slot_change(
            query_utterance="play w. m. x. j.",
            rewrite_utterance="play one oh one point five",
            query_hypothesis="Music|PlayStationIntent|CallSign:w. m. x. j.",
            rewrite_hypothesis="Music|PlayStationIntent|StationNumber:one oh one point five",
            risk_slot_change_configs=get_conflict_slot_change_configs
        )
    )
    assert (
        has_conflict_slot_change(
            query_utterance="stream t. n. t. on hulu",
            rewrite_utterance="stream n. b. c. on hulu",
            query_hypothesis="Video|PlayVideoIntent|ChannelName:t. n. t.|MediaPlayer:hulu",
            rewrite_hypothesis="Video|PlayVideoIntent|ChannelName:n. b. c.|MediaPlayer:hulu",
            risk_slot_change_configs=get_conflict_slot_change_configs
        )
    )
    assert (
        has_conflict_slot_change(
            query_utterance="play rose camp station",
            rewrite_utterance="play future station",
            query_hypothesis="Music|PlayStationIntent|MediaType:station|StationName:rose camp",
            rewrite_hypothesis="Music|PlayStationIntent|ArtistName:future|MediaType:station",
            risk_slot_change_configs=get_conflict_slot_change_configs
        )
    )
    assert (
        has_conflict_slot_change(
            query_utterance="play k. f. x. r.",
            rewrite_utterance="play k. e. g. l.",
            query_hypothesis="Music|PlayStationIntent|CallSign:k. f. x. r.",
            rewrite_hypothesis="Music|PlayStationIntent|CallSign:k. e. g. l.",
            risk_slot_change_configs=get_conflict_slot_change_configs
        )
    )
    assert (
        has_conflict_slot_change(
            query_utterance="play zayde wolf",
            rewrite_utterance="play wonderland",
            query_hypothesis="Music|PlayMusicIntent|ArtistName:zayde wolf",
            rewrite_hypothesis="Music|PlayMusicIntent|SongName:wonderland",
            risk_slot_change_configs=get_risk_slot_change_configs
        )
    )
    assert (
        has_conflict_slot_change(
            query_utterance="play dance monkey",
            rewrite_utterance="play cotton eye joe",
            query_hypothesis="Music|PlayMusicIntent|SongName:dance monkey",
            rewrite_hypothesis="Music|PlayMusicIntent|SongName:cotton eye joe",
            risk_slot_change_configs=get_risk_slot_change_configs
        )
    )
    assert (
        has_conflict_slot_change(
            query_utterance="simon says frog",
            rewrite_utterance="simon says an old rotten frog",
            query_hypothesis="Global|EchoIntent|EchoText:frog",
            rewrite_hypothesis="Global|EchoIntent|EchoText:an old rotten frog",
            risk_slot_change_configs=get_risk_slot_change_configs
        )
    )
    assert (
        has_conflict_slot_change(
            query_utterance="play songs by yes yes no maybe",
            rewrite_utterance="play watch me",
            query_hypothesis="Music|PlayMusicIntent|ArtistName:yes yes no maybe|MediaType:songs",
            rewrite_hypothesis="Music|PlayMusicIntent|SongName:watch me",
            risk_slot_change_configs=get_risk_slot_change_configs
        )
    )
    assert (
        has_conflict_slot_change(
            query_utterance="play radio bright",
            rewrite_utterance="play radio sussex",
            query_hypothesis="Music|PlayStationIntent|StationName:radio bright",
            rewrite_hypothesis="Music|PlayStationIntent|StationName:radio sussex",
            risk_slot_change_configs=get_risk_slot_change_configs
        )
    )
    assert (
        has_conflict_slot_change(
            query_utterance="play the last of akram",
            rewrite_utterance="play raglan road",
            query_hypothesis="amzn1.ask.skill.a70e1c23-90f1-4333-9130-010f4d11fe9a_Live|Play|Query:the last of akram",
            rewrite_hypothesis="Music|PlayMusicIntent|SongName:raglan road",
            risk_slot_change_configs=get_risk_slot_change_configs
        )
    )
    assert (
        not has_conflict_slot_change(
            query_utterance="play pop county boogie by fat",
            rewrite_utterance='play pike county boogie by the fade',
            query_hypothesis="Music|PlayMusicIntent|ArtistName:fat|SongName:pop county boogie",
            rewrite_hypothesis="Music|PlayMusicIntent|ArtistName:the fade|SongName:pike county boogie",
            risk_slot_change_configs=get_conflict_slot_change_configs
        )
    )
    assert (
        not has_conflict_slot_change(
            query_utterance="begin the rosary",
            rewrite_utterance="play the holy rosary",
            query_hypothesis="Books|ReadBookIntent|BookName:the rosary",
            rewrite_hypothesis="GeneralMedia|LaunchNativeAppIntent|AppName:the holy rosary",
            risk_slot_change_configs=get_risk_slot_change_configs
        )
    )
    assert (
        not has_conflict_slot_change(
            query_utterance="read neonism",
            rewrite_utterance="read neon nihilism",
            query_hypothesis="Books|ReadBookIntent|BookName:neonism",
            rewrite_hypothesis="Books|ReadBookIntent|BookName:neon nihilism",
            risk_slot_change_configs=get_risk_slot_change_configs
        )
    )
    assert (
        not has_conflict_slot_change(
            query_utterance="play war by chewiecatt",
            rewrite_utterance="play why by chewiecatt",
            query_hypothesis="Music|PlayMusicIntent|ArtistName:chewiecatt|SongName:war",
            rewrite_hypothesis="Music|PlayMusicIntent|ArtistName:chewiecatt|SongName:why",
            risk_slot_change_configs=get_risk_slot_change_configs
        )
    )
    assert (
        not has_conflict_slot_change(
            query_utterance="play the jesus a little children",
            rewrite_utterance="play jesus loves the little children",
            query_hypothesis="Books|ReadBookIntent|BookName:the jesus a little children",
            rewrite_hypothesis="Music|PlayMusicIntent|SongName:jesus loves the little children",
            risk_slot_change_configs=get_risk_slot_change_configs
        )
    )
    assert (
        not has_conflict_slot_change(
            query_utterance="play audiobook guy jam",
            rewrite_utterance="play audiobook gai jin",
            query_hypothesis="Books|ReadBookIntent|BookName:guy|DeviceBrand:jam|MediaType:audiobook",
            rewrite_hypothesis="Books|ReadBookIntent|BookName:gai jin|MediaType:audiobook",
            risk_slot_change_configs=get_risk_slot_change_configs
        )
    )
    assert (
        not has_conflict_slot_change(
            query_utterance="read madame mayo",
            rewrite_utterance="read mattimeo",
            query_hypothesis="Books|ReadBookIntent|BookName:madame mayo",
            rewrite_hypothesis="Books|ReadBookIntent|BookName:mattimeo",
            risk_slot_change_configs=get_risk_slot_change_configs
        )
    )
    assert (
        not has_conflict_slot_change(
            query_utterance="read i am the term",
            rewrite_utterance="read i am determined",
            query_hypothesis="Books|ReadBookIntent|BookName:i am the term",
            rewrite_hypothesis="Books|ReadBookIntent|BookName:i am determined",
            risk_slot_change_configs=get_risk_slot_change_configs
        )
    )
    assert (
        not has_conflict_slot_change(
            query_utterance="read the arnold labelle collection",
            rewrite_utterance="read the arnold lobel collection",
            query_hypothesis="Books|ReadBookIntent|BookName:arnold labelle collection",
            rewrite_hypothesis="Books|ReadBookIntent|BookName:arnold lobel collection",
            risk_slot_change_configs=get_risk_slot_change_configs
        )
    )
    assert (
        not has_conflict_slot_change(
            query_utterance="play baby be my last song on repeat",
            rewrite_utterance='play baby be my love song on repeat',
            query_hypothesis="Music|PlayMusicIntent|MediaType:song|RepeatTrigger:repeat|SongName:baby be my last",
            rewrite_hypothesis="Music|PlayMusicIntent|RepeatTrigger:repeat|SongName:baby be my love song",
            risk_slot_change_configs=get_risk_slot_change_configs
        )
    )
    assert (
        has_conflict_slot_change(
            query_utterance="play windfall podcast",
            rewrite_utterance="play civilized podcast",
            query_hypothesis="Music|PlayStationIntent|MediaType:podcast|ProgramName:windfall",
            rewrite_hypothesis="Music|PlayStationIntent|MediaType:podcast|ProgramName:civilized",
            risk_slot_change_configs=get_risk_slot_change_configs
        )
    )
    assert (
        has_conflict_slot_change(
            query_utterance="play metal rock",
            rewrite_utterance="play magic",
            query_hypothesis="Music|PlayMusicIntent|GenreName:metal rock",
            rewrite_hypothesis="Music|PlayMusicIntent|SongName:magic",
            risk_slot_change_configs=get_risk_slot_change_configs
        )
    )
    assert (
        has_conflict_slot_change(
            query_utterance="play tar by chewiecatt",
            rewrite_utterance="play why by chewiecatt",
            query_hypothesis="Music|PlayMusicIntent|ArtistName:chewiecatt|SongName:tar",
            rewrite_hypothesis="Music|PlayMusicIntent|ArtistName:chewiecatt|SongName:why",
            risk_slot_change_configs=get_risk_slot_change_configs
        )
    )
    assert (
        has_conflict_slot_change(
            query_utterance="play sydney devine's music",
            rewrite_utterance="play patsy cline's music",
            query_hypothesis="Music|PlayMusicIntent|ArtistName:sydney devine's|MediaType:music",
            rewrite_hypothesis="Music|PlayMusicIntent|ArtistName:patsy cline's|MediaType:music",
            risk_slot_change_configs=get_risk_slot_change_configs
        )
    )
    assert (
        has_conflict_slot_change(
            query_utterance="play songs by santa",
            rewrite_utterance="play shotgun",
            query_hypothesis="Music|PlayMusicIntent|ArtistName:santa|MediaType:songs",
            rewrite_hypothesis="Music|PlayMusicIntent|SongName:shotgun",
            risk_slot_change_configs=get_risk_slot_change_configs
        )
    )
    assert (
        has_conflict_slot_change(
            query_utterance="play songs by marky b.",
            rewrite_utterance="play songs by bru c.",
            query_hypothesis="Music|PlayMusicIntent|ArtistName:marky b.|MediaType:songs",
            rewrite_hypothesis="Music|PlayMusicIntent|ArtistName:bru c.|MediaType:songs",
            risk_slot_change_configs=get_risk_slot_change_configs
        )
    )
    assert (
        has_conflict_slot_change(
            query_utterance="play ocean pie by shed seven",
            rewrite_utterance="play heroes by shed seven",
            query_hypothesis="Music|PlayMusicIntent|ArtistName:shed seven|SongName:ocean pie",
            rewrite_hypothesis="Music|PlayMusicIntent|ArtistName:shed seven|SongName:heroes",
            risk_slot_change_configs=get_risk_slot_change_configs
        )
    )
    assert (
        has_conflict_slot_change(
            query_utterance="play freaky deaky",
            rewrite_utterance="play i really like you",
            query_hypothesis="Music|PlayMusicIntent|SongName:freaky deaky",
            rewrite_hypothesis="Music|PlayMusicIntent|SongName:i really like you",
            risk_slot_change_configs=get_risk_slot_change_configs
        )
    )
    assert (
        has_conflict_slot_change(
            query_utterance="play the world's worst parents on audible",
            rewrite_utterance="play world's worst pets from audible",
            query_hypothesis="Books|ReadBookIntent|BookName:the world's worst parents|ServiceName:audible",
            rewrite_hypothesis="Books|ReadBookIntent|BookName:world's worst pets|ServiceName:audible",
            risk_slot_change_configs=get_risk_slot_change_configs
        )
    )
    assert (
        has_conflict_slot_change(
            query_utterance="play kay's marvelous medicine on audible",
            rewrite_utterance="play k. s. mars medicine on audible",
            query_hypothesis="Books|ReadBookIntent|BookName:kay's marvelous medicine|ServiceName:audible",
            rewrite_hypothesis="Books|ReadBookIntent|BookName:k. s. mars medicine|ServiceName:audible",
            risk_slot_change_configs=get_risk_slot_change_configs
        )
    )
    assert (
        has_conflict_slot_change(
            query_utterance="read hilda and the white wolf",
            rewrite_utterance="read hilda and the time worm",
            query_hypothesis="Books|ReadBookIntent|BookName:hilda and the white wolf",
            rewrite_hypothesis="Books|ReadBookIntent|BookName:hilda and the time worm",
            risk_slot_change_configs=get_risk_slot_change_configs
        )
    )
    assert (
        has_conflict_slot_change(
            query_utterance="how do i make potatoes",
            rewrite_utterance="how do i make pork chops",
            query_hypothesis="Recipes|SearchRecipeIntent|DishName:potatoes|Interrogative:how",
            rewrite_hypothesis="Recipes|SearchRecipeIntent|DishName:pork chops|Interrogative:how",
            risk_slot_change_configs=get_risk_slot_change_configs
        )
    )
    assert (
        has_conflict_slot_change(
            query_utterance="how me my outlook calendar",
            rewrite_utterance="show me my microsoft calendar",
            query_hypothesis="Calendar|BrowseCalendarIntent|AccountProviderName:outlook|ActiveUserTrigger:my|DataSource:calendar|VisualModeTrigger:show",
            rewrite_hypothesis="Calendar|BrowseCalendarIntent|AccountProviderName:microsoft|ActiveUserTrigger:my|DataSource:calendar|VisualModeTrigger:show",
            risk_slot_change_configs=get_risk_slot_change_configs
        )
    )
    assert (
        has_conflict_slot_change(
            query_utterance="when's my next appointment with doctor mills",
            rewrite_utterance="when's my next appointment with doctor anderson",
            query_hypothesis="Calendar|BrowseCalendarIntent|ActiveUserTrigger:my|EventDetail:when's|EventType:appointment|Participant:doctor mills",
            rewrite_hypothesis="Calendar|BrowseCalendarIntent|ActiveUserTrigger:my|EventDetail:when's|EventType:appointment|Participant:doctor anderson",
            risk_slot_change_configs=get_risk_slot_change_configs
        )
    )
    assert (
        has_conflict_slot_change(
            query_utterance="play lion king stories",
            rewrite_utterance='open disney stories',
            query_hypothesis="Books|ReadBookIntent|BookName:lion king|MediaType:stories",
            rewrite_hypothesis="GeneralMedia|LaunchNativeAppIntent|AppName:disney stories",
            risk_slot_change_configs=get_risk_slot_change_configs
        )
    )
    assert (
        has_conflict_slot_change(
            query_utterance="play k. o. s. u. n. p. r.",
            rewrite_utterance='play n. p. r.',
            query_hypothesis="Music|PlayStationIntent|CallSign:k. o. s. u. n. p. r.",
            rewrite_hypothesis="GeneralMedia|LaunchNativeAppIntent|AppName:n. p. r.",
            risk_slot_change_configs=get_risk_slot_change_configs
        )
    )
    assert (
        has_conflict_slot_change(
            query_utterance="play k. o. s. u. n. p. r.",
            rewrite_utterance='play n. p. r.',
            query_hypothesis="Music|PlayStationIntent|CallSign:k. o. s. u. n. p. r.",
            rewrite_hypothesis="GeneralMedia|LaunchNativeAppIntent|AppName:n. p. r.",
            risk_slot_change_configs=get_risk_slot_change_configs
        )
    )
    assert (
        has_conflict_slot_change(
            query_utterance="play c. p. r.",
            rewrite_utterance='play n. p. r.',
            query_hypothesis="Music|PlayStationIntent|StationName:c. p. r.",
            rewrite_hypothesis="GeneralMedia|LaunchNativeAppIntent|AppName:n. p. r.",
            risk_slot_change_configs=get_risk_slot_change_configs
        )
    )
    assert (
        has_conflict_slot_change(
            query_utterance="play w. b. r. w.",
            rewrite_utterance='play w. r. o. w.',
            query_hypothesis="Music|PlayStationIntent|CallSign:w. b. r. w.",
            rewrite_hypothesis="Music|PlayStationIntent|CallSign:w. r. o. w.",
            risk_slot_change_configs=get_risk_slot_change_configs
        )
    )

    assert (
        has_conflict_slot_change(
            query_utterance="open devotional",
            rewrite_utterance='open verse a day',
            query_hypothesis="GeneralMedia|LaunchNativeAppIntent|AppName:devotional",
            rewrite_hypothesis="GeneralMedia|LaunchNativeAppIntent|AppName:verse a day",
            risk_slot_change_configs=get_risk_slot_change_configs
        )
    )
    assert (
        has_conflict_slot_change(
            query_utterance='play legendary by finger death punch"',
            rewrite_utterance='play no more dream',
            query_hypothesis='Music|PlayMusicIntent|SongName:legendary|ArtistName:finger death punch',
            rewrite_hypothesis='Music|PlayMusicIntent|SongName:no more dream',
            risk_slot_change_configs=get_risk_slot_change_configs
        )
    )

    assert (
        has_conflict_slot_change(
            query_utterance='play ice cream truck songs on amazon music',
            rewrite_utterance='play ice cream trucks on youtube',
            query_hypothesis='Music|PlayMusicIntent|AppName:amazon music|MediaType:songs|SongName:ice cream truck',
            rewrite_hypothesis='Video|PlayVideoIntent|AppName:youtube|VideoName:ice cream trucks',
            risk_slot_change_configs=get_risk_slot_change_configs
        )
    )
    assert (
        has_conflict_slot_change(
            query_utterance='shuffle music by peter hollen',
            rewrite_utterance='shuffle music by liquid mind',
            query_hypothesis='Music|PlayMusicIntent|ArtistName:peter hollen|MediaType:music|ShuffleTrigger:shuffle',
            rewrite_hypothesis='Music|PlayMusicIntent|ArtistName:liquid mind|MediaType:music|ShuffleTrigger:shuffle',
            risk_slot_change_configs=get_risk_slot_change_configs
        )
    )

    assert (
        has_one_slot_drop_when_all_other_slots_carry_over(
            "please play the song by marc bolan",
            "Music|PlayMusicIntent|ArtistName:marc bolan|MediaType:song|SongName:get it on",
            "Music|PlayMusicIntent|ArtistName:marc bolan|MediaType:song"
        )
    )

    assert (
        has_one_slot_drop_when_all_other_slots_carry_over(
            "play songs by b. t. s.",
            "Music|PlayMusicIntent|ArtistName:b. t. s.|SongName:butter",
            "Music|PlayMusicIntent|ArtistName:b. t. s.|MediaType:songs"
        )
    )

    assert (
        has_one_slot_drop_when_all_other_slots_carry_over(
            "play spotify",
            "Music|PlayMusicIntent|GenreName:r. and b.|ServiceName:spotify",
            "Music|PlayMusicIntent|ServiceName:spotify"
        )
    )
    assert (
        has_one_slot_drop_when_all_other_slots_carry_over(
            "play spotify",
            "Music|PlayMusicIntent|GenreName:r. and b.|ServiceName:spotify",
            "Music|PlayMusicIntent|ServiceName:spotify"
        )
    )
    assert (
        has_one_slot_drop_when_all_other_slots_carry_over(
            "play spotify",
            "Music|PlayMusicIntent|GenreName:r. and b.|ServiceName:spotify",
            "Music|PlayMusicIntent|ServiceName:spotify"
        )
    )

    assert (
        has_one_slot_drop_when_all_other_slots_carry_over(
            "play play r. and b. songs from the nineties",
            "Music|PlayMusicIntent|GenreName:nineties|SongName:are and b.",
            "Music|PlayMusicIntent|GenreName:r. and b. nineties|MediaType:songs"
        )
    )
    assert (
        has_one_slot_drop_when_all_other_slots_carry_over(
            "play we don't talk about bruno",
            "Music|PlayMusicIntent|ArtistName:bruno mars|SongName:don't we don't talk about",
            "Music|PlayMusicIntent|SongName:don't we don't talk about"
        )
    )

    assert has_all_target_slot_dropped(
        "play my favorite songs",
        "Music|PlayMusicIntent|ArtistName:o. t. w.|SongName:o. t. w.",
        "Music|PlayMusicIntent|ActiveUserTrigger:my|MediaType:songs|SortType:favorite"
    )

    assert not has_all_target_slot_dropped(
        "play spotify",
        "Music|PlayMusicIntent|GenreName:r. and b.|ServiceName:spotify",
        "Music|PlayMusicIntent|ServiceName:spotify"
    )
