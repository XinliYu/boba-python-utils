from functools import partial
from typing import Union, Callable, Iterable, Optional

from pyspark.sql import Column
from pyspark.sql.functions import udf
from pyspark.sql.types import FloatType

from boba_python_utils.general_utils.nlp_utility.common import Languages
from boba_python_utils.general_utils.nlp_utility.metrics.edit_distance import edit_distance, EditDistanceOptions
from boba_python_utils.general_utils.nlp_utility.string_sanitization import StringSanitizationOptions, remove_common_tokens_except_for_sub_tokens
from boba_python_utils.production_utils._nlu.target_slots import normalize_utterance_text_by_slot_values, is_non_stop_slot_type

def edit_distance_ratio_with_slot_values(
        query_utterance,
        rewrite_utterance,
        query_nlu_hypothesis,
        rewrite_nlu_hypothesis,
        language: Languages = Languages.English,
        slot_type_filter: Optional[Union[Callable, Iterable]] = is_non_stop_slot_type,
        ignore_overlap: bool = False,
        reduction=max
):
    """

    Examples:
        >>> from boba_python_utils.product_utils.nlu import is_for_comparison_slot_type
        >>> edit_distance_ratio_with_slot_values(
        ...   "turn on j. j.",
        ...   "turn on baby shark",
        ...   "Music|PlayMusicIntent|ArtistName:j. j.",
        ...   "Music|PlayMusicIntent|SongName:baby shark"
        ... )
        0.5
        >>> edit_distance_ratio_with_slot_values(
        ...   "play party",
        ...   "play party in the u. s. a.",
        ...   "Music|PlayMusicIntent|SongName:party",
        ...   "Music|PlayMusicIntent|SongName:party in the u. s. a.",
        ...   slot_type_filter=is_for_comparison_slot_type,
        ...   ignore_overlap=True
        ... )
        0.46571428571428564
        >>> edit_distance_ratio_with_slot_values(
        ...   "play some gospel christian music",
        ...   "play christian gospel music",
        ...   "Music|PlayMusicIntent|GenreName:gospel christian|MediaType:music|Quantifier:some",
        ...   "Music|PlayMusicIntent|GenreName:christian gospel|MediaType:music",
        ...   slot_type_filter=is_for_comparison_slot_type,
        ...   ignore_overlap=True
        ... )
        1.0
        >>> edit_distance_ratio_with_slot_values(
        ...   "shuffle my playlist easy",
        ...   "shuffle my playlist izzie",
        ...   "Music|PlayMusicIntent|ActiveUserTrigger:my|MediaType:playlist|PlaylistName:easy|ShuffleTrigger:shuffle",
        ...   "Music|PlayMusicIntent|ActiveUserTrigger:my|MediaType:playlist|PlaylistName:izzie|ShuffleTrigger:shuffle",
        ...   slot_type_filter=is_for_comparison_slot_type,
        ...   ignore_overlap=True
        ... )
        0.28
        >>> edit_distance_ratio_with_slot_values(
        ...   "turn on the garbage",
        ...   "turn on the goblet",
        ...   "HomeAutomation|TurnOnApplianceIntent|ActionTrigger:turn on|DeviceName:garbage",
        ...   "HomeAutomation|TurnOnApplianceIntent|ActionTrigger:turn on|DeviceName:goblet",
        ...   slot_type_filter=is_for_comparison_slot_type,
        ...   ignore_overlap=True
        ... )
        0.33333333333333337
        >>> edit_distance_ratio_with_slot_values(
        ...   "play sneeze",
        ...   "play s. m. e. e. z. e.",
        ...   "Music|PlayMusicIntent|SongName:sneeze",
        ...   "Music|PlayMusicIntent|SongName:s. m. e. e. z. e.",
        ...   slot_type_filter=is_for_comparison_slot_type,
        ...   ignore_overlap=True
        ... )
        1.0
        >>> from boba_python_utils.product_utils.nlu import is_for_comparison_slot_type
        >>> edit_distance_ratio_with_slot_values(
        ...   "how many minutes left in the alarm",
        ...   "how many minutes left on the timer",
        ...   "Notifications|BrowseNotificationIntent|OnType:alarm",
        ...   "Notifications|BrowseNotificationIntent|OnType:timer",
        ...   slot_type_filter=is_for_comparison_slot_type,
        ...   ignore_overlap=True
        ... )
        1.0
        >>> edit_distance_ratio_with_slot_values(
        ...   "drop in echo k.",
        ...   "drop in echo keya",
        ...   "Communication|InstantConnectIntent|CallType:drop in|Device:k.|DeviceBrand:echo",
        ...   "Communication|InstantConnectIntent|CallType:drop in|Device:echo keya",
        ...   slot_type_filter=None,
        ...   ignore_overlap=True
        ... )
        1.0
        >>> edit_distance_ratio_with_slot_values(
        ...   "drop in on jim dot",
        ...   "drop in on gym dot",
        ...   "Communication|InstantConnectIntent|CallType:drop in|ContactName:jim|PhoneNumberType:dot",
        ...   "Communication|InstantConnectIntent|CallType:drop in|Device:gym dot",
        ...   slot_type_filter=None,
        ...   ignore_overlap=True
        ... )
        0.6666666666666667
        >>> edit_distance_ratio_with_slot_values(
        ...   "drop in on ellary's room",
        ...   "drop in on elloree's room",
        ...   "Communication|InstantConnectIntent|CallType:drop in|DeviceLocation:ellary's room",
        ...   "Communication|InstantConnectIntent|CallType:drop in|ContactName:elloree's|DeviceLocation:room",
        ...   slot_type_filter=None,
        ...   ignore_overlap=True
        ... )
        0.8
        >>> edit_distance_ratio_with_slot_values(
        ...   "cancel alarm in gemma's room",
        ...   "cancel alarm in chase's room",
        ...   "Notifications|CancelNotificationIntent|DeviceLocation:room|DeviceName:gemma's|OnType:alarm",
        ...   "Notifications|CancelNotificationIntent|DeviceLocation:room|DeviceName:chase's|OnType:alarm",
        ...   slot_type_filter=None,
        ...   ignore_overlap=True
        ... )
        0.20000000000000007
        >>> edit_distance_ratio_with_slot_values(
        ...   "listen to the big d. m. on iheartradio",
        ...   "go to the big d. m. on iheartradio",
        ...   "Music|PlayStationIntent|AppName:iheartradio|StationName:big d. m.",
        ...   "Music|PlayStationIntent|AppName:iheartradio|StationName:big d. m.",
        ...   slot_type_filter=None,
        ...   ignore_overlap=True
        ... )
        1.0
        >>> edit_distance_ratio_with_slot_values(
        ...   "the post",
        ...   "play the pulse on siriusxm",
        ...   "amzn1.ask.skill.cb01deae-b670-400f-be87-20b868275e18_Live|PlayAudio",
        ...   "Music|PlayStationIntent|ServiceName:siriusxm|StationName:the pulse"
        ... )
        0.5714285714285714
        >>> edit_distance_ratio_with_slot_values(
        ...   "pen kids' song",
        ...   "open baby song",
        ...   "Music|PlayMusicIntent|GenreName:kids'|MediaType:song",
        ...   "GeneralMedia|LaunchNativeAppIntent|AppName:baby song"
        ... )
        0.6153846153846154
    """
    (
        non_stop_slot_query_utterance,
        non_stop_slot_rewrite_utterance
    ) = normalize_utterance_text_by_slot_values(
        query_utterance,
        rewrite_utterance,
        query_nlu_hypothesis,
        rewrite_nlu_hypothesis,
        slot_type_filter=slot_type_filter,
        remove_overlap=ignore_overlap
    )

    if not non_stop_slot_query_utterance or not non_stop_slot_rewrite_utterance:
        if ignore_overlap:
            return 1.0
        else:
            non_stop_slot_query_utterance = query_utterance
            non_stop_slot_rewrite_utterance = rewrite_utterance
    elif ignore_overlap:
        (
            query_utterance, rewrite_utterance
        ) = remove_common_tokens_except_for_sub_tokens(query_utterance, rewrite_utterance)

    slot_value_edit_distance_options = EditDistanceOptions(
        weight_distance_if_strs_have_common_start=0.85,
        weight_distance_if_str1_is_substr=0.8,
        weight_distance_for_short_strs=0.6,
        weight_distance_by_comparing_start=language,
        weight_distance_by_comparing_end=language,
        min_length_for_distance_for_str1=3,
        min_length_for_distance_for_str2=3
    )

    edit_distance_ratio = edit_distance(
        query_utterance,
        rewrite_utterance,
        consider_same_num_tokens=True,
        consider_sorted_tokens=True,
        sanitization_config=[
            StringSanitizationOptions.REMOVE_ACRONYMS_PERIODS_AND_SPACES,
            StringSanitizationOptions.REMOVE_PUNCTUATIONS_EXCEPT_FOR_HYPHEN,
            StringSanitizationOptions.MAKE_FUZZY
        ],
        options=slot_value_edit_distance_options if ignore_overlap else None,
        return_ratio=True
    )

    if (
            non_stop_slot_query_utterance == query_utterance and
            non_stop_slot_rewrite_utterance == rewrite_utterance
    ):
        return edit_distance_ratio

    return reduction(
        edit_distance(
            non_stop_slot_query_utterance,
            non_stop_slot_rewrite_utterance,
            consider_same_num_tokens=False,
            consider_sorted_tokens=True,
            sanitization_config=[
                StringSanitizationOptions.REMOVE_ACRONYMS_PERIODS_AND_SPACES,
                StringSanitizationOptions.REMOVE_PUNCTUATIONS_EXCEPT_FOR_HYPHEN,
                StringSanitizationOptions.MAKE_FUZZY
            ],
            options=slot_value_edit_distance_options,
            return_ratio=True
        ),
        edit_distance_ratio
    )


def edit_distance_ratio_by_slot_values_udf(
        query_utterance,
        rewrite_utterance,
        query_nlu_hypothesis,
        rewrite_nlu_hypothesis,
        language: Union[str, Languages] = Languages.English,
        slot_type_filter=is_non_stop_slot_type,
        ignore_overlap: bool = False
) -> Column:
    return udf(
        partial(
            edit_distance_ratio_with_slot_values,
            language=language,
            slot_type_filter=slot_type_filter,
            ignore_overlap=ignore_overlap
        ),
        returnType=FloatType()
    )(
        query_utterance,
        rewrite_utterance,
        query_nlu_hypothesis,
        rewrite_nlu_hypothesis
    )
