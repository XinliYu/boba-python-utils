from functools import partial

from pyspark.sql.functions import udf
from pyspark.sql.types import BooleanType

from boba_python_utils.general_utils.nlp_utility.common import Languages
from boba_python_utils.general_utils.nlp_utility.metrics.edit_distance import EditDistanceOptions
from boba_python_utils.general_utils.nlp_utility.numbers import has_conflict_numbers as _has_conflict_numbers
from boba_python_utils.general_utils.nlp_utility.numbers import remove_acronym_periods_and_spaces as _remove_acronym_periods_and_spaces
from boba_python_utils.string_utils.common import contains_any
from boba_python_utils.production_utils._nlu.hypothesis_parsing import get_slots_dict_from_hypothesis
from boba_python_utils.production_utils.pdfs.filters._configs.conflict_numbers import SLOT_SPECIFIC_CONFLICT_NUMBER_SETUP


def has_conflict_numbers(
        utterance1,
        utterance2,
        allows_add: bool = False,
        allows_drop: bool = False,
        edit_distance_threshold=None,
        allows_one_being_substr_of_the_other=True,
        language: Languages = Languages.English
):
    """
    Checkes if two utterances contain conflict numbers,

    Examples:
        >>> assert has_conflict_numbers('set alarm for five minutes', 'set alarm for six minutes')
        >>> assert has_conflict_numbers('hotel transylvania two', 'hotel transylvania 3')
        >>> assert not has_conflict_numbers(None, 'play sixties music')
        >>> assert not has_conflict_numbers('play sixty music', 'play sixties music')
    """
    if not (utterance1 and utterance2):
        return False
    if (
            allows_one_being_substr_of_the_other and
            (
                    utterance1 in utterance2 or
                    utterance2 in utterance1
            )
    ):
        return False
    return _has_conflict_numbers(
        str1=_remove_acronym_periods_and_spaces(utterance1),
        str2=_remove_acronym_periods_and_spaces(utterance2),
        allows_add=allows_add,
        allows_drop=allows_drop,
        edit_distance_threshold_to_ignore_conflict=edit_distance_threshold,
        language=language,
        consider_edit_distance_with_sorted_tokens=True,
        consider_same_num_tokens=True,
        edit_distance_options=EditDistanceOptions(
            weight_distance_if_strs_have_common_start=0.85,
            weight_distance_if_str1_is_substr=0.95,
            weight_distance_if_str2_is_substr=0.95
        )
    )


def has_conflict_numbers_udf(
        utterance1_colname,
        utterance2_colname,
        allows_add: bool = False,
        allows_drop: bool = False,
        edit_distance_threshold=0.65,
        allows_one_being_substr_of_the_other=True,
        language: str = 'en'
):
    return udf(
        partial(
            has_conflict_numbers,
            allows_add=allows_add,
            allows_drop=allows_drop,
            edit_distance_threshold=edit_distance_threshold,
            allows_one_being_substr_of_the_other=allows_one_being_substr_of_the_other,
            language=language
        ),
        returnType=BooleanType()
    )(utterance1_colname, utterance2_colname)


def _solve_slots_for_number_conflict(nlu_hypothesis1, nlu_hypothesis2, target_slot_types):
    slots1 = get_slots_dict_from_hypothesis(nlu_hypothesis1)
    if not slots1:
        return None, None, None
    slots2 = get_slots_dict_from_hypothesis(nlu_hypothesis2)
    if not slots2:
        return None, None, None

    slot_types_to_check1 = set(
        filter(
            partial(contains_any, targets=target_slot_types),
            slots1.keys()
        )
    )
    slot_types_to_check2 = set(
        filter(
            partial(contains_any, targets=target_slot_types),
            slots2.keys()
        )
    )

    if (
            len(slot_types_to_check1) == 1 and
            len(slot_types_to_check2) == 1
    ):
        sole_slot_type1 = next(iter(slot_types_to_check1))
        sole_slot_type2 = next(iter(slot_types_to_check2))
        if 'Number' in sole_slot_type1 and 'Number' in sole_slot_type2:
            slot_types_to_check1 = slot_types_to_check2
            slots1 = {sole_slot_type2: slots1[sole_slot_type1]}

    slot_types_to_check = tuple(
        filter(
            lambda x: x in slots2,
            slot_types_to_check1
        )
    )
    return slots1, slots2, slot_types_to_check


def _slot_has_conflict_numbers(
        nlu_hypothesis1: str,
        nlu_hypothesis2: str,
        slot_conflict_number_config=SLOT_SPECIFIC_CONFLICT_NUMBER_SETUP,
        edit_distance_threshold=None,
        language: Languages = Languages.English
):
    if not nlu_hypothesis1 or not nlu_hypothesis2:
        return False

    slots1, slots2, slot_types_to_check = _solve_slots_for_number_conflict(
        nlu_hypothesis1=nlu_hypothesis1,
        nlu_hypothesis2=nlu_hypothesis2,
        target_slot_types=tuple(slot_conflict_number_config.keys())
    )

    if not slot_types_to_check:
        return False

    return any(
        any(
            has_conflict_numbers(
                slots1[slot_type], slots2[slot_type],
                allows_add=allows_add,
                allows_drop=allows_drop,
                edit_distance_threshold=_edit_distance_threshold or edit_distance_threshold,
                allows_one_being_substr_of_the_other=allows_substr,
                language=language
            )
            for slot_type_keyword, (allows_add, allows_drop, allows_substr, _edit_distance_threshold)
            in slot_conflict_number_config.items()
            if slot_type_keyword in slot_type
        )
        for slot_type in slot_types_to_check
    )


def slot_has_conflict_numbers(
        nlu_hypothesis1: str,
        nlu_hypothesis2: str,
        slot_types=SLOT_SPECIFIC_CONFLICT_NUMBER_SETUP,
        edit_distance_threshold=None,
        language: Languages = Languages.English,
        always_include_en: bool = True
):
    result = _slot_has_conflict_numbers(
        nlu_hypothesis1=nlu_hypothesis1,
        nlu_hypothesis2=nlu_hypothesis2,
        slot_conflict_number_config=slot_types,
        edit_distance_threshold=edit_distance_threshold,
        language=language
    )

    if (
            language == Languages.English or
            not always_include_en
    ):
        return result

    return result or _slot_has_conflict_numbers(
        nlu_hypothesis1=nlu_hypothesis1,
        nlu_hypothesis2=nlu_hypothesis2,
        slot_conflict_number_config=slot_types,
        edit_distance_threshold=edit_distance_threshold,
        language=Languages.English
    )


def slot_has_conflict_numbers_udf(
        nlu_hypothesis1_colname: str,
        nlu_hypothesis2_colname: str,
        slot_types=SLOT_SPECIFIC_CONFLICT_NUMBER_SETUP,
        edit_distance_threshold=None,
        language: Languages = Languages.English,
        always_include_en: bool = True
):
    return udf(
        partial(
            slot_has_conflict_numbers,
            slot_types=slot_types,
            edit_distance_threshold=edit_distance_threshold,
            language=language,
            always_include_en=always_include_en
        ),
        returnType=BooleanType()
    )(nlu_hypothesis1_colname, nlu_hypothesis2_colname)


if __name__ == '__main__':
    edit_distance_threshold = 0.80
    assert slot_has_conflict_numbers(
        nlu_hypothesis1='Video|PlayVideoIntent|ChannelNumber:eight oh four|MediaType:channel',
        nlu_hypothesis2='Video|PlayVideoIntent|ChannelNumber:eight four zero|MediaType:channel',
        edit_distance_threshold=edit_distance_threshold
    )
    assert slot_has_conflict_numbers(
        nlu_hypothesis1='amzn1.echo-sdk-ams.app.40208bde-9a3a-4d7f-8506-9ad676cfc3d3_Live|NumberCounterIntent|NumberOnes:channel three',
        nlu_hypothesis2='Global|NavigateIntent|ChannelNumber:two hundred|MediaType:channel',
        edit_distance_threshold=edit_distance_threshold
    )
    assert slot_has_conflict_numbers(
        nlu_hypothesis1='Global|NavigateIntent|ChannelNumber:eight zero five|MediaType:channel',
        nlu_hypothesis2='Global|NavigateIntent|ChannelNumber:eight one zero|DeviceType:tv|MediaType:channel',
        edit_distance_threshold=edit_distance_threshold
    )
    assert slot_has_conflict_numbers(
        nlu_hypothesis1='Global|NavigateIntent|ChannelNumber:seven|MediaType:channel',
        nlu_hypothesis2='Global|NavigateIntent|ChannelNumber:seventy|MediaType:channel',
        edit_distance_threshold=edit_distance_threshold
    )
    assert slot_has_conflict_numbers(
        nlu_hypothesis1='Music|PlayStationIntent|StationName:hits|StationNumber:ninety six point one',
        nlu_hypothesis2='Music|PlayStationIntent|StationName:kiss|StationNumber:ninety five point one',
        edit_distance_threshold=edit_distance_threshold
    )
    assert slot_has_conflict_numbers(
        nlu_hypothesis1='Music|PlayMusicIntent|AlbumName:high school musical three|MediaType:soundtrack',
        nlu_hypothesis2='Music|PlayMusicIntent|AlbumName:high school musical|MediaType:soundtrack',
        edit_distance_threshold=edit_distance_threshold
    )
    assert slot_has_conflict_numbers(
        nlu_hypothesis1='Global|NavigateIntent|ChannelNumber:three|MediaType:channel',
        nlu_hypothesis2='Global|NavigateIntent|ChannelNumber:two hundred|MediaType:channel',
        edit_distance_threshold=edit_distance_threshold
    )
    assert slot_has_conflict_numbers(
        nlu_hypothesis1='Global|NavigateIntent|ChannelNumber:eight zero five|MediaType:channel',
        nlu_hypothesis2='Global|NavigateIntent|ChannelNumber:eight one zero|DeviceType:tv|MediaType:channel',
        edit_distance_threshold=edit_distance_threshold
    )
    assert slot_has_conflict_numbers(
        nlu_hypothesis1='Music|PlayMusicIntent|AlbumName:zombies three',
        nlu_hypothesis2='Music|PlayMusicIntent|AlbumName:zombies two',
        edit_distance_threshold=edit_distance_threshold
    )
    assert slot_has_conflict_numbers(
        nlu_hypothesis1='Books|NavigateBooksIntent|SectionNumber:one|SectionType:chapter',
        nlu_hypothesis2='Books|NavigateBooksIntent|SectionNumber:three|SectionType:chapter',
        edit_distance_threshold=edit_distance_threshold
    )
    assert slot_has_conflict_numbers(
        nlu_hypothesis1='HomeAutomation|SetValueIntent|DeviceType:light|SettingValue:hundred|ValueType:percent',
        nlu_hypothesis2='HomeAutomation|SetValueIntent|ActionTrigger:turn|DeviceName:living room|DeviceType:lamp|SettingValue:eighty|ValueType:percent',
        edit_distance_threshold=edit_distance_threshold
    )
    assert slot_has_conflict_numbers(
        nlu_hypothesis1='HomeAutomation|SetValueIntent|ActionTrigger:adjust|DeviceType:lights|SettingValue:fifty|ValueType:percent',
        nlu_hypothesis2='HomeAutomation|SetValueIntent|ActionTrigger:turn|DeviceName:tv|DeviceType:light strip|SettingValue:blue',
        edit_distance_threshold=edit_distance_threshold
    )
    assert slot_has_conflict_numbers(
        nlu_hypothesis1='HomeAutomation|ApplianceSettingUpIntent|ActionTrigger:raise|DeviceType:heat|SettingValue:sixty sixty seven',
        nlu_hypothesis2='HomeAutomation|ApplianceSettingUpIntent|ActionTrigger:raise|DeviceType:heat|SettingValue:one|ValueType:degree',
        edit_distance_threshold=edit_distance_threshold
    )
    assert slot_has_conflict_numbers(
        nlu_hypothesis1='HomeAutomation|ApplianceSettingUpIntent|ActionTrigger:raise|DeviceType:heat|SettingValue:sixty sixty seven',
        nlu_hypothesis2='HomeAutomation|ApplianceSettingUpIntent|ActionTrigger:raise|DeviceType:heat|SettingValue:one|ValueType:degree',
        edit_distance_threshold=edit_distance_threshold
    )
    assert slot_has_conflict_numbers(
        nlu_hypothesis1='Music|PlayMusicIntent|AlbumName:frozen first',
        nlu_hypothesis2='Music|PlayMusicIntent|AlbumName:frozen second',
        edit_distance_threshold=edit_distance_threshold
    )
    assert slot_has_conflict_numbers(
        nlu_hypothesis1='Music|PlayMusicIntent|AlbumName:frozen two',
        nlu_hypothesis2='Music|PlayMusicIntent|AlbumName:frozen',
        edit_distance_threshold=edit_distance_threshold
    )
    assert not slot_has_conflict_numbers(
        nlu_hypothesis1='Video|ContentOnlyIntent|VideoName:six days in no',
        nlu_hypothesis2='Video|ContentOnlyIntent|VideoName:sixty days in',
        edit_distance_threshold=edit_distance_threshold
    )
    assert not slot_has_conflict_numbers(
        nlu_hypothesis1='HomeAutomation|SetValueIntent|DeviceType:light|SettingValue:one hundred|ValueType:percent',
        nlu_hypothesis2='HomeAutomation|SetValueIntent|ActionTrigger:turn|DeviceName:living room|DeviceType:lamp|SettingValue:a hundred|ValueType:percent',
        edit_distance_threshold=edit_distance_threshold
    )
    assert not slot_has_conflict_numbers(
        nlu_hypothesis1='HomeAutomation|SetValueIntent|DeviceName:a. c.|SettingValue:seven eighty eight',
        nlu_hypothesis2='HomeAutomation|SetValueIntent|DeviceName:a. c.|SettingValue:seventy eight',
        edit_distance_threshold=edit_distance_threshold
    )
    assert not slot_has_conflict_numbers(
        nlu_hypothesis1='Book|ReadBookIntent|BookName:harry potter book second',
        nlu_hypothesis2='Book|ReadBookIntent|BookName:harry potter book two',
        edit_distance_threshold=edit_distance_threshold
    )
    assert not slot_has_conflict_numbers(
        nlu_hypothesis1='Book|ReadBookIntent|BookName:harry potter book five',
        nlu_hypothesis2='Book|ReadBookIntent|BookName:harry potter and the order of the phoenix',
        edit_distance_threshold=edit_distance_threshold
    )
    assert not slot_has_conflict_numbers(
        nlu_hypothesis1='Music|PlayMusicIntent|AlbumName:frozen',
        nlu_hypothesis2='Music|PlayMusicIntent|AlbumName:frozen two',
        edit_distance_threshold=edit_distance_threshold
    )
    assert not slot_has_conflict_numbers(
        nlu_hypothesis1='HomeAutomation|SetValueIntent|ActionTrigger:set|DeviceLocation:boys room|DeviceName:one|DeviceType:light|SettingValue:four white',
        nlu_hypothesis2='HomeAutomation|SetValueIntent|ActionTrigger:set|DeviceName:boys room light one|SettingValue:white',
        edit_distance_threshold=edit_distance_threshold
    )
    assert not slot_has_conflict_numbers(
        nlu_hypothesis1='HomeAutomation|SetValueIntent|ActionTrigger:make|ActiveUserTrigger:my|DeviceType:light|SettingValue:one hundred bright|ValueType:percent',
        nlu_hypothesis2='HomeAutomation|SetValueIntent|ActionTrigger:make|ActiveUserTrigger:my|DeviceType:light|Setting:brightness|SettingValue:hundred',
        edit_distance_threshold=edit_distance_threshold
    )
    assert not slot_has_conflict_numbers(
        nlu_hypothesis1='HomeAutomation|SetValueIntent|ActionTrigger:set|DeviceType:a. c.|SettingValue:eighteen eighty',
        nlu_hypothesis2='HomeAutomation|SetValueIntent|ActionTrigger:set|DeviceType:a. c.|SettingValue:eighty',
        edit_distance_threshold=edit_distance_threshold
    )
    assert not slot_has_conflict_numbers(
        nlu_hypothesis1='HomeAutomation|SetValueIntent|ActiveUserTrigger:my|DeviceType:lights|SettingValue:brighten ten a hundred|ValueType:percent',
        nlu_hypothesis2='HomeAutomation|SetValueIntent|ActionTrigger:brighten|ActiveUserTrigger:my|DeviceType:lights|SettingValue:a hundred|ValueType:percent',
        edit_distance_threshold=edit_distance_threshold
    )
