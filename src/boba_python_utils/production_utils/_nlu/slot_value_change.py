from functools import partial
from typing import Iterable, Union, Callable

from attr import attrs, attrib
from pyspark.sql.functions import udf
from pyspark.sql.types import FloatType, BooleanType

from boba_python_utils.general_utils.algorithms.pair_alignment import iter_best_scored_pairs_of_distinct_source
from boba_python_utils.common_utils.iter_helper import iter__
from boba_python_utils.general_utils.nlp_utility.metrics.edit_distance import EditDistanceOptions, edit_distance, equals_in_tokens
from boba_python_utils.general_utils.nlp_utility.punctuations import remove_acronym_periods_and_spaces, remove_punctuation_except_for_hyphen
from boba_python_utils.general_utils.nlp_utility.string_sanitization import StringSanitizationOptions, StringSanitizationConfig
from boba_python_utils.general_utils.strex import has_token
from boba_python_utils.production_utils._nlu.entity_pairing import ENTITY_STOP_PREFIX_TOKENS
from boba_python_utils.production_utils._nlu.hypothesis_parsing import get_hypothesis_splits, get_slots_dict_from_hypothesis_splits
from boba_python_utils.production_utils._nlu.target_slots import is_for_comparison_slot_type


def _get_slot_drop_score(boolean: bool):
    return True if boolean else 1.0


def _get_no_slot_drop_or_change_score(boolean: bool):
    return False if boolean else 0.0


def _remove_acronym_and_punctuations(text):
    return remove_punctuation_except_for_hyphen(remove_acronym_periods_and_spaces(text))


@attrs(slots=True)
class SlotChangeOptions:
    consider_utterance = attrib(type=bool, default=True)
    allows_slot_type_change = attrib(type=bool, default=False)
    allows_slot_type_change_to_unspecified_slot_type = attrib(type=bool, default=False)
    allows_query_slot_value_being_substr = attrib(type=bool, default=True)
    allows_rewrite_slot_value_being_substr = attrib(type=bool, default=True)


def has_slot_change_or_drop(
        query_hypothesis: str,
        rewrite_hypothesis: str,
        query_utterance: str = None,
        rewrite_utterance: str = None,
        # region slot change config
        slot_types=None,
        allows_slot_value_change: bool = False,
        allows_slot_drop: bool = True,
        slot_change_options: SlotChangeOptions = None,
        # endregion
        # region edit distance config
        sanitization_config: Union[Iterable[StringSanitizationOptions], StringSanitizationConfig] = None,
        edit_distance_consider_sorted_tokens: Union[bool, Callable] = min,
        edit_distance_consider_same_num_tokens: Union[bool, Callable] = min,
        edit_distance_options: EditDistanceOptions = None,
        # endregion
        # region misc
        binary_score=False,
        allows_slot_value_change_for_extra_rewrite_slot_values=None,
        enable_for_extra_query_slot_values=None,
        disable_for_query_slot_values=None,
        query_slot_value_alternatives=None,
        # endregion
        **kwargs
):
    if not isinstance(sanitization_config, StringSanitizationConfig):
        sanitization_config = StringSanitizationConfig(actions=sanitization_config)
    if slot_change_options:
        allows_slot_type_change = slot_change_options.allows_slot_type_change
        allows_slot_type_change_to_unspecified_slot_type = slot_change_options.allows_slot_type_change_to_unspecified_slot_type
        allows_query_slot_value_being_substr = slot_change_options.allows_query_slot_value_being_substr
        allows_rewrite_slot_value_being_substr = slot_change_options.allows_rewrite_slot_value_being_substr
        if not slot_change_options.consider_utterance:
            query_utterance = rewrite_utterance = None
    else:
        allows_slot_type_change = allows_slot_type_change_to_unspecified_slot_type = \
            allows_query_slot_value_being_substr = allows_rewrite_slot_value_being_substr = False

    # region STEP1: derives hypothesis parts, and then extracts the slots
    if (not query_hypothesis) or (not rewrite_hypothesis):
        return None

    hyp_splits1, hyp_splits2 = (
        get_hypothesis_splits(query_hypothesis),
        get_hypothesis_splits(rewrite_hypothesis)
    )

    if len(hyp_splits1) <= 2:
        return _get_no_slot_drop_or_change_score(binary_score)

    slots1, slots2 = (
        get_slots_dict_from_hypothesis_splits(hyp_splits1, return_empty_dict_if_no_slots=True),
        get_slots_dict_from_hypothesis_splits(hyp_splits2, return_empty_dict_if_no_slots=True)
    )

    # endregion

    # region STEP2: reading the entity values
    ignore_acronym_periods_and_spaces = (
            bool(sanitization_config.actions) and
            StringSanitizationOptions.REMOVE_ACRONYMS_PERIODS_AND_SPACES in sanitization_config.actions
    )
    slot_values = []
    all_rewrite_slot_values = set()
    if (
            allows_slot_type_change and
            allows_slot_type_change_to_unspecified_slot_type and
            (
                    (slot_types is None) or
                    (
                            # len(slot_types) == 1 and
                            len(slots1) == 1 and
                            len(slots2) == 1 and
                            (
                                    next(iter(slots1)) in slot_types or
                                    next(iter(slots2)) in slot_types
                            )
                    )
            )
    ):
        slot_types = list(
            set(filter(is_for_comparison_slot_type, slots1)) | set(filter(is_for_comparison_slot_type, slots2))
        )

    def _diff_request_rewrite_slot_value_with_utterance(
            slot_value1,
            slot_value2,
            utterance1,
            utterance2
    ):
        # `slot_value1` is ensured not None
        if slot_value1 == slot_value2:
            return False

        request_slot_value_is_substring = (
                slot_value2 is not None and
                slot_value1 in slot_value2
        )
        if request_slot_value_is_substring:
            return not allows_query_slot_value_being_substr

        rewrite_slot_value_is_substring = (
                slot_value2 is not None and
                slot_value2 in slot_value1
        )
        if rewrite_slot_value_is_substring:
            return not allows_rewrite_slot_value_being_substr

        return (
                (
                        (utterance1 is None) or
                        not has_token(
                            utterance1,
                            slot_value2
                        )
                ) or
                (
                        (utterance2 is None) or
                        not has_token(
                            utterance2,
                            slot_value1
                        )
                )
        )

    for slot_type in iter__(slot_types):
        if slot_type in slots1:
            request_slot_value = slots1[slot_type]
            # `request_entity_value_constraint` is set of values and this rule ONLY triggers if the `request_entity_value` is in this set;
            # for example, when we block 'StationName' slot value change, we only block it if the slot value is in a pre-defined set {'c. n. n.', 'b. b. c.', ...}
            if (
                    request_slot_value and
                    ((not enable_for_extra_query_slot_values) or request_slot_value in enable_for_extra_query_slot_values) and
                    ((not disable_for_query_slot_values) or request_slot_value not in disable_for_query_slot_values)
            ):
                rewrite_slot_value = slots2.get(slot_type, None) if slots2 else None
                if _diff_request_rewrite_slot_value_with_utterance(
                        request_slot_value, rewrite_slot_value,
                        request_slot_value, rewrite_slot_value
                ):
                    slot_values.append((request_slot_value, rewrite_slot_value))
        if slots2 and slot_type in slots2:
            slot_value = slots2[slot_type]
            if slot_value:
                all_rewrite_slot_values.add(slot_value)
    # endregion

    all_scores = []
    if query_utterance is not None:
        query_utterance_sanitized = _remove_acronym_and_punctuations(query_utterance)
    else:
        query_utterance_sanitized = None

    if rewrite_utterance is not None:
        rewrite_utterance_sanitized = _remove_acronym_and_punctuations(rewrite_utterance)
    else:
        rewrite_utterance_sanitized = None

    def _add_score():
        if allows_slot_type_change:
            all_scores.append((score, request_slot_value, rewrite_slot_value))
        else:
            all_scores.append(score)

    for request_slot_value, rewrite_slot_value in slot_values:
        rewrite_slot_values = (
            [
                rewrite_slot_value,
                *(x for x in all_rewrite_slot_values if x != rewrite_slot_value)
            ]
            if allows_slot_type_change
            else [rewrite_slot_value]
        )
        for i, rewrite_slot_value in enumerate(rewrite_slot_values):
            # `request_entity_value_alternatives` is a dictionary for kind of "synonyms" for request entity values,
            # e.g. `{'light': ['lights', 'lamp', 'lamps'], 'lamp': ['light', 'lights', 'lamps']}`.
            if query_slot_value_alternatives is not None and request_slot_value in query_slot_value_alternatives:
                _request_slot_value_alternatives = query_slot_value_alternatives[request_slot_value]
            else:
                _request_slot_value_alternatives = None

            # region STEP3.1: checking if there is entity drop
            if i == 0 and rewrite_slot_value is None and (not allows_slot_drop):
                if rewrite_utterance is None:
                    # if the `rewrite_utterance` is not provided, then we consider there is entity drop
                    return _get_slot_drop_score(binary_score)
                else:
                    # otherwise, we will check if the `request_entity_value` and its "alternatives" exists in the `rewrite`; if it exists_path, then there is no slot value drop
                    request_entity_value_not_in_rewrite_utterance = not has_token(rewrite_utterance_sanitized, _remove_acronym_and_punctuations(request_slot_value))
                    if _request_slot_value_alternatives:
                        # checking slot value drop with the alternatives
                        request_entity_value_alternatives_not_in_rewrite_utterance = all(
                            not has_token(rewrite_utterance_sanitized, _remove_acronym_and_punctuations(request_slot_value_alternative))
                            for request_slot_value_alternative in _request_slot_value_alternatives
                        )
                        if request_entity_value_not_in_rewrite_utterance and request_entity_value_alternatives_not_in_rewrite_utterance:
                            return _get_slot_drop_score(binary_score)
                    elif request_entity_value_not_in_rewrite_utterance:
                        return _get_slot_drop_score(binary_score)
            # endregion

            # region STEP3.2: checking if there is entity value change
            # `allows_slot_value_change_for_these_rewrite_slot_values` is a set of slot values that if the rewrite slot value is in this set, then we allow the slot value change
            if (
                    (
                            (not allows_slot_value_change) or
                            (
                                    allows_slot_value_change_for_extra_rewrite_slot_values is not None and
                                    rewrite_slot_value in allows_slot_value_change_for_extra_rewrite_slot_values
                            )
                    ) and
                    (rewrite_slot_value is not None)
            ):
                _request_slot_value = _remove_acronym_and_punctuations(request_slot_value)
                _rewrite_slot_value = _remove_acronym_and_punctuations(rewrite_slot_value)
                if _request_slot_value_alternatives:
                    __request_slot_value_alternatives = list(map(
                        _remove_acronym_and_punctuations, _request_slot_value_alternatives
                    ))

                if ignore_acronym_periods_and_spaces:
                    request_slot_value = _request_slot_value
                    rewrite_slot_value = _rewrite_slot_value
                    if _request_slot_value_alternatives:
                        _request_slot_value_alternatives = __request_slot_value_alternatives

                if (
                        (
                                not _diff_request_rewrite_slot_value_with_utterance(
                                    _request_slot_value, _rewrite_slot_value,
                                    query_utterance_sanitized, rewrite_utterance_sanitized
                                )
                        ) or
                        (
                                _request_slot_value_alternatives and
                                any(
                                    not _diff_request_rewrite_slot_value_with_utterance(
                                        request_slot_value_alternative, _rewrite_slot_value,
                                        query_utterance_sanitized, rewrite_utterance_sanitized
                                    )
                                    for request_slot_value_alternative
                                    in __request_slot_value_alternatives
                                )
                        )
                ):
                    score = 0.0
                    _add_score()
                    break

                if binary_score:
                    score = (not equals_in_tokens(request_slot_value, rewrite_slot_value)) and (
                            _request_slot_value_alternatives is None or
                            all(
                                (not equals_in_tokens(request_slot_value_alternative, rewrite_slot_value))
                                for request_slot_value_alternative in _request_slot_value_alternatives
                            )
                    )
                    _add_score()
                else:
                    base_edit_dist_ratio = edit_distance(
                        request_slot_value,
                        rewrite_slot_value,
                        consider_sorted_tokens=edit_distance_consider_sorted_tokens,
                        sanitization_config=sanitization_config,
                        consider_same_num_tokens=edit_distance_consider_same_num_tokens,
                        return_ratio=True,
                        options=edit_distance_options,
                        **kwargs
                    )
                    if _request_slot_value_alternatives:
                        score = 1 - max(
                            base_edit_dist_ratio,
                            *(
                                edit_distance(
                                    request_slot_value_alternative,
                                    rewrite_slot_value,
                                    consider_sorted_tokens=edit_distance_consider_sorted_tokens,
                                    sanitization_config=sanitization_config,
                                    consider_same_num_tokens=edit_distance_consider_same_num_tokens,
                                    return_ratio=True,
                                    options=edit_distance_options,
                                    **kwargs
                                )
                                for request_slot_value_alternative in _request_slot_value_alternatives
                            )
                        )
                        _add_score()
                    else:
                        score = 1 - base_edit_dist_ratio
                        _add_score()

    # endregion

    if not all_scores:
        return _get_no_slot_drop_or_change_score(binary_score)

    if allows_slot_type_change:
        if ignore_acronym_periods_and_spaces:
            all_rewrite_slot_values = map(_remove_acronym_and_punctuations, all_rewrite_slot_values)
        return max(next(zip(*iter_best_scored_pairs_of_distinct_source(all_scores, all_rewrite_slot_values))))
    else:
        return max(all_scores)


def has_slot_change_or_drop_udf(
        hypothesis1,
        hypothesis2,
        utterance1=None,
        utterance2=None,
        slot_types=None,
        consider_sorted_tokens_for_edit_distance=True,
        consider_same_num_tokens_for_edit_distance=True,
        allows_slot_type_change=True,
        sanitization_config: Union[Iterable[StringSanitizationOptions], StringSanitizationConfig] = None,
        binary_score=False,
        **kwargs
):
    _has_slot_change_or_drop_udf = udf(
        partial(
            has_slot_change_or_drop,
            slot_types=slot_types,
            consider_sorted_tokens=consider_sorted_tokens_for_edit_distance,
            consider_same_num_tokens=consider_same_num_tokens_for_edit_distance,
            allows_slot_type_change=allows_slot_type_change,
            sanitization_config=sanitization_config,
            binary_score=binary_score,
            **kwargs
        ),
        returnType=(BooleanType() if binary_score else FloatType())
    )

    if utterance1 and utterance2:
        return _has_slot_change_or_drop_udf(hypothesis1, hypothesis2, utterance1, utterance2)
    else:
        return _has_slot_change_or_drop_udf(hypothesis1, hypothesis2)


if __name__ == '__main__':
    query_utterance = "play songs by benee"
    rewrite_utterance = 'play songs by her'
    query_hypothesis = "Music|PlayMusicIntent|ArtistName:benee|MediaType:songs"
    rewrite_hypothesis = "Music|PlayMusicIntent|ArtistName:her|MediaType:songs"
    print(
        has_slot_change_or_drop(
            query_utterance=query_utterance,
            rewrite_utterance=rewrite_utterance,
            query_hypothesis=query_hypothesis,
            rewrite_hypothesis=rewrite_hypothesis,
            sanitization_config=[
                StringSanitizationOptions.REMOVE_ACRONYMS_PERIODS_AND_SPACES,
                StringSanitizationOptions.REMOVE_PUNCTUATIONS_EXCEPT_FOR_HYPHEN,
                StringSanitizationOptions.MAKE_FUZZY
            ],
            prefixes=ENTITY_STOP_PREFIX_TOKENS,
            weight_if_slot_values_have_common_start=0.9,
            min_slot_value_common_start_to_enable_weight=3,
            allows_slot_type_change=True
        )
    )

    query_utterance = "play airport noise"
    rewrite_utterance = 'play airplane sounds'
    query_hypothesis = "Music|PlayMusicIntent|SongName:airport noise"
    rewrite_hypothesis = "GeneralMedia|LaunchNativeAppIntent|AppName:airplane sounds"
    print(
        has_slot_change_or_drop(
            query_utterance=query_utterance,
            rewrite_utterance=rewrite_utterance,
            query_hypothesis=query_hypothesis,
            rewrite_hypothesis=rewrite_hypothesis,
            sanitization_config=[
                StringSanitizationOptions.REMOVE_ACRONYMS_PERIODS_AND_SPACES,
                StringSanitizationOptions.REMOVE_PUNCTUATIONS_EXCEPT_FOR_HYPHEN,
                StringSanitizationOptions.REMOVE_COMMON_PREFIX,
                StringSanitizationOptions.REMOVE_COMMON_SUFFIX,
                StringSanitizationOptions.MAKE_FUZZY
            ],
            prefixes=ENTITY_STOP_PREFIX_TOKENS,
            weight_if_slot_values_have_common_start=0.9,
            min_slot_value_common_start_to_enable_weight=3,
            allows_slot_type_change=True
        )
    )

    query_utterance = "play pop county boogie by fat"
    rewrite_utterance = 'play pike county boogie by the fade'
    query_hypothesis = "Music|PlayMusicIntent|ArtistName:fat|SongName:pop county boogie"
    rewrite_hypothesis = "Music|PlayMusicIntent|ArtistName:the fade|SongName:pike county boogie"
    print(
        has_slot_change_or_drop(
            query_utterance=query_utterance,
            rewrite_utterance=rewrite_utterance,
            query_hypothesis=query_hypothesis,
            rewrite_hypothesis=rewrite_hypothesis,
            sanitization_config=[
                StringSanitizationOptions.REMOVE_ACRONYMS_PERIODS_AND_SPACES,
                StringSanitizationOptions.REMOVE_PUNCTUATIONS_EXCEPT_FOR_HYPHEN,
                StringSanitizationOptions.REMOVE_PREFIX,
                StringSanitizationOptions.REMOVE_COMMON_SUFFIX,
                StringSanitizationOptions.MAKE_FUZZY
            ],
            weight_if_slot_values_have_common_start=0.9,
            min_slot_value_common_start_to_enable_weight=3,
            allows_slot_type_change=True
        )
    )

    request_utterance = "the crawl show"
    rewrite_utterance = 'kroll show'
    request_hypothesis = "Video|ContentOnlyIntent|MediaType:show|VideoName:crawl"
    rewrite_hypothesis = "Video|ContentOnlyIntent|VideoName:kroll show"
    print(
        has_slot_change_or_drop(
            query_utterance=request_utterance,
            rewrite_utterance=rewrite_utterance,
            query_hypothesis=request_hypothesis,
            rewrite_hypothesis=rewrite_hypothesis,
            sanitization_config=[StringSanitizationOptions.MAKE_FUZZY],
            allows_slot_type_change=True
        )
    )

    request_hypothesis = "Communication|CallIntent|CallType:call|ContactName:monk"
    rewrite_hypothesis = "Communication|CallIntent|CallType:call|ContactName:mom"
    print(
        has_slot_change_or_drop(
            query_hypothesis=request_hypothesis,
            rewrite_hypothesis=rewrite_hypothesis,
            query_utterance=None,
            rewrite_utterance=None,
            slot_types=['ContactName'],
            allows_query_slot_value_being_substr=True,
            allows_rewrite_slot_value_being_substr=False,
            edit_distance_consider_sorted_tokens=False,
            edit_distance_consider_same_num_tokens=False,
            sanitization_config=[
                StringSanitizationOptions.REMOVE_ACRONYMS_PERIODS_AND_SPACES,
                StringSanitizationOptions.REMOVE_PUNCTUATIONS_EXCEPT_FOR_HYPHEN,
                StringSanitizationOptions.REMOVE_COMMON_PREFIX,
                StringSanitizationOptions.REMOVE_COMMON_SUFFIX,
                StringSanitizationOptions.MAKE_FUZZY
            ]
        )
    )

    request_hypothesis = "Communication|CallIntent|CallType:call|ContactName:jeff richardson"
    rewrite_hypothesis = "Communication|CallIntent|CallType:call|ContactName:jeffrey richardson"
    print(
        has_slot_change_or_drop(
            query_hypothesis=request_hypothesis,
            rewrite_hypothesis=rewrite_hypothesis,
            query_utterance=None,
            rewrite_utterance=None,
            slot_types=['ContactName'],
            allows_query_slot_value_being_substr=True,
            allows_rewrite_slot_value_being_substr=False,
            edit_distance_consider_sorted_tokens=False,
            edit_distance_consider_same_num_tokens=False,
            sanitization_config=[
                StringSanitizationOptions.REMOVE_ACRONYMS_PERIODS_AND_SPACES,
                StringSanitizationOptions.REMOVE_PUNCTUATIONS_EXCEPT_FOR_HYPHEN,
                StringSanitizationOptions.REMOVE_COMMON_PREFIX,
                StringSanitizationOptions.REMOVE_COMMON_SUFFIX,
                StringSanitizationOptions.MAKE_FUZZY
            ]
        )
    )
    request_hypothesis = "Communication|CallIntent|CallType:call|ContactName:sonya"
    rewrite_hypothesis = "Communication|CallIntent|CallType:call|ContactName:mia"
    print(
        has_slot_change_or_drop(
            query_hypothesis=request_hypothesis,
            rewrite_hypothesis=rewrite_hypothesis,
            query_utterance=None,
            rewrite_utterance=None,
            slot_types=['ContactName'],
            allows_query_slot_value_being_substr=True,
            allows_rewrite_slot_value_being_substr=False,
            edit_distance_consider_sorted_tokens=False,
            edit_distance_consider_same_num_tokens=False,
            sanitization_config=[
                StringSanitizationOptions.REMOVE_ACRONYMS_PERIODS_AND_SPACES,
                StringSanitizationOptions.REMOVE_PUNCTUATIONS_EXCEPT_FOR_HYPHEN,
                StringSanitizationOptions.REMOVE_COMMON_PREFIX,
                StringSanitizationOptions.REMOVE_COMMON_SUFFIX,
                StringSanitizationOptions.MAKE_FUZZY
            ]
        )
    )

    request_hypothesis = "Communication|CallIntent|CallType:call|ContactName:knight"
    rewrite_hypothesis = "Communication|CallIntent|CallType:call|ContactName:nike"
    print(
        has_slot_change_or_drop(
            query_hypothesis=request_hypothesis,
            rewrite_hypothesis=rewrite_hypothesis,
            query_utterance=None,
            rewrite_utterance=None,
            slot_types=['ContactName'],
            allows_query_slot_value_being_substr=True,
            allows_rewrite_slot_value_being_substr=False,
            edit_distance_consider_sorted_tokens=False,
            edit_distance_consider_same_num_tokens=False,
            sanitization_config=[
                StringSanitizationOptions.REMOVE_ACRONYMS_PERIODS_AND_SPACES,
                StringSanitizationOptions.REMOVE_PUNCTUATIONS_EXCEPT_FOR_HYPHEN,
                StringSanitizationOptions.REMOVE_COMMON_PREFIX,
                StringSanitizationOptions.REMOVE_COMMON_SUFFIX,
                StringSanitizationOptions.MAKE_FUZZY
            ]
        )
    )

    request_hypothesis = "Communication|InstantConnectIntent|CallType:drop in|ContactName:sara halper"
    rewrite_hypothesis = "Communication|InstantConnectIntent|CallType:drop in|ContactName:susan halper"
    print(
        has_slot_change_or_drop(
            query_hypothesis=request_hypothesis,
            rewrite_hypothesis=rewrite_hypothesis,
            query_utterance=None,
            rewrite_utterance=None,
            slot_types=['ContactName'],
            allows_query_slot_value_being_substr=True,
            allows_rewrite_slot_value_being_substr=False,
            edit_distance_consider_sorted_tokens=False,
            edit_distance_consider_same_num_tokens=False,
            sanitization_config=[
                StringSanitizationOptions.REMOVE_ACRONYMS_PERIODS_AND_SPACES,
                StringSanitizationOptions.REMOVE_PUNCTUATIONS_EXCEPT_FOR_HYPHEN,
                StringSanitizationOptions.REMOVE_COMMON_PREFIX,
                StringSanitizationOptions.REMOVE_COMMON_SUFFIX,
                StringSanitizationOptions.MAKE_FUZZY
            ]
        )
    )

    request_utterance = "drop in to wesley's room"
    rewrite_utterance = 'drop in to wes room'
    request_hypothesis = "Communication|InstantConnectIntent|CallType:drop in|ContactName:wesley's|DeviceLocation:room"
    rewrite_hypothesis = "Communication|InstantConnectIntent|CallType:drop in|ContactName:wes|DeviceLocation:room"
    print(
        has_slot_change_or_drop(
            query_hypothesis=request_hypothesis,
            rewrite_hypothesis=rewrite_hypothesis,
            query_utterance=None,
            rewrite_utterance=None,
            slot_types=['ContactName'],
            allows_query_slot_value_being_substr=True,
            allows_rewrite_slot_value_being_substr=False,
            edit_distance_consider_sorted_tokens=False,
            edit_distance_consider_same_num_tokens=False,
            sanitization_config=[
                StringSanitizationOptions.REMOVE_ACRONYMS_PERIODS_AND_SPACES,
                StringSanitizationOptions.REMOVE_PUNCTUATIONS_EXCEPT_FOR_HYPHEN,
                StringSanitizationOptions.REMOVE_COMMON_PREFIX,
                StringSanitizationOptions.REMOVE_COMMON_SUFFIX,
                StringSanitizationOptions.MAKE_FUZZY
            ]
        )
    )

    request_utterance = 'drop in hamed'
    rewrite_utterance = 'drop in fairuz'
    request_hypothesis = "Communication|InstantConnectIntent|CallType:drop in|ContactName:hamed"
    rewrite_hypothesis = "Communication|InstantConnectIntent|CallType:drop in|ContactName:fairuz"
    print(
        has_slot_change_or_drop(
            query_hypothesis=request_hypothesis,
            rewrite_hypothesis=rewrite_hypothesis,
            query_utterance=None,
            rewrite_utterance=None,
            slot_types=['ContactName'],
            allows_query_slot_value_being_substr=True,
            allows_rewrite_slot_value_being_substr=False,
            edit_distance_consider_sorted_tokens=False,
            edit_distance_consider_same_num_tokens=False,
            sanitization_config=[
                StringSanitizationOptions.REMOVE_ACRONYMS_PERIODS_AND_SPACES,
                StringSanitizationOptions.REMOVE_PUNCTUATIONS_EXCEPT_FOR_HYPHEN,
                StringSanitizationOptions.REMOVE_COMMON_PREFIX,
                StringSanitizationOptions.REMOVE_COMMON_SUFFIX,
                StringSanitizationOptions.MAKE_FUZZY
            ]
        )
    )

    request_utterance = 'call boo sultan'
    rewrite_utterance = 'call abu sultan'
    request_hypothesis = "Communication|CallIntent|CallType:call|ContactName:boo sultan"
    rewrite_hypothesis = "Communication|CallIntent|CallType:call|ContactName:abu sultan"
    print(
        has_slot_change_or_drop(
            query_hypothesis=request_hypothesis,
            rewrite_hypothesis=rewrite_hypothesis,
            query_utterance=None,
            rewrite_utterance=None,
            slot_types=['ContactName'],
            allows_query_slot_value_being_substr=True,
            allows_rewrite_slot_value_being_substr=False,
            edit_distance_consider_sorted_tokens=False,
            edit_distance_consider_same_num_tokens=False,
            sanitization_config=[
                StringSanitizationOptions.REMOVE_ACRONYMS_PERIODS_AND_SPACES,
                StringSanitizationOptions.REMOVE_PUNCTUATIONS_EXCEPT_FOR_HYPHEN,
                StringSanitizationOptions.REMOVE_COMMON_PREFIX,
                StringSanitizationOptions.REMOVE_COMMON_SUFFIX,
                StringSanitizationOptions.MAKE_FUZZY
            ]
        )
    )

    request_utterance = 'call nick kurian'
    rewrite_utterance = 'call nicholas kurian'
    request_hypothesis = "Communication|CallIntent|CallType:call|ContactName:nick kurian"
    rewrite_hypothesis = "Communication|CallIntent|CallType:call|ContactName:nicholas kurian"
    print(
        has_slot_change_or_drop(
            query_hypothesis=request_hypothesis,
            rewrite_hypothesis=rewrite_hypothesis,
            query_utterance=None,
            rewrite_utterance=None,
            slot_types=['ContactName'],
            allows_query_slot_value_being_substr=True,
            allows_rewrite_slot_value_being_substr=False,
            edit_distance_consider_sorted_tokens=False,
            edit_distance_consider_same_num_tokens=False,
            sanitization_config=[
                StringSanitizationOptions.REMOVE_ACRONYMS_PERIODS_AND_SPACES,
                StringSanitizationOptions.REMOVE_PUNCTUATIONS_EXCEPT_FOR_HYPHEN,
                StringSanitizationOptions.REMOVE_COMMON_PREFIX,
                StringSanitizationOptions.REMOVE_COMMON_SUFFIX,
                StringSanitizationOptions.MAKE_FUZZY
            ]
        )
    )

    request_utterance = 'call autumn pierce'
    rewrite_utterance = 'call autumn spears'
    request_hypothesis = "Communication|CallIntent|CallType:call|ContactName:autumn pierce"
    rewrite_hypothesis = "Communication|CallIntent|CallType:call|ContactName:autumn spears"
    print(
        has_slot_change_or_drop(
            query_hypothesis=request_hypothesis,
            rewrite_hypothesis=rewrite_hypothesis,
            query_utterance=None,
            rewrite_utterance=None,
            slot_types=['ContactName'],
            allows_query_slot_value_being_substr=True,
            allows_rewrite_slot_value_being_substr=False,
            edit_distance_consider_sorted_tokens=False,
            edit_distance_consider_same_num_tokens=False,
            sanitization_config=[
                StringSanitizationOptions.REMOVE_ACRONYMS_PERIODS_AND_SPACES,
                StringSanitizationOptions.REMOVE_PUNCTUATIONS_EXCEPT_FOR_HYPHEN,
                StringSanitizationOptions.REMOVE_COMMON_PREFIX,
                StringSanitizationOptions.REMOVE_COMMON_SUFFIX,
                StringSanitizationOptions.MAKE_FUZZY
            ]
        )
    )

    request_utterance = 'call jessie ross'
    rewrite_utterance = 'call jessie raasch'
    request_hypothesis = "Communication|CallIntent|CallType:call|ContactName:jessie ross"
    rewrite_hypothesis = "Communication|CallIntent|CallType:call|ContactName:jessie raasch"
    print(
        has_slot_change_or_drop(
            query_hypothesis=request_hypothesis,
            rewrite_hypothesis=rewrite_hypothesis,
            query_utterance=None,
            rewrite_utterance=None,
            slot_types=['ContactName'],
            allows_query_slot_value_being_substr=True,
            allows_rewrite_slot_value_being_substr=False,
            edit_distance_consider_sorted_tokens=False,
            edit_distance_consider_same_num_tokens=False,
            sanitization_config=[
                StringSanitizationOptions.REMOVE_ACRONYMS_PERIODS_AND_SPACES,
                StringSanitizationOptions.REMOVE_PUNCTUATIONS_EXCEPT_FOR_HYPHEN,
                StringSanitizationOptions.REMOVE_COMMON_PREFIX,
                StringSanitizationOptions.REMOVE_COMMON_SUFFIX,
                StringSanitizationOptions.MAKE_FUZZY
            ]
        )
    )

    request_utterance = 'play my playlist leaks'
    rewrite_utterance = 'play my playlist t. t.'
    request_hypothesis = "Music|PlayMusicIntent|ActiveUserTrigger:my|MediaType:playlist|PlaylistName:leaks"
    rewrite_hypothesis = "Music|PlayMusicIntent|ActiveUserTrigger:my|MediaType:playlist|PlaylistName:t. t."
    print(
        has_slot_change_or_drop(
            query_hypothesis=request_hypothesis,
            rewrite_hypothesis=rewrite_hypothesis,
            query_utterance=request_utterance,
            rewrite_utterance=rewrite_utterance,
            sanitization_config=[
                StringSanitizationOptions.REMOVE_ACRONYMS_PERIODS_AND_SPACES,
                StringSanitizationOptions.REMOVE_PUNCTUATIONS_EXCEPT_FOR_HYPHEN,
                StringSanitizationOptions.MAKE_FUZZY,
                StringSanitizationOptions.REMOVE_SPACES
            ],
            edit_distance_consider_sorted_tokens=True,
            edit_distance_consider_same_num_tokens=True,
            allows_slot_type_change=True
        )
    )

    request_utterance = 'play duty truck'
    rewrite_utterance = 'play doo doo'
    request_hypothesis = "Music|PlayMusicIntent|SongName:duty truck"
    rewrite_hypothesis = "Music|PlayMusicIntent|SongName:doo doo"
    print(
        has_slot_change_or_drop(
            query_hypothesis=request_hypothesis,
            rewrite_hypothesis=rewrite_hypothesis,
            query_utterance=request_utterance,
            rewrite_utterance=rewrite_utterance,
            sanitization_config=[
                StringSanitizationOptions.REMOVE_ACRONYMS_PERIODS_AND_SPACES,
                StringSanitizationOptions.REMOVE_PUNCTUATIONS_EXCEPT_FOR_HYPHEN,
                StringSanitizationOptions.MAKE_FUZZY,
                StringSanitizationOptions.REMOVE_SPACES
            ],
            edit_distance_consider_sorted_tokens=True,
            edit_distance_consider_same_num_tokens=True,
            allows_slot_type_change=True
        )
    )

    request_utterance = 'play this is what by marilyn manson'
    rewrite_utterance = 'play no more dream'
    request_hypothesis = "Music|PlayMusicIntent|SongName:this is what|ArtistName:marilyn manson"
    rewrite_hypothesis = "Music|PlayMusicIntent|SongName:no more dream"
    print(
        has_slot_change_or_drop(
            query_hypothesis=request_hypothesis,
            rewrite_hypothesis=rewrite_hypothesis,
            query_utterance=request_utterance,
            rewrite_utterance=rewrite_utterance,
            sanitization_config=[
                StringSanitizationOptions.REMOVE_ACRONYMS_PERIODS_AND_SPACES,
                StringSanitizationOptions.REMOVE_PUNCTUATIONS_EXCEPT_FOR_HYPHEN,
                StringSanitizationOptions.MAKE_FUZZY,
                StringSanitizationOptions.REMOVE_SPACES
            ],
            edit_distance_consider_sorted_tokens=True,
            edit_distance_consider_same_num_tokens=True,
            allows_slot_type_change=True
        )
    )

    request_utterance = 'play legendary by finger death punch'
    rewrite_utterance = 'play my liked songs'
    request_hypothesis = "Music|PlayMusicIntent|SongName:legendary|ArtistName:finger death punch"
    rewrite_hypothesis = "Music|PlayMusicIntent|SongName:no more dream"
    print(
        has_slot_change_or_drop(
            query_hypothesis=request_hypothesis,
            rewrite_hypothesis=rewrite_hypothesis,
            query_utterance=request_utterance,
            rewrite_utterance=rewrite_utterance,
            sanitization_config=[
                StringSanitizationOptions.REMOVE_ACRONYMS_PERIODS_AND_SPACES,
                StringSanitizationOptions.REMOVE_PUNCTUATIONS_EXCEPT_FOR_HYPHEN,
                StringSanitizationOptions.MAKE_FUZZY,
                StringSanitizationOptions.REMOVE_SPACES
            ],
            edit_distance_consider_sorted_tokens=True,
            edit_distance_consider_same_num_tokens=True,
            allows_slot_type_change=True
        )
    )

    request_hypothesis = "Music|PlayMusicIntent|ArtistName:j. i."
    rewrite_hypothesis = "Music|PlayMusicIntent|ArtistName:j.i the prince of n.y|MediaType:songs"
    print(
        has_slot_change_or_drop(
            query_hypothesis=request_hypothesis,
            rewrite_hypothesis=rewrite_hypothesis,
            sanitization_config=[
                StringSanitizationOptions.REMOVE_ACRONYMS_PERIODS_AND_SPACES,
                StringSanitizationOptions.REMOVE_PUNCTUATIONS_EXCEPT_FOR_HYPHEN,
                StringSanitizationOptions.MAKE_FUZZY,
                StringSanitizationOptions.REMOVE_SPACES
            ],
            edit_distance_consider_sorted_tokens=True,
            edit_distance_consider_same_num_tokens=True,
            allows_slot_type_change=True
        )
    )

    request_hypothesis = "Music|PlayMusicIntent|ArtistName:cancan|SongName:ending"
    rewrite_hypothesis = "Music|PlayMusicIntent|ArtistName:capcom sound team|SongName:an ending"
    print(
        has_slot_change_or_drop(
            query_hypothesis=request_hypothesis,
            rewrite_hypothesis=rewrite_hypothesis,
            sanitization_config=[StringSanitizationOptions.MAKE_FUZZY, StringSanitizationOptions.REMOVE_SPACES],
            edit_distance_consider_sorted_tokens=True,
            allows_slot_type_change=True
        )
    )
    request_hypothesis = "Music|PlayStationIntent|AppName:tunein|CallSign:k. f. m."
    rewrite_hypothesis = "Music|PlayStationIntent|AppName:tunein|RadioBand:f. m.|StationName:gay"
    print(
        has_slot_change_or_drop(
            query_hypothesis=request_hypothesis,
            rewrite_hypothesis=rewrite_hypothesis,
            sanitization_config=[StringSanitizationOptions.MAKE_FUZZY],
            allows_slot_type_change=True
        )
    )

    request_hypothesis = "Music|PlayStationIntent|ArtistName:schwinn|MediaType:station"
    rewrite_hypothesis = "Music|PlayStationIntent|ArtistName:shwekey|MediaType:station"
    print(
        has_slot_change_or_drop(
            query_hypothesis=request_hypothesis,
            rewrite_hypothesis=rewrite_hypothesis,
            sanitization_config=[StringSanitizationOptions.MAKE_FUZZY],
            allows_slot_type_change=True,
        )
    )

    request_hypothesis = "Communication|CallIntent|ContactName:joe"
    rewrite_hypothesis = "Communication|CallIntent|ContactName:john"
    print(
        has_slot_change_or_drop(
            query_hypothesis=request_hypothesis,
            rewrite_hypothesis=rewrite_hypothesis,
            slot_types=['ContactName'],
            sanitization_config=[StringSanitizationOptions.MAKE_FUZZY],
            allows_slot_type_change=True
        )
    )

    request_hypothesis = "Music|PlayStationIntent|AppName:pandora|SongName:as we"
    rewrite_hypothesis = "Music|PlayStationIntent|AppName:pandora|ArtistName:ash b."
    print(
        has_slot_change_or_drop(
            query_hypothesis=request_hypothesis,
            rewrite_hypothesis=rewrite_hypothesis,
            sanitization_config=[StringSanitizationOptions.MAKE_FUZZY],
            allows_slot_type_change=True
        )
    )

    request_hypothesis = "Music|PlayStationIntent|AppName:pandora|SongName:narcos"
    rewrite_hypothesis = "Music|PlayStationIntent|AppName:pandora|ArtistName:the knockouts"
    print(
        has_slot_change_or_drop(
            query_hypothesis=request_hypothesis,
            rewrite_hypothesis=rewrite_hypothesis,
            sanitization_config=[StringSanitizationOptions.MAKE_FUZZY],
            allows_slot_type_change=True,
        )
    )

    request_hypothesis = "Music|PlayMusicIntent|ArtistName:aqua|MediaType:songs"
    rewrite_hypothesis = "Music|PlayMusicIntent|ArtistName:akwid|MediaType:songs"
    print(
        has_slot_change_or_drop(
            query_hypothesis=request_hypothesis,
            rewrite_hypothesis=rewrite_hypothesis,
            sanitization_config=[StringSanitizationOptions.MAKE_FUZZY],
            allows_slot_type_change=True
        )
    )

    request_hypothesis = "Music|PlayMusicIntent|ArtistName:eva ma|ShuffleTrigger:shuffle"
    rewrite_hypothesis = "Music|PlayMusicIntent|ArtistName:yiruma|ShuffleTrigger:shuffle"
    print(
        has_slot_change_or_drop(
            query_hypothesis=request_hypothesis,
            rewrite_hypothesis=rewrite_hypothesis,
            sanitization_config=[StringSanitizationOptions.MAKE_FUZZY],
            allows_slot_type_change=True
        )
    )

    request_hypothesis = "Music|PlayMusicIntent|ArtistName:partynextdoor|SongName:you've been messed"
    rewrite_hypothesis = "Music|PlayMusicIntent|SongName:you've been missed"
    print(
        has_slot_change_or_drop(
            query_hypothesis=request_hypothesis,
            rewrite_hypothesis=rewrite_hypothesis,
            sanitization_config=[StringSanitizationOptions.MAKE_FUZZY],
            allows_slot_type_change=True
        )
    )

    request_hypothesis = "Music|PlayMusicIntent|ArtistName:john smith|SongName:hymns about words"
    rewrite_hypothesis = "Music|PlayMusicIntent|AlbumName:hymns without words"
    print(
        has_slot_change_or_drop(
            query_hypothesis=request_hypothesis,
            rewrite_hypothesis=rewrite_hypothesis,
            sanitization_config=[StringSanitizationOptions.MAKE_FUZZY],
            allows_slot_type_change=True
        )
    )

    request_hypothesis = "Music|PlayMusicIntent|ArtistName:soundcheck|SongName:current"
    rewrite_hypothesis = "Music|PlayMusicIntent|AlbumName:currents|MediaType:album"
    print(
        has_slot_change_or_drop(
            query_hypothesis=request_hypothesis,
            rewrite_hypothesis=rewrite_hypothesis,
            sanitization_config=[StringSanitizationOptions.MAKE_FUZZY],
            allows_slot_type_change=True
        )
    )

    request_hypothesis = "Music|PlayMusicIntent|AlbumName:too fast for love|ArtistName:motley crue"
    rewrite_hypothesis = "Music|PlayMusicIntent|MediaType:song|SongName:too fast for love"
    print(
        has_slot_change_or_drop(
            query_hypothesis=request_hypothesis,
            rewrite_hypothesis=rewrite_hypothesis,
            sanitization_config=[StringSanitizationOptions.MAKE_FUZZY],
            allows_slot_type_change=True,
        )
    )

    request_hypothesis = "Music|PlayMusicIntent|ArtistName:kane|MediaType:songs"
    rewrite_hypothesis = "Music|PlayMusicIntent|ArtistName:cain|MediaType:songs"
    print(
        has_slot_change_or_drop(
            query_hypothesis=request_hypothesis,
            rewrite_hypothesis=rewrite_hypothesis,
            sanitization_config=[StringSanitizationOptions.MAKE_FUZZY],
            allows_slot_type_change=True
        )
    )

    request_hypothesis = "Music|PlayMusicIntent|ArtistName:heart broken"
    rewrite_hypothesis = "Music|PlayMusicIntent|ArtistName:iamjakehill|SongName:brokenheart"
    print(
        has_slot_change_or_drop(
            query_hypothesis=request_hypothesis,
            rewrite_hypothesis=rewrite_hypothesis,
            allows_slot_type_change=True
        )
    )

    request_hypothesis = "Music|PlayMusicIntent|ArtistName:daddy king"
    rewrite_hypothesis = "Music|PlayMusicIntent|ArtistName:iamjakehill|SongName:die a king"
    print(
        has_slot_change_or_drop(
            query_hypothesis=request_hypothesis,
            rewrite_hypothesis=rewrite_hypothesis,
            allows_slot_type_change=True
        )
    )

    request_hypothesis = "Music|PlayMusicIntent|ArtistName:rise roots|SongName:part of my life"
    rewrite_hypothesis = "Music|PlayMusicIntent|ArtistName:rise of my life|SongName:rise roots"
    print(
        has_slot_change_or_drop(
            query_hypothesis=request_hypothesis,
            rewrite_hypothesis=rewrite_hypothesis,
            allows_slot_type_change=True
        )
    )

    request_hypothesis = "Video|PlayVideoIntent|AppName:youtube|VideoName:eight ball"
    rewrite_hypothesis = "Video|PlayVideoIntent|AppName:youtube|ArtistName:eight ball|VideoName:know me"
    assert has_slot_change_or_drop(
        query_hypothesis=request_hypothesis,
        rewrite_hypothesis=rewrite_hypothesis,
        rewrite_utterance="play know me by eight ball and on youtube",
        sanitization_config=[
            StringSanitizationOptions.REMOVE_COMMON_PREFIX,
            StringSanitizationOptions.REMOVE_COMMON_SUFFIX
        ],
    ) == 0.0
    request_hypothesis = "Music|PlayMusicIntent|ArtistName:n. l. e. choppa reverb|SongName:capo"
    rewrite_hypothesis = "Music|PlayMusicIntent|ArtistName:nle choppa|SongName:capo"
    assert has_slot_change_or_drop(
        query_hypothesis=request_hypothesis,
        rewrite_hypothesis=rewrite_hypothesis,
        rewrite_utterance="play capo by nle choppa",
        sanitization_config=[
            StringSanitizationOptions.REMOVE_COMMON_PREFIX,
            StringSanitizationOptions.REMOVE_COMMON_SUFFIX
        ],
    ) == 0.0
    request_hypothesis = "HomeAutomation|SetValueIntent|ActionTrigger:turn|DeviceName:telva's|DeviceType:lamp|SettingValue:rainbow"
    rewrite_hypothesis = "HomeAutomation|SetValueIntent|ActionTrigger:turn|DeviceName:telva's lamp|SettingValue:purple"
    assert has_slot_change_or_drop(
        query_hypothesis=request_hypothesis,
        rewrite_hypothesis=rewrite_hypothesis,
        rewrite_utterance="turn telva's lamp to purple",
        sanitization_config=[
            StringSanitizationOptions.REMOVE_COMMON_PREFIX,
            StringSanitizationOptions.REMOVE_COMMON_SUFFIX
        ],
    ) == 0.0
    request_hypothesis = 'Music|PlayMusicIntent|MediaType:songs|SongName:sunlight|ArtistName:luke|MediaType:songs'
    rewrite_hypothesis = 'Music|PlayMusicIntent|SongName:moonlight|ArtistName:luke combs|MediaType:songs'
    assert has_slot_change_or_drop(
        query_hypothesis=request_hypothesis,
        rewrite_hypothesis=rewrite_hypothesis
    )
    assert has_slot_change_or_drop(
        query_hypothesis=request_hypothesis,
        rewrite_hypothesis=rewrite_hypothesis,
        sanitization_config=[
            StringSanitizationOptions.REMOVE_COMMON_PREFIX,
            StringSanitizationOptions.REMOVE_COMMON_SUFFIX
        ],
    ) < 0.8
    request_hypothesis = 'Music|PlayMusicIntent|ArtistName:luke|MediaType:songs'
    rewrite_hypothesis = 'Music|PlayMusicIntent|ArtistName:luke combs|MediaType:songs'
    assert not has_slot_change_or_drop(
        query_hypothesis=request_hypothesis,
        rewrite_hypothesis=rewrite_hypothesis
    )
    assert has_slot_change_or_drop(
        query_hypothesis=request_hypothesis,
        rewrite_hypothesis=rewrite_hypothesis,
        sanitization_config=[
            StringSanitizationOptions.REMOVE_COMMON_PREFIX,
            StringSanitizationOptions.REMOVE_COMMON_SUFFIX
        ],
    ) == 0.0
    request_hypothesis = 'Music|PlayMusicIntent|ArtistName:luke|MediaType:songs'
    rewrite_hypothesis = 'Music|PlayMusicIntent|ArtistName:combs luke|MediaType:songs'
    assert not has_slot_change_or_drop(
        query_hypothesis=request_hypothesis,
        rewrite_hypothesis=rewrite_hypothesis
    )
    assert has_slot_change_or_drop(
        query_hypothesis=request_hypothesis,
        rewrite_hypothesis=rewrite_hypothesis,
        sanitization_config=[
            StringSanitizationOptions.REMOVE_COMMON_PREFIX,
            StringSanitizationOptions.REMOVE_COMMON_SUFFIX
        ],
    ) == 0.0
    request_hypothesis = 'Music|PlayMusicIntent|ArtistName:luke bryan|MediaType:songs'
    rewrite_hypothesis = 'Music|PlayMusicIntent|ArtistName:luke combs|MediaType:songs'
    assert has_slot_change_or_drop(
        query_hypothesis=request_hypothesis,
        rewrite_hypothesis=rewrite_hypothesis
    )
    assert has_slot_change_or_drop(
        query_hypothesis=request_hypothesis,
        rewrite_hypothesis=rewrite_hypothesis,
        sanitization_config=[
            StringSanitizationOptions.REMOVE_COMMON_PREFIX,
            StringSanitizationOptions.REMOVE_COMMON_SUFFIX
        ],
    ) == 1.0
    request_hypothesis = 'Music|PlayMusicIntent|ArtistName:kodak black|SongName:by myself'
    rewrite_hypothesis = 'Music|PlayMusicIntent|ArtistName:kodak black|SongName:like dat'
    assert has_slot_change_or_drop(
        query_hypothesis=request_hypothesis,
        rewrite_hypothesis=rewrite_hypothesis
    )
    assert has_slot_change_or_drop(
        query_hypothesis=request_hypothesis,
        rewrite_hypothesis=rewrite_hypothesis,
        sanitization_config=[
            StringSanitizationOptions.REMOVE_COMMON_PREFIX,
            StringSanitizationOptions.REMOVE_COMMON_SUFFIX
        ],
    ) > 0.8
    request_hypothesis = 'Music|PlayMusicIntent|ArtistName:atlantic starr|SongName:sin for me'
    rewrite_hypothesis = 'Music|PlayMusicIntent|ArtistName:atlantic starr|SongName:send for me'
    assert has_slot_change_or_drop(
        query_hypothesis=request_hypothesis,
        rewrite_hypothesis=rewrite_hypothesis
    )
    assert has_slot_change_or_drop(
        query_hypothesis=request_hypothesis,
        rewrite_hypothesis=rewrite_hypothesis,
        sanitization_config=[
            StringSanitizationOptions.REMOVE_COMMON_PREFIX,
            StringSanitizationOptions.REMOVE_COMMON_SUFFIX
        ],
    ) == 0.5
    request_hypothesis = 'HomeAutomation|TurnOffApplianceIntent|DeviceName:laura|DeviceType:light'
    rewrite_hypothesis = 'HomeAutomation|TurnOffApplianceIntent|DeviceName:scott|DeviceType:light'
    assert has_slot_change_or_drop(
        query_hypothesis=request_hypothesis,
        rewrite_hypothesis=rewrite_hypothesis
    )
    assert has_slot_change_or_drop(
        query_hypothesis=request_hypothesis,
        rewrite_hypothesis=rewrite_hypothesis,
        sanitization_config=[
            StringSanitizationOptions.REMOVE_COMMON_PREFIX,
            StringSanitizationOptions.REMOVE_COMMON_SUFFIX
        ],
    ) == 1.0
    request_hypothesis = 'HomeAutomation|DisplayVideoFeedIntent|ContentSourceDeviceLocation:pool|ContentSourceDeviceType:camera|VisualModeTrigger:show'
    rewrite_hypothesis = 'HomeAutomation|DisplayVideoFeedIntent|ContentSourceDeviceLocation:backyard|ContentSourceDeviceType:camera|VisualModeTrigger:show'
    assert has_slot_change_or_drop(
        query_hypothesis=request_hypothesis,
        rewrite_hypothesis=rewrite_hypothesis
    )
    assert has_slot_change_or_drop(
        query_hypothesis=request_hypothesis,
        rewrite_hypothesis=rewrite_hypothesis,
        sanitization_config=[
            StringSanitizationOptions.REMOVE_COMMON_PREFIX,
            StringSanitizationOptions.REMOVE_COMMON_SUFFIX
        ],
    ) == 1.0
    request_hypothesis = 'HomeAutomation|DisplayVideoFeedIntent|ContentSourceDeviceLocation:backyard pool|ContentSourceDeviceType:camera|VisualModeTrigger:show'
    rewrite_hypothesis = 'HomeAutomation|DisplayVideoFeedIntent|ContentSourceDeviceLocation:pool backyard|ContentSourceDeviceType:camera|VisualModeTrigger:show'
    assert not has_slot_change_or_drop(
        query_hypothesis=request_hypothesis,
        rewrite_hypothesis=rewrite_hypothesis
    )
    assert has_slot_change_or_drop(
        query_hypothesis=request_hypothesis,
        rewrite_hypothesis=rewrite_hypothesis,
        sanitization_config=[
            StringSanitizationOptions.REMOVE_COMMON_PREFIX,
            StringSanitizationOptions.REMOVE_COMMON_SUFFIX
        ],
    ) == 0.0
