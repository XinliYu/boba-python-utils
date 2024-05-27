import warnings
from collections import OrderedDict
from collections import defaultdict
from functools import partial
from itertools import product
from typing import Callable, Iterable, List, Union, Tuple, Optional, Mapping

import pyspark.sql.functions as F

from boba_python_utils.general_utils.algorithms.pair_alignment import iter_best_scored_pairs_of_distinct_source
from boba_python_utils.general_utils.nlp_utility.metrics.edit_distance import EditDistanceOptions, edit_distance
from boba_python_utils.general_utils.nlp_utility.string_sanitization import StringSanitizationOptions, StringSanitizationConfig
from boba_python_utils.production_utils._nlu.target_slots import is_for_comparison_slot_type, is_target_slot_type

try:
    from Levenshtein import ratio
except:
    warnings.warn("unable to load module 'Levenshtein'")
from pyspark.sql.types import ArrayType, StructType, StructField, StringType, BooleanType, FloatType

from boba_python_utils.general_utils.general import compose
from boba_python_utils.general_utils.strex import (
    remove_all_prefix_tokens,
    remove_all_suffix_tokens,
    get_processed_strings_and_map,
    split_,
)
from boba_python_utils.string_utils.regex import contains_as_whole_word, remove_by_whole_word
from boba_python_utils.general_utils.nlp_utility.punctuations import remove_acronym_periods_and_spaces
from boba_python_utils.production_utils._nlu.hypothesis_parsing import (
    get_slots_dict_from_hypothesis, get_hypothesis_splits, get_slots_dict_from_hypothesis_splits,
)
from boba_python_utils.production_utils._nlu.non_overlap import (
    get_utterance_non_overlap,
)
from boba_python_utils.production_utils.greenwich_data.constants import (
    KEY_ENTITY_FIRST,
    KEY_ENTITY_SECOND,
)

ENTITY_STOP_PREFIX_TOKENS = ('a', 'the', 'some', 'my', 'this', 'that')
ENTITY_STOP_SUFFIX_TOKENS = ('radio', 'playlist', 'music', 'song', 'songs')
ENTITY_STOP_TOKENS = (
    *ENTITY_STOP_PREFIX_TOKENS,
    *ENTITY_STOP_SUFFIX_TOKENS,
    'by',
    'in',
    'on',
    'to',
    'as',
    'of',
    'and',
    'or',
    'yes',
    'no',
    'oh',
    'is',
)
UTTERANCE_REPLACES = (
    ("where's", "where is"),
    ("what's", "what is"),
    ("who's", "who is"),
    ("how's", "how is"),
)
UTTERANCE_STOP_PREFIXES = (
    'play',
    'turn',
    'put',
    "stop",
    "alexa",
    "do you",
    "can you",
    "what is",
    "where is",
    "who is",
)

# region functions for text clean up
_clean_entity_value = compose(
    remove_acronym_periods_and_spaces,
    partial(remove_all_prefix_tokens, prefixes=ENTITY_STOP_PREFIX_TOKENS),
    partial(remove_all_suffix_tokens, suffixes=ENTITY_STOP_SUFFIX_TOKENS),
)


def _entity_value_filter(x):
    return x not in ENTITY_STOP_TOKENS


def _get_cleaned_entity_values(hyp, slot_type_filter):
    """
    Gets cleaned entity values. An entity is a slot with its type passing the `slot_type_filter`.

    We removes prefixes in `ENTITY_STOP_PREFIX_TOKENS`,
    and replaces any acronyms like 'c. n. n.' or 'c.n.n' by just 'cnn'.
    """
    slots = get_slots_dict_from_hypothesis(hyp)
    if slots:
        return get_processed_strings_and_map(
            strs=(
                slots.values()
                if slot_type_filter is None
                else (
                    slot_value
                    for slot_type, slot_value in slots.items()
                    if slot_type_filter(slot_type)
                )
            ),
            str_proc=_clean_entity_value,
            filter=_entity_value_filter,
        )
    return None, None


def _clean_utt_and_ref_utt(utt, ref_utt):
    for src, trg in UTTERANCE_REPLACES:
        utt = utt.replace(src, trg)
        ref_utt = ref_utt.replace(src, trg)

    utt = remove_all_prefix_tokens(utt, UTTERANCE_STOP_PREFIXES)
    ref_utt = remove_all_prefix_tokens(ref_utt, UTTERANCE_STOP_PREFIXES)
    return utt, ref_utt


# endregion

# region functions handling the case when a reference entity value is inside a potential request_second entity value  # noqa: E501
def _get_string_nesting_info(str_list1, str_list2):
    identity = set()  # saves identical strings from `str_list1` and `str_list2`
    str_list1_inside_str_list2 = defaultdict(
        list
    )  # saves cases where a string of `str_list1` is inside a string from `str_list2`
    str_list2_inside_str_list1 = defaultdict(
        list
    )  # saves cases where a string of `str_list2` is inside a string from `str_list1`

    for i, j in product(range(len(str_list1)), range(len(str_list2))):
        slot_value1 = str_list1[i]
        slot_value2 = str_list2[j]
        if slot_value1 == slot_value2:
            identity.add(slot_value1)
        elif len(slot_value2) > 1 and slot_value2 in slot_value1:
            str_list2_inside_str_list1[j].append(i)
        elif len(slot_value1) > 1 and slot_value1 in slot_value2:
            str_list1_inside_str_list2[i].append(j)
    return identity, str_list1_inside_str_list2, str_list2_inside_str_list1


def _get_shortest_str_nest(str_nest, str_list1, str_list2, output):
    for j, i in str_nest.items():  # `str_list2[j]` is inside `str_list1[i]`
        i = min(i, key=lambda _i: len(str_list1[_i]))
        str1 = str_list1[i]
        str2 = str_list2[j]
        if str2 not in output:
            # if `str1` indeed contains `str2` as a whole word, then
            _str1 = remove_by_whole_word(str1, str2, include_space_after=True).strip()
            if str1 != _str1:
                output[str2] = str2
                str_list1[i] = _str1


# endregion

# region functions for cases when an reference entity is the same as the whole reference utterance


def _compare_token(token1, token2):
    return token1 == token2 or (
            len(token1) == 2  # noqa: E126
            and len(token2) == 2
            and token1[-1] == '.'
            and token2[-1] == '.'
    )


def _same_begin_end_token(turn1, turn2):
    tokens1 = turn1.split()
    tokens2 = turn2.split()
    if len(tokens1) == len(tokens2):
        return _compare_token(tokens1[0], tokens2[0]) or _compare_token(tokens1[-1], tokens2[-1])
    else:
        tokens1 = [token for token in tokens1 if token not in ENTITY_STOP_TOKENS]
        tokens2 = [token for token in tokens2 if token not in ENTITY_STOP_TOKENS]

        if len(tokens1) == len(tokens2):
            return (
                    len(tokens1) == 0  # noqa: E126
                    or _compare_token(tokens1[0], tokens2[0])
                    or _compare_token(tokens1[-1], tokens2[-1])
            )
        else:
            return (
                    len(tokens1) != 0  # noqa: E126
                    and len(tokens2) != 0
                    and _compare_token(tokens1[0], tokens2[0])
                    and _compare_token(tokens1[-1], tokens2[-1])
            )


# endregion

# region functions for processing reference utterance & hypothesis
def _process_ref_hyp(
        source_utterance,
        target_utterance,
        target_hypothesis,
        slot_type_filter
):
    """
    Processes the reference hypothesis and extracts information.
    """
    entity_pairs = None
    ref_slots = get_slots_dict_from_hypothesis(target_hypothesis)
    if ref_slots:
        ref_cleaned_entity_value_to_original_slot_value_map = {}
        ref_template = target_utterance
        ref_cleaned_entity_values = []
        ref_entity_value_tokens = set()
        ref_non_entity_slot_values = set()

        # the following process entity values
        #     (i.e. those slot values passing the `slot_type_filter`) from the `ref_utt`,
        # so that we can extract a template
        for slot_type, slot_value in ref_slots.items():
            if slot_type_filter is None or slot_type_filter(slot_type):
                # region corner case 1: one slot value in the reference hypothesis
                #     is identical to the reference utterance itself;
                # there are many such cases in Video, Shopping domains,
                #     like customer simply says "squid game"
                # in this case we may directly create a mapping between
                #     the request_second utterance and the reference utterance
                if slot_value == target_utterance and _same_begin_end_token(source_utterance, target_utterance):
                    entity_pairs = {source_utterance: target_utterance}
                    break
                # endregion
                slot_value = remove_all_suffix_tokens(
                    remove_all_prefix_tokens(slot_value, ENTITY_STOP_PREFIX_TOKENS),
                    ENTITY_STOP_SUFFIX_TOKENS,
                )
                if slot_value not in ENTITY_STOP_TOKENS:
                    ref_template = remove_by_whole_word(ref_template, slot_value)
                    entity_cleaned = remove_acronym_periods_and_spaces(slot_value)
                    ref_cleaned_entity_values.append(entity_cleaned)
                    if entity_cleaned != slot_value:
                        ref_cleaned_entity_value_to_original_slot_value_map[
                            entity_cleaned
                        ] = slot_value
                    ref_entity_value_tokens.update(slot_value.split())
            else:
                ref_non_entity_slot_values.add(slot_value)
        return (
            ref_template,  # the template extracted from the reference utterance
            ref_cleaned_entity_values,
            # the cleaned slot values
            #     (with `ENTITY_STOP_PREFIX_TOKENS` removed if they are prefixes)
            ref_entity_value_tokens,  # contains all distinct tokens in the entities
            ref_non_entity_slot_values,
            # non-entity slot values (those slot types not passing the `slot_type_filter`)
            ref_cleaned_entity_value_to_original_slot_value_map,
            # a "cleaned entity" to the original slot value mapping
            entity_pairs,
            # (above corner case 1) we can already establish entity pairs if "corner case 1" holds
        )
    else:
        return None, None, None, None, None, None


def _get_ref_template_tokens(
        ref_template, ref_utt, ref_non_entity_slot_values, ref_entity_value_tokens
):
    """
    Gets reference template tokens from the reference template.
    """
    ref_template_tokens = None
    ref_template = ref_template.strip()
    if ref_template and ref_template != ref_utt:
        # when we do have a non-empty "template", and the template is not the the same as `ref_utt`
        # then split the template into template tokens
        ref_template_tokens = split_(
            ref_template, remove_empty_split=True, lstrip=True, rstrip=True
        )
    elif ref_non_entity_slot_values:
        # otherwise, use non-entity slot values as the template tokens
        ref_template_tokens = list(ref_non_entity_slot_values)

    if ref_template_tokens:
        # removes any reference tokens that overlap with entity value tokens
        ref_template_tokens = list(
            filter(lambda x: x not in ref_entity_value_tokens, ref_template_tokens)
        )
    return ref_template_tokens


# endregion

# region functions for matching entities
def _pair_entity_values(entity_values1, entity_values2):
    """
    Pairs two lists of entity values.
    """
    # region STEP1: naive entity mapping when both only contains a single entity
    if len(entity_values1) == 1 and len(entity_values2) == 1:
        return {entity_values1[0]: entity_values2[0]}
    # endregion

    # region STEP2: handles nested entity value nesting cases.
    # 1. if one entity from `entity_values1` is the same as `entity_values2`, we should pair them
    # 2. if one entity is "we fire tv" ('we' is noise), and the other entity is "fire tv",
    # we want to identify "fire tv" is an actual entity for `entity_values1`
    (
        identical_entity_values,
        entity_values1_inside_entity_values2,
        entity_values2_inside_entity_values1,
    ) = _get_string_nesting_info(entity_values1, entity_values2)
    used_turn2_slot_value = identical_entity_values

    # for above case 1
    paired_entity_values = OrderedDict()
    for slot_value in identical_entity_values:
        if slot_value not in ENTITY_STOP_TOKENS:
            paired_entity_values[slot_value] = slot_value
    # for above case 2
    _get_shortest_str_nest(
        entity_values2_inside_entity_values1, entity_values1, entity_values2, paired_entity_values
    )
    _get_shortest_str_nest(
        entity_values1_inside_entity_values2, entity_values2, entity_values1, paired_entity_values
    )

    # get remaining entities
    entity_values1_remaining = list(filter(lambda x: x not in paired_entity_values, entity_values1))
    entity_values2_remaining = list(filter(lambda x: x not in paired_entity_values, entity_values2))
    # endregion

    # region STEP3: match remaining pairs by similarity
    # We rank all possible entity pairs by edit distance ratio, highest to lowest.
    # Then we take the entity pairs sequentially, until all entities in `entity_values2` get paired.
    _count = 0
    for entity_value1, entity_value2 in sorted(
            product(entity_values1_remaining, entity_values2_remaining),
            key=lambda x: ratio(x[0], x[1]),
            reverse=True,
    ):
        if (
                entity_value1 not in ENTITY_STOP_TOKENS
                and entity_value1 not in paired_entity_values  # `entity_value1` not a stop token
                and entity_value2  # `entity_value1` is not already paired
                not in used_turn2_slot_value  # `entity_value2` is not already paired
        ):
            entity_value1 = remove_all_suffix_tokens(
                remove_all_prefix_tokens(entity_value1, ENTITY_STOP_PREFIX_TOKENS),
                ENTITY_STOP_SUFFIX_TOKENS,
            )
            paired_entity_values[entity_value1] = entity_value2
            used_turn2_slot_value.add(entity_value2)
            _count += 1
            if _count == len(entity_values2_remaining):
                break
    # endregion
    return paired_entity_values


# endregion


def extract_slot_pairs_using_reference_utterance_and_hypothesis(
        source_utterance: str,
        reference_utterance: str,
        reference_hypothesis: str,
        source_hypothesis: str = None,
        slot_type_filter: Callable[[str], bool] = None,
        return_original_entity_values=True,
        requires_entity_value_in_utterance=False,
):
    """
    Extracts one-one slot value mapping between slot values in `source_utterance`
        and slot values in `reference_utterance` with slot information provided
        by `reference_hypothesis` or/and `label_hypothesis`.
    The `label_hypothesis` is optional,
        which is the True hypothesis of `source_utterance`. When `label_hypothesis` is

    Args:
        source_utterance: the source utterance we would like to extract entities from;
            we then create  mapping between this `utterance`
        reference_utterance:
        reference_hypothesis:
        source_hypothesis:
        slot_type_filter:

    Returns:

    """
    if not source_hypothesis:
        # request_second entity pair extraction without label hypothesis
        return _extract_slot_pairs_without_source_hypothesis(
            utt=source_utterance,
            ref_utt=reference_utterance,
            ref_hyp_for_template_extraction=reference_hypothesis,
            slot_type_filter=slot_type_filter,
            return_original_entity_values=return_original_entity_values,
            requires_entity_value_in_utterance=requires_entity_value_in_utterance,
        )
    else:
        # if a label hypothesis is not provided, also send this information
        #     to assist request_second entity extraction
        return _extract_slot_pairs_without_source_hypothesis(
            utt=source_utterance,
            ref_utt=reference_utterance,
            ref_hyp_for_template_extraction=reference_hypothesis,
            ref_hyp_for_pairing=source_hypothesis,
            ref_hyp_for_pairing_is_utt_hyp=True,
            slot_type_filter=slot_type_filter,
            return_original_entity_values=return_original_entity_values,
            requires_entity_value_in_utterance=requires_entity_value_in_utterance,
        )


def _entity_matches_multiple_slot_values(entity, slot_values):
    for slot_value in slot_values:
        entity = remove_by_whole_word(entity, slot_value)
    return not entity.strip()


def is_good_entity_extraction(
        utterance,
        reference_utterance,
        reference_hypothesis,
        label_hypothesis,
        exact_match=False,
        slot_type_filter=None,
        label_hypothesis_visible_when_extracting_entities=False,
):
    if label_hypothesis_visible_when_extracting_entities:
        entity_pairs = extract_slot_pairs_using_reference_utterance_and_hypothesis(
            source_utterance=utterance,
            reference_utterance=reference_utterance,
            reference_hypothesis=reference_hypothesis,
            source_hypothesis=label_hypothesis,
            slot_type_filter=slot_type_filter,
            return_original_entity_values=True,
            requires_entity_value_in_utterance=False,
        )
    else:
        entity_pairs = extract_slot_pairs_using_reference_utterance_and_hypothesis(
            source_utterance=utterance,
            reference_utterance=reference_utterance,
            reference_hypothesis=reference_hypothesis,
            slot_type_filter=slot_type_filter,
            return_original_entity_values=True,
            requires_entity_value_in_utterance=False,
        )
    label_hypothesis_slots = get_slots_dict_from_hypothesis(label_hypothesis)

    if label_hypothesis_slots:
        label_hypothesis_slots_values = set(label_hypothesis_slots.values())

        entity_pairs2 = _extract_slot_pairs_without_source_hypothesis(
            utt=utterance,
            ref_utt=reference_utterance,
            ref_hyp_for_template_extraction=reference_hypothesis,
            ref_hyp_for_pairing=label_hypothesis,
            ref_hyp_for_pairing_is_utt_hyp=False,
            slot_type_filter=slot_type_filter,
            return_original_entity_values=True,
            requires_entity_value_in_utterance=False,
        )

        if exact_match:
            return all(
                (entity in label_hypothesis_slots_values)
                or (
                        entity in entity_pairs2
                        and (entity_pairs2[entity] in label_hypothesis_slots_values)  # noqa: E126
                )
                for entity in entity_pairs.keys()
            )
        else:
            return all(
                (
                        entity in label_hypothesis_slots_values
                        or
                        # the entity might be split into multiple NLU slots in the reference
                        _entity_matches_multiple_slot_values(entity, label_hypothesis_slots_values)
                )
                or (
                        entity in entity_pairs2
                        and (
                                (entity_pairs2[entity] in label_hypothesis_slots_values)
                                or _entity_matches_multiple_slot_values(
                            entity_pairs2[entity], label_hypothesis_slots_values
                        )
                            # noqa: E501
                        )
                )
                for entity in entity_pairs.keys()
            )


def _extract_slot_pairs_using_reference_utterance_and_hypothesis_udf(
        utterance: str,
        reference_utterance: str,
        reference_hypothesis: str,
        hypothesis: str = None,
        slot_type_filter: Callable[[str], bool] = None,
        return_original_entity_values: bool = True,
        requires_entity_value_in_utterance: bool = False,
):
    out = []
    for entity1, entity2 in extract_slot_pairs_using_reference_utterance_and_hypothesis(
            source_utterance=utterance,
            reference_utterance=reference_utterance,
            reference_hypothesis=reference_hypothesis,
            source_hypothesis=hypothesis,
            slot_type_filter=slot_type_filter,
            return_original_entity_values=return_original_entity_values,
            requires_entity_value_in_utterance=requires_entity_value_in_utterance,
    ).items():
        out.append({KEY_ENTITY_FIRST: entity1, KEY_ENTITY_SECOND: entity2})
    return None if not out else out


def extract_slot_pairs_using_reference_utterance_and_hypothesis_udf(
        utterance,
        reference_utterance,
        reference_hypothesis,
        hypothesis=None,
        slot_type_filter: Callable[[str], bool] = None,
        return_original_entity_values: bool = True,
        requires_entity_value_in_utterance: bool = False,
):
    return_type = ArrayType(
        StructType(
            [
                StructField(name=KEY_ENTITY_FIRST, dataType=StringType()),
                StructField(name=KEY_ENTITY_SECOND, dataType=StringType()),
            ]
        )
    )
    if hypothesis is None:
        return F.udf(
            partial(
                _extract_slot_pairs_using_reference_utterance_and_hypothesis_udf,
                hypothesis=None,
                slot_type_filter=slot_type_filter,
                return_original_entity_values=return_original_entity_values,
                requires_entity_value_in_utterance=requires_entity_value_in_utterance,
            ),
            returnType=return_type,
        )(utterance, reference_utterance, reference_hypothesis)
    else:
        return F.udf(
            partial(
                _extract_slot_pairs_using_reference_utterance_and_hypothesis_udf,
                slot_type_filter=slot_type_filter,
                return_original_entity_values=return_original_entity_values,
                requires_entity_value_in_utterance=requires_entity_value_in_utterance,
            ),
            returnType=return_type,
        )(utterance, reference_utterance, reference_hypothesis, hypothesis)


def is_good_entity_extraction_udf(
        utterance,
        reference_utterance,
        reference_hypothesis,
        label_hypothesis,
        exact_match=False,
        slot_type_filter=None,
        label_hypothesis_visible_when_extracting_entities=False,
):
    return F.udf(
        partial(
            is_good_entity_extraction,
            slot_type_filter=slot_type_filter,
            exact_match=exact_match,
            label_hypothesis_visible_when_extracting_entities=label_hypothesis_visible_when_extracting_entities,  # noqa: E501
        ),
        returnType=BooleanType(),
    )(utterance, reference_utterance, reference_hypothesis, label_hypothesis)


# endregion

def _extract_slot_pairs_from_hypothesis_single_slot_corner_case(slots: Mapping, utterance: str):
    if (
            utterance is not None and
            slots is not None and
            len(slots) == 1 and
            (next(iter(slots.values())) == utterance)
    ):
        target_slot_type, target_slot_value = next(iter(slots.items()))
        return [
            (1.0, target_slot_type, target_slot_value, target_slot_type, target_slot_value)
        ]
    else:
        return []


def extract_slot_pairs_from_hypothesis(
        source_hypothesis: Union[str, List[str], Tuple[str, ...]],
        target_hypothesis: Union[str, List[str], Tuple[str, ...]],
        source_utterance: str = None,
        target_utterance: str = None,
        slot_type_filter: Optional[Union[Callable, Iterable[str]]] = is_for_comparison_slot_type,
        return_align_score: bool = True,
        # region edit distance config
        sanitization_config: Union[Iterable[StringSanitizationOptions], StringSanitizationConfig] = None,
        edit_distance_consider_sorted_tokens: Union[bool, Callable] = min,
        edit_distance_options: EditDistanceOptions = None,
        **edit_distance_kwargs,
        # endregion
) -> List[Union[Tuple[float, str, str, str, str], Tuple[str, str, str, str]]]:
    if isinstance(source_hypothesis, str):
        source_hypothesis = get_hypothesis_splits(source_hypothesis)

    if isinstance(target_hypothesis, str):
        target_hypothesis = get_hypothesis_splits(target_hypothesis)

    source_slots, target_slots = (
        get_slots_dict_from_hypothesis_splits(source_hypothesis),
        get_slots_dict_from_hypothesis_splits(target_hypothesis)
    )

    if not source_slots:
        return _extract_slot_pairs_from_hypothesis_single_slot_corner_case(
            slots=target_slots,
            utterance=source_utterance
        )

    if not target_slots:
        return _extract_slot_pairs_from_hypothesis_single_slot_corner_case(
            slots=source_slots,
            utterance=target_utterance
        )

    if slot_type_filter is not None:
        if not callable(slot_type_filter):
            slot_type_filter = partial(is_target_slot_type, target_slot_types=slot_type_filter)
        source_slots = {
            slot_type: source_slots[slot_type]
            for slot_type in filter(slot_type_filter, source_slots)
        }
        target_slots = {
            slot_type: target_slots[slot_type]
            for slot_type in filter(slot_type_filter, target_slots)
        }
        if (not source_slots) or (not target_slots):
            return []

    source_target_slot_pairs = [
        (
            edit_distance(
                source_slot[1],
                target_slot[1],
                consider_sorted_tokens=edit_distance_consider_sorted_tokens,
                sanitization_config=sanitization_config,
                consider_same_num_tokens=False,
                return_ratio=True,
                options=edit_distance_options,
                **edit_distance_kwargs
            ),
            source_slot, target_slot
        )
        for source_slot, target_slot
        in product(
            source_slots.items(), target_slots.items()
        )
    ]

    return [
        (x[0], *x[1], *x[2]) if return_align_score else (*x[1], *x[2]) for x in
        iter_best_scored_pairs_of_distinct_source(
            source_target_slot_pairs,
            reversed_sort_by_scores=True,
            identity_score=1.0
        )
    ]


def extract_slot_pairs_from_hypothesis_udf(
        source_hypothesis: Union[str, List[str], Tuple[str, ...]],
        target_hypothesis: Union[str, List[str], Tuple[str, ...]],
        source_slot_type_colname: str,
        source_slot_value_colname: str,
        target_slot_type_colname: str,
        target_slot_value_colname: str,
        slot_align_score_colname: str = None,
        source_utterance: str = None,
        target_utterance: str = None,
        slot_type_filter: Optional[Union[Callable, Iterable[str]]] = is_for_comparison_slot_type,
        # region edit distance config
        sanitization_config: Union[Iterable[StringSanitizationOptions], StringSanitizationConfig] = None,
        edit_distance_consider_sorted_tokens: Union[bool, Callable] = min,
        edit_distance_options: EditDistanceOptions = None,
        **edit_distance_kwargs,
        # endregion
):
    return_align_score = bool(slot_align_score_colname)
    return_fields = [
        StructField(name=source_slot_type_colname, dataType=StringType()),
        StructField(name=source_slot_value_colname, dataType=StringType()),
        StructField(name=target_slot_type_colname, dataType=StringType()),
        StructField(name=target_slot_value_colname, dataType=StringType()),
    ]
    if return_align_score:
        return_fields = [
            StructField(name=slot_align_score_colname, dataType=FloatType()),
            *return_fields
        ]

    return F.udf(
        partial(
            extract_slot_pairs_from_hypothesis,
            slot_type_filter=slot_type_filter,
            return_align_score=return_align_score,
            sanitization_config=sanitization_config,
            edit_distance_consider_sorted_tokens=edit_distance_consider_sorted_tokens,
            edit_distance_options=edit_distance_options,
            **edit_distance_kwargs
        ),
        returnType=ArrayType(StructType(return_fields))
    )(
        source_hypothesis,
        target_hypothesis,
        (F.lit(None) if source_utterance is None else source_utterance),
        (F.lit(None) if target_utterance is None else target_utterance)
    )


def _extract_slot_pairs_without_source_hypothesis(
        utt: str,
        ref_utt: str,
        ref_hyp_for_template_extraction: str,
        ref_hyp_for_pairing: str = None,
        ref_hyp_for_pairing_is_utt_hyp=False,
        slot_type_filter: Callable[[str], bool] = None,
        return_original_entity_values=True,
        requires_entity_value_in_utterance=False,
):
    """

    Args:
        utt:
        ref_utt:
        ref_hyp_for_template_extraction:
        ref_hyp_for_pairing:
        ref_hyp_for_pairing_is_utt_hyp:
        slot_type_filter:

    Returns:

    """

    # region STEP1: sanitize utterance strings; e.g. remove common prefixes like "play", "turn on", "turn off" to ease further extraction.  # noqa: E501
    # utt, ref_utt = _clean_utt_and_ref_utt(utt, ref_utt)
    # endregion

    # region STEP2: extracts information from utterance NLU,
    # including reference template, reference entity values, etc.
    (
        ref_template,
        ref_cleaned_entity_values,
        ref_entity_value_tokens,
        ref_non_entity_slot_values,
        ref_cleaned_entity_value_to_original_slot_value_map,
        paired_entity_values,
    ) = _process_ref_hyp(
        source_utterance=utt,
        target_utterance=ref_utt,
        target_hypothesis=ref_hyp_for_template_extraction,
        slot_type_filter=slot_type_filter,
    )
    # endregion

    utt_cleaned_entity_value_to_original_slot_value_map = {}
    if (ref_template is not None) and (
            # `ref_template` is None
            #     when the `ref_hyp_for_template_extraction` does not contain any slots
            paired_entity_values
            is None
    ):  # `entity_pairs` is not None when the above STEP2 already gener
        if ref_hyp_for_pairing_is_utt_hyp:
            # the case when the true hypothesis for `utt` is provided;
            #     passed in as the argument `ref_hyp_for_pairing`
            # in this case we directly compare the slots in the two hypotheses to
            #     get request_second entities
            (
                utt_entity_values,
                utt_cleaned_entity_value_to_original_slot_value_map,
            ) = _get_cleaned_entity_values(ref_hyp_for_pairing, slot_type_filter)
            if utt_entity_values:
                paired_entity_values = _pair_entity_values(
                    utt_entity_values, ref_cleaned_entity_values
                )
        else:
            # the case when the true hypothesis for `utt` is missing
            # then we use the template tokens we found from reference hypothesis to split `utt`
            ref_template_tokens = _get_ref_template_tokens(
                ref_template=ref_template,
                ref_utt=ref_utt,
                ref_non_entity_slot_values=ref_non_entity_slot_values,
                ref_entity_value_tokens=ref_entity_value_tokens,
            )
            if ref_template_tokens:
                # obtains utterance values by splitting the `utt` by the reference template tokens
                utt_entity_values = split_(
                    utt,
                    sep=ref_template_tokens,
                    remove_empty_split=True,
                    lstrip=True,
                    rstrip=True,
                    split_by_whole_words=True,
                    use_space_as_word_boundary=True,
                )
                (
                    utt_entity_values,
                    utt_cleaned_entity_value_to_original_slot_value_map,
                ) = get_processed_strings_and_map(
                    strs=utt_entity_values,
                    str_proc=_clean_entity_value,
                    filter=_entity_value_filter,
                )

                if ref_hyp_for_pairing is not None:
                    # if a `ref_hyp_for_pairing` is provided,
                    #     we instead use it for the entity pairing
                    #     (otherwise we use the results for `ref_hyp_for_template_extraction`)
                    (
                        ref_cleaned_entity_values,
                        ref_cleaned_entity_value_to_original_slot_value_map,
                    ) = _get_cleaned_entity_values(ref_hyp_for_pairing, slot_type_filter)

                # finally get the paired entity values
                if ref_cleaned_entity_values:
                    paired_entity_values = _pair_entity_values(
                        utt_entity_values, ref_cleaned_entity_values
                    )

    paired_entity_values_final = {}
    if paired_entity_values:
        if (
                utt_cleaned_entity_value_to_original_slot_value_map
                or ref_cleaned_entity_value_to_original_slot_value_map
        ):
            for entity1, entity2 in paired_entity_values.items():
                entity1_original = utt_cleaned_entity_value_to_original_slot_value_map.get(
                    entity1, entity1
                )
                entity2_original = ref_cleaned_entity_value_to_original_slot_value_map.get(
                    entity2, entity2
                )
                if (not requires_entity_value_in_utterance) or (
                        (  # noqa: E126
                                contains_as_whole_word(utt, entity1)
                                or contains_as_whole_word(utt, entity1_original)
                        )
                        and (  # noqa: E126
                                contains_as_whole_word(ref_utt, entity2)
                                or contains_as_whole_word(ref_utt, entity2_original)
                        )
                ):
                    if return_original_entity_values:
                        paired_entity_values_final[entity1_original] = entity2_original
                    else:
                        paired_entity_values_final[entity1] = entity2
        elif not requires_entity_value_in_utterance:
            paired_entity_values_final = paired_entity_values
        else:
            paired_entity_values_final = dict(
                filter(lambda x: (x[0] in utt) and (x[1] in ref_utt), paired_entity_values.items())
            )

    if not paired_entity_values_final:
        non_overlap1, non_overlap2 = get_utterance_non_overlap(utt, ref_utt)
        if non_overlap1:
            paired_entity_values_final[non_overlap1] = non_overlap2

    return paired_entity_values_final


if __name__ == '__main__':
    print(
        _extract_slot_pairs_without_source_hypothesis(
            'turn w. p. g. l. on',
            'turn on w. p. g. m.',
            'Music|PlayStationIntent|CallSign:w. p. g. m.|RadioBand:f. m.',
            return_original_entity_values=False,
            requires_entity_value_in_utterance=True,
            slot_type_filter=is_for_comparison_slot_type,
        )
    )

    print(
        _extract_slot_pairs_without_source_hypothesis(
            'turn the radio music on',
            'turn on the rain music',
            'Music|PlayMusicIntent|GenreName:rain|MediaType:music',
            return_original_entity_values=False,
            requires_entity_value_in_utterance=True,
            slot_type_filter=is_for_comparison_slot_type,
        )
    )
    print(
        _extract_slot_pairs_without_source_hypothesis(
            'play mic junior',
            'play mike junior',
            'Music|PlayMusicIntent|ArtistName:michael junior',
            return_original_entity_values=False,
            requires_entity_value_in_utterance=True,
            slot_type_filter=is_for_comparison_slot_type,
        )
    )
    print(
        _extract_slot_pairs_without_source_hypothesis(
            'play the arcade playlist',
            'play mike junior',
            'Music|PlayMusicIntent|MediaType:playlist|PlaylistName:r. k.',
            return_original_entity_values=False,
            requires_entity_value_in_utterance=True,
            slot_type_filter=is_for_comparison_slot_type,
        )
    )
    print(
        _extract_slot_pairs_without_source_hypothesis(
            'play gay music',
            'play day music',
            'Routines|InvokeRoutineIntent|TriggerPhrase:play day music',
            return_original_entity_values=False,
            requires_entity_value_in_utterance=True,
            slot_type_filter=is_for_comparison_slot_type,
        )
    )
    print(
        _extract_slot_pairs_without_source_hypothesis(
            'shuffle me playlist',
            'shuffle my playlist',
            'Music|PlayMusicIntent|ArtistName:caamp|SongName:by and by',
            return_original_entity_values=False,
            requires_entity_value_in_utterance=True,
            slot_type_filter=is_for_comparison_slot_type,
        )
    )
    print(
        _extract_slot_pairs_without_source_hypothesis(
            'add this song to my song list playlist',
            'add this song to my song life playlist',
            'Music|AddToPlaylistIntent|ActiveUserTrigger:'
            + 'my|Anaphor:this|MediaType:song playlist|PlaylistName:song life',
            return_original_entity_values=False,
            requires_entity_value_in_utterance=True,
            slot_type_filter=is_for_comparison_slot_type,
        )
    )

    print(
        extract_slot_pairs_from_hypothesis(
            'Music|PlayStationIntent|StationName:n. p. r.|StationNumber:two',
            'Music|PlayStationIntent|AppName:tunein|StationName:n. p. o. radio two',
            sanitization_config=[
                StringSanitizationOptions.REMOVE_ACRONYMS_PERIODS_AND_SPACES,
                StringSanitizationOptions.REMOVE_PUNCTUATIONS_EXCEPT_FOR_HYPHEN,
                StringSanitizationOptions.MAKE_FUZZY,
                StringSanitizationOptions.REMOVE_SPACES
            ],
            edit_distance_options=EditDistanceOptions(
                weight_distance_if_str1_is_substr=0.5,
                weight_distance_if_str2_is_substr=0.5,
                weight_distance_if_strs_have_common_start=0.8,
                min_str_common_start_to_enable_weight=2
            )
        ) == [
            (0.6363636363636364, 'StationNumber', 'two', 'StationName', 'n. p. o. radio two'),
            (0.4181818181818181, 'StationName', 'n. p. r.', 'StationName', 'n. p. o. radio two')
        ]
    )

    print(
        extract_slot_pairs_from_hypothesis(
            'Music|PlayMusicIntent|SongName:airport noise',
            'GeneralMedia|LaunchNativeAppIntent|AppName:airplane sounds',
            sanitization_config=[
                StringSanitizationOptions.REMOVE_ACRONYMS_PERIODS_AND_SPACES,
                StringSanitizationOptions.REMOVE_PUNCTUATIONS_EXCEPT_FOR_HYPHEN,
                StringSanitizationOptions.MAKE_FUZZY,
                StringSanitizationOptions.REMOVE_SPACES
            ],
            edit_distance_options=EditDistanceOptions(
                weight_distance_if_str1_is_substr=0.5,
                weight_distance_if_str2_is_substr=0.5,
                weight_distance_if_strs_have_common_start=0.8,
                min_str_common_start_to_enable_weight=2
            )
        ) == [(0.5428571428571429, 'SongName', 'airport noise', 'AppName', 'airplane sounds')]
    )

    print(
        extract_slot_pairs_from_hypothesis(
            'Music|PlayMusicIntent|ArtistName:fat|SongName:pop county boogie',
            'Music|PlayMusicIntent|ArtistName:the fade|SongName:pike county boogie',
            sanitization_config=[
                StringSanitizationOptions.REMOVE_ACRONYMS_PERIODS_AND_SPACES,
                StringSanitizationOptions.REMOVE_PUNCTUATIONS_EXCEPT_FOR_HYPHEN,
                StringSanitizationOptions.MAKE_FUZZY,
                StringSanitizationOptions.REMOVE_SPACES
            ],
            edit_distance_options=EditDistanceOptions(
                weight_distance_if_str1_is_substr=0.5,
                weight_distance_if_str2_is_substr=0.5,
                weight_distance_if_strs_have_common_start=0.8,
                min_str_common_start_to_enable_weight=2
            )
        ) == [
            (0.8, 'SongName', 'pop county boogie', 'SongName', 'pike county boogie'),
            (0.75, 'ArtistName', 'fat', 'ArtistName', 'the fade')
        ]
    )

    print(
        extract_slot_pairs_from_hypothesis(
            'Music|PlayStationIntent|AppName:tunein|CallSign:k. f. m.',
            'Music|PlayStationIntent|AppName:tunein|RadioBand:f. m.|StationName:gay',
            sanitization_config=[
                StringSanitizationOptions.REMOVE_ACRONYMS_PERIODS_AND_SPACES,
                StringSanitizationOptions.REMOVE_PUNCTUATIONS_EXCEPT_FOR_HYPHEN,
                StringSanitizationOptions.MAKE_FUZZY,
                StringSanitizationOptions.REMOVE_SPACES
            ],
            edit_distance_options=EditDistanceOptions(
                weight_distance_if_str1_is_substr=0.5,
                weight_distance_if_str2_is_substr=0.5,
                weight_distance_if_strs_have_common_start=0.8,
                min_str_common_start_to_enable_weight=2
            )
        ) == [
            (1.0, 'AppName', 'tunein', 'AppName', 'tunein'),
            (0.33333333333333337, 'CallSign', 'k. f. m.', 'StationName', 'gay')
        ]
    )

    print(
        extract_slot_pairs_from_hypothesis(
            'Music|PlayStationIntent|AppName:pandora|SongName:as we',
            'Music|PlayStationIntent|AppName:pandora|ArtistName:ash b.',
            sanitization_config=[
                StringSanitizationOptions.REMOVE_ACRONYMS_PERIODS_AND_SPACES,
                StringSanitizationOptions.REMOVE_PUNCTUATIONS_EXCEPT_FOR_HYPHEN,
                StringSanitizationOptions.MAKE_FUZZY,
                StringSanitizationOptions.REMOVE_SPACES
            ],
            edit_distance_options=EditDistanceOptions(
                weight_distance_if_str1_is_substr=0.5,
                weight_distance_if_str2_is_substr=0.5,
                weight_distance_if_strs_have_common_start=0.8,
                min_str_common_start_to_enable_weight=2
            )
        ) == [
            (1.0, 'AppName', 'pandora', 'AppName', 'pandora'),
            (0.6, 'SongName', 'as we', 'ArtistName', 'ash b.')
        ]
    )

    print(
        extract_slot_pairs_from_hypothesis(
            'Music|PlayMusicIntent|AlbumName:too fast for love|ArtistName:motley crue',
            'Music|PlayMusicIntent|MediaType:song|SongName:too fast for love',
            sanitization_config=[
                StringSanitizationOptions.REMOVE_ACRONYMS_PERIODS_AND_SPACES,
                StringSanitizationOptions.REMOVE_PUNCTUATIONS_EXCEPT_FOR_HYPHEN,
                StringSanitizationOptions.MAKE_FUZZY,
                StringSanitizationOptions.REMOVE_SPACES
            ],
            edit_distance_options=EditDistanceOptions(
                weight_distance_if_str1_is_substr=0.5,
                weight_distance_if_str2_is_substr=0.5,
                weight_distance_if_strs_have_common_start=0.8,
                min_str_common_start_to_enable_weight=2
            )
        ) == [(1.0, 'AlbumName', 'too fast for love', 'SongName', 'too fast for love')]
    )

    print(
        extract_slot_pairs_from_hypothesis(
            'Music|PlayStationIntent',
            'Music|PlayStationIntent|StationName:n. p. o. radio two',
            source_utterance='n. p. o. radio two',
            sanitization_config=[
                StringSanitizationOptions.REMOVE_ACRONYMS_PERIODS_AND_SPACES,
                StringSanitizationOptions.REMOVE_PUNCTUATIONS_EXCEPT_FOR_HYPHEN,
                StringSanitizationOptions.MAKE_FUZZY,
                StringSanitizationOptions.REMOVE_SPACES
            ],
            edit_distance_options=EditDistanceOptions(
                weight_distance_if_str1_is_substr=0.5,
                weight_distance_if_str2_is_substr=0.5,
                weight_distance_if_strs_have_common_start=0.8,
                min_str_common_start_to_enable_weight=2
            )
        ) == [(1.0, 'StationName', 'n. p. o. radio two', 'StationName', 'n. p. o. radio two')]
    )
