from functools import partial
from itertools import product
from typing import Union, Optional, List, Tuple, Mapping, Iterable, Callable

import pyspark.sql.functions as F
from pyspark.sql import DataFrame, Column
from pyspark.sql.types import ArrayType, StructType, StructField, StringType, BooleanType

from boba_python_utils.common_utils.iter_helper import filter_tuples_by_head_element
from boba_python_utils.production_utils.greenwich_data.constants import (
    KEY_ENTITY_TYPE,
    KEY_ENTITIES,
    KEY_ENTITY,
    KEY_ENTITY_TYPE_FIRST,
    KEY_ENTITY_FIRST,
    KEY_ENTITY_TYPE_SECOND,
    KEY_ENTITY_SECOND,
    DOMAIN_PREFIX_3P,
    DOMAIN_NAME_ALL_3P,
)

SEP_NLU_HYPOTHESIS_PARTS = '|'
SEP_NLU_SLOT_PARTS = ':'


# region domain resolution

def is_third_party_skill_domain(domain_or_hyp: str) -> Optional[bool]:
    """
    Checks if a domain is a 3p domain.
    Args:
        domain_or_hyp: the domain string or a NLU hypothesis starting with a domain string.

    Returns: True if the input is a 3p domain string/hypothesis; otherwise False;
                returns None if `domain_or_hyp` is empty;

    """
    if domain_or_hyp:
        return domain_or_hyp.startswith(DOMAIN_PREFIX_3P)


is_third_party_skill_domain_udf = F.udf(is_third_party_skill_domain, returnType=BooleanType())


def solve_domain(domain: str, third_party_skill_as_one_domain: bool = True) -> str:
    """
    
    Solves all 3p domain string as a single domain string defined by constant `DOMAIN_NAME_ALL_3P`.
    
    When `third_party_skill_as_one_domain` is True and `domain` is a 3p domain,
    then returns `DOMAIN_NAME_ALL_3P` as the domain string;
    otherwise, returns the input `domain` itself.

    """
    if domain and third_party_skill_as_one_domain and is_third_party_skill_domain(domain):
        return DOMAIN_NAME_ALL_3P
    else:
        return domain


def solve_domain_udf(domain, third_party_skill_as_one_domain=True):
    return F.udf(
        partial(solve_domain, third_party_skill_as_one_domain=third_party_skill_as_one_domain)
    )(domain)


# endregion

# region extract domain/intent from NLU hypotehsis

def get_hypothesis_splits(hypothesis: str) -> Optional[List[str]]:
    """
    Gets the splits of an NLU hypothesis string.
    The splits contains the domain, intent and slot strings for the NLU hypothesis string.
    
    Args:
        hypothesis: the NLU hypothesis string.

    Returns: the NLU hypothesis splits containing domain, intent and slot strings;
        returns None if the input `hypothesis` is empty.

    """
    if hypothesis:
        return hypothesis.split(SEP_NLU_HYPOTHESIS_PARTS)


def get_domain_from_hypothesis_splits(
        hypothesis_splits: Union[List[str], Tuple[str, ...]],
        third_party_skill_as_one_domain: bool = True
) -> Optional[str]:
    """
    Gets domain string from the NLU hypothesis splits.
    Args:
        hypothesis_splits: the NLU hypothesis splits.
        third_party_skill_as_one_domain: uses `DOMAIN_NAME_ALL_3P` 
            as the domain string for all 3p domain.

    Returns: the domain string of the NLU hypothesis; 
        returns None if `hypothesis_splits` is empty;

    """
    if hypothesis_splits:
        return solve_domain(
            hypothesis_splits[0], third_party_skill_as_one_domain=third_party_skill_as_one_domain
        )


def get_domain_from_hypothesis(
        hypothesis: str,
        third_party_skill_as_one_domain: bool = True
) -> Optional[str]:
    """
    See `get_domain_from_hypothesis_splits`.
    """
    return get_domain_from_hypothesis_splits(
        get_hypothesis_splits(hypothesis), third_party_skill_as_one_domain
    )


def get_intent_from_hypothesis(
        hypothesis: str
) -> Optional[str]:
    return get_intent_from_hypothesis_splits(get_hypothesis_splits(hypothesis))


def get_domain_from_hypothesis_udf(hypothesis, third_party_skill_as_one_domain=True):
    return F.udf(
        partial(
            get_domain_from_hypothesis, third_party_skill_as_one_domain=third_party_skill_as_one_domain
        )
    )(hypothesis)


def get_intent_from_hypothesis_splits(
        hypothesis_splits: Union[List[str], Tuple[str, ...]]
) -> Optional[str]:
    if hypothesis_splits:
        intent = hypothesis_splits[1] if len(hypothesis_splits) > 1 else None
        return intent
    else:
        return None


def get_domain_intent_from_hypothesis_splits(
        hypothesis_splits: Union[List[str], Tuple[str, ...]],
        third_party_skill_as_one_domain: bool = True
) -> Tuple[Optional[str], Optional[str]]:
    """
    Gets domain and intent string from the NLU hypothesis splits.
    Args:
        hypothesis_splits: the NLU hypothesis splits.
        third_party_skill_as_one_domain: uses `DOMAIN_NAME_ALL_3P` 
            as the domain string for all 3p domain.

    Returns: the domain and intent string of the NLU hypothesis; 
        returns None, None if the input `hypothesis_splits` is empty;
        returns intent string as None if there is only domain in the NLU hypothesis.

    """
    if hypothesis_splits:
        domain = solve_domain(
            hypothesis_splits[0], third_party_skill_as_one_domain=third_party_skill_as_one_domain
        )
        intent = hypothesis_splits[1] if len(hypothesis_splits) > 1 else None
        return domain, intent
    else:
        return None, None


def get_domain_intent_from_hypothesis(
        hypothesis: str,
        third_party_skill_as_one_domain=True
) -> Tuple[Optional[str], Optional[str]]:
    """
    See `get_domain_intent_from_hypothesis_splits`.
    """
    return get_domain_intent_from_hypothesis_splits(
        get_hypothesis_splits(hypothesis), third_party_skill_as_one_domain
    )


# endregion

# region extraction of intent, slots from hypothesis
def hypothesis_has_slots(hypothesis: str) -> bool:
    hypothesis_splits = get_hypothesis_splits(hypothesis)
    return hypothesis_splits and len(hypothesis_splits) > 2


def hypothesis_has_slots_udf(hypothesis):
    return F.col(hypothesis).isNotNull() & (F.size(F.split(hypothesis, rf'\{SEP_NLU_HYPOTHESIS_PARTS}')) > 2)


def get_slots_dict_from_hypothesis_splits(
        hypothesis_splits: Union[List[str], Tuple[str, ...]],
        return_empty_dict_if_no_slots: bool = False,
        slot_type_filter: Union[Callable, Iterable] = None
) -> Optional[Mapping[str, str]]:
    """
    Gets the slot type/value dictionary from the NLU hypothesis splits.

    Args:
        hypothesis_splits: the NLU hypothesis splits.
        return_empty_dict_if_no_slots: returns an empty dictionary instead of None
            when there is no slots in the NLU hypothesis.
        slot_type_filter: a filter for slot types; can be a function or an iterable such as a set.

    Returns: a dictionary of slot type/value mappings from the NLU hypothesis splits;
        returns an empty dictionary if there is no slots in the NLU hypothesis splits.

    Examples:
        >>> hypothesis = 'Music|PlayMusicIntent'
        >>> assert get_slots_dict_from_hypothesis_splits(get_hypothesis_splits(hypothesis)) is None

        >>> hypothesis = 'Music|PlayMusicIntent'
        >>> get_slots_dict_from_hypothesis_splits(
        ...    get_hypothesis_splits(hypothesis),
        ...    return_empty_dict_if_no_slots=True
        ... )
        {}

        >>> hypothesis = 'Music|PlayMusicIntent|MediaType:song|SongName:chandelier'
        >>> get_slots_dict_from_hypothesis_splits(get_hypothesis_splits(hypothesis))
        {'MediaType': 'song', 'SongName': 'chandelier'}


        >>> hypothesis = 'Music|PlayMusicIntent|MediaType:song|SongName:chandelier'
        >>> get_slots_dict_from_hypothesis_splits(
        ...    get_hypothesis_splits(hypothesis),
        ...    slot_type_filter = {'SongName', 'ArtistName'}
        ... )
        {'SongName': 'chandelier'}

        >>> from boba_python_utils.product_utils.nlu import is_for_comparison_slot_type
        >>> hypothesis = 'Music|PlayMusicIntent|MediaType:song|SongName:chandelier'
        >>> get_slots_dict_from_hypothesis_splits(
        ...    get_hypothesis_splits(hypothesis),
        ...    slot_type_filter = is_for_comparison_slot_type
        ... )
        {'SongName': 'chandelier'}
    """

    if hypothesis_splits:
        slots = dict(filter_tuples_by_head_element(
            slot_type_filter,
            map(
                lambda x: (x[0].strip(), (x[1] if len(x) == 2 else '')),
                (
                    slots_str.split(SEP_NLU_SLOT_PARTS, 1)
                    for slots_str in sorted(hypothesis_splits[2:])
                )
            )
        ))

        return slots if (return_empty_dict_if_no_slots or slots) else None
    else:
        return {} if return_empty_dict_if_no_slots else None


def get_slots_dict_from_hypothesis(
        hypothesis: str,
        return_empty_dict_if_no_slots: bool = False,
        slot_type_filter: Union[Callable, Iterable] = None
) -> Optional[Mapping[str, str]]:
    """
    See `get_slots_dict_from_hypothesis_splits`.
    """
    return get_slots_dict_from_hypothesis_splits(
        get_hypothesis_splits(hypothesis),
        return_empty_dict_if_no_slots=return_empty_dict_if_no_slots,
        slot_type_filter=slot_type_filter
    )


def has_same_slot_types(
        nlu_hypothesis1: str,
        nlu_hypothesis2: str,
        slot_type_filter: Union[Callable, Iterable] = None
):
    """

    Args:
        nlu_hypothesis1:
        nlu_hypothesis2:
        slot_type_filter:

    Returns:

    Examples:
        >>> from boba_python_utils.product_utils.nlu import is_for_comparison_slot_type
        >>> has_same_slot_types(
        ...    "Music|PlayMusicIntent|MediaType:song|SongName:chandelier",
        ...    "Music|PlayMusicIntent|ArtistName:sia|SongName:chandelier",
        ...    slot_type_filter = is_for_comparison_slot_type
        ... )
        False

        >>> has_same_slot_types(
        ...    "Notifications|CancelNotificationIntent|DeviceLocation:room|DeviceName:gemma's|OnType:alarm",
        ...    "Notifications|CancelNotificationIntent|DeviceLocation:room|DeviceName:chase's|OnType:alarm",
        ...    slot_type_filter = is_for_comparison_slot_type
        ... )
        True
    """
    slots1 = get_slots_dict_from_hypothesis(
        nlu_hypothesis1,
        return_empty_dict_if_no_slots=True,
        slot_type_filter=slot_type_filter
    )
    slots2 = get_slots_dict_from_hypothesis(
        nlu_hypothesis2,
        return_empty_dict_if_no_slots=True,
        slot_type_filter=slot_type_filter
    )
    return set(slots1) == set(slots2)


def has_same_slot_types_udf(
        nlu_hypothesis1: str,
        nlu_hypothesis2: str,
        slot_type_filter: Union[Callable, Iterable] = None
):
    return F.udf(
        partial(has_same_slot_types, slot_type_filter=slot_type_filter),
        returnType=BooleanType()
    )(
        nlu_hypothesis1, nlu_hypothesis2
    )


def get_slot_value_from_hypothesis(hypothesis: str, slot_type: str) -> Optional[str]:
    """
    Gets slot value of the specified slot type from an NLU hypothesis.
    Args:
        hypothesis: the NLU hypothesis.
        slot_type: the slot type.

    Returns: the slot value of specified slot type from the hypothesis;
        None if teh slot type does not exist in the hypothesis.

    """
    slots = get_slots_dict_from_hypothesis(hypothesis)
    if slots is None:
        return None
    return slots.get(slot_type, None)


def get_slot_value_from_hypothesis_udf(hypothesis: str, slot_type: str):
    return F.udf(
        partial(get_slot_value_from_hypothesis, slot_type=slot_type),
    )(hypothesis)


def get_slot_types_values_tuples_from_hypothesis_splits(
        hypothesis_splits
) -> Tuple[List[str], List[str]]:
    """
    Gets a tuple of slot type list and value list from the NLU hypothesis splits.

    Args:
        hypothesis_splits: the NLU hypothesis splits.

    Returns: the slot type list and the slot value list from the NLU hypothesis splits;
        returns two emtpy lists if there is no slots in the NLU hypothesis splits.
    """
    slot_types, slot_values = [], []
    for slotstr in sorted(hypothesis_splits[2:]):
        slot_splits = slotstr.split(':', 1)
        if len(slot_splits) == 1:
            slot_types.append(slot_splits[0].strip())
            slot_values.append(None)
        else:
            slot_types.append(slot_splits[0].strip())
            slot_values.append(slot_splits[1].strip())
    return slot_types, slot_values


def get_domain_intent_slots_from_hypothesis_splits(
        hypothesis_splits: Union[List[str], Tuple[str, ...]],
        slots_as_dict: bool = False,
        slots_as_types_and_values: bool = False,
        third_party_skill_as_one_domain: bool = False,
        return_empty_dict_if_no_slots: bool = False,
):
    """
    Gets domain/intent string and the slots from the NLU hypothesis splits.
    The returned slots can be in different datq structure given the options.

    Args:
        hypothesis_splits: the NLU hypothesis splits.
        slots_as_dict: True to return slots as a slot type/value dictionary.
        slots_as_types_and_values: True to return slots as a slot type list and a slot value list.
        third_party_skill_as_one_domain: uses `DOMAIN_NAME_ALL_3P`
            as the domain string for all 3p domain.
        return_empty_dict_if_no_slots: returns an empty dictionary instead of None
            when `slots_as_dict` is set True and there is no slots in the NLU hypothesis.

    Returns: domain/intent string and the slots from the NLU hypothesis splits.
    If `slots_as_dict` is True, then returns slots as a slot type/value dictionary;
    otherwise, if `slots_as_types_and_values` is True,
        then returns slots as a slot type list and a slot value list;
    otherwise,
        returns slots a a list of the raw slot strings.

    """
    if not hypothesis_splits:
        return None, None, ({} if return_empty_dict_if_no_slots else None)

    domain = solve_domain(
        hypothesis_splits[0],
        third_party_skill_as_one_domain=third_party_skill_as_one_domain
    )
    if len(hypothesis_splits) == 1:
        return domain, None, ({} if return_empty_dict_if_no_slots else None)
    elif len(hypothesis_splits) == 2:
        return domain, hypothesis_splits[1], ({} if return_empty_dict_if_no_slots else None)
    else:
        if slots_as_dict:
            return domain, \
                hypothesis_splits[1], \
                get_slots_dict_from_hypothesis_splits(hypothesis_splits)
        elif slots_as_types_and_values:
            return domain, \
                hypothesis_splits[1], \
                get_slot_types_values_tuples_from_hypothesis_splits(hypothesis_splits)
        else:
            return domain, \
                hypothesis_splits[1], \
                hypothesis_splits[2:]


def get_domain_intent_slots_from_hypothesis(
        hypothesis: str,
        slots_as_dict: bool = False,
        slots_as_types_and_values: bool = False,
        third_party_skill_as_one_domain: bool = False,
        return_empty_dict_if_no_slots: bool = False
):
    """
    See `get_domain_intent_slots_from_hypothesis_splits`.
    """
    return get_domain_intent_slots_from_hypothesis_splits(
        hypothesis_splits=get_hypothesis_splits(hypothesis),
        slots_as_dict=slots_as_dict,
        slots_as_types_and_values=slots_as_types_and_values,
        third_party_skill_as_one_domain=third_party_skill_as_one_domain,
        return_empty_dict_if_no_slots=return_empty_dict_if_no_slots
    )


# endregion

# region entities

def extract_entities(hypothesis, return_dict=False, start_split_idx: int = 2):
    out = []
    if hypothesis:
        splits = hypothesis.split('|')[start_split_idx:]
        if splits:
            for split in splits:
                split2 = split.split(':', 1)
                if len(split2) == 1:
                    continue
                if return_dict:
                    out.append(
                        {
                            KEY_ENTITY_TYPE: split2[0],
                            KEY_ENTITY: split2[1],
                        }
                    )
                else:
                    out.append(split2)
    return out


def extract_entities_udf(
        nlu_hypothesis_col: Union[str, Column],
        entity_type_colname: str = KEY_ENTITY_TYPE,
        entity_value_colname: str = KEY_ENTITY,
        start_split_idx: int = 2
) -> Column:
    return F.udf(
        partial(extract_entities, start_split_idx=start_split_idx),
        returnType=ArrayType(
            StructType(
                [
                    StructField(entity_type_colname, StringType(), True),
                    StructField(entity_value_colname, StringType(), True),
                ]
            )
        ),
    )(nlu_hypothesis_col)


def spark_explode_entities_from_nlu_hypothesis(
        df_data_with_nlu_hypothesis: DataFrame,
        nlu_hypothesis_colname: str,
        entity_colname_prefix: str = None,
        entity_colname_suffix: str = None
) -> DataFrame:
    from boba_python_utils.spark_utils.common import get_internal_colname
    from boba_python_utils.spark_utils.data_transform import explode_as_flat_columns
    _KEY_ENTITIES = get_internal_colname(KEY_ENTITIES)
    return explode_as_flat_columns(
        df_data_with_nlu_hypothesis.withColumn(
            _KEY_ENTITIES,
            extract_entities_udf(nlu_hypothesis_colname)
        ),
        col_to_explode=_KEY_ENTITIES,
        explode_colname_or_prefix=entity_colname_prefix,
        explode_colname_suffix=entity_colname_suffix
    )


def spark_explode_entities_from_resolved_slots(
        df_data_with_resolved_slots: DataFrame,
        resolved_slots_colname: str,
        entity_colname_prefix: str = None,
        entity_colname_suffix: str = None
) -> DataFrame:
    from boba_python_utils.spark_utils.common import get_internal_colname
    from boba_python_utils.spark_utils.data_transform import explode_as_flat_columns
    _KEY_ENTITIES = get_internal_colname(KEY_ENTITIES)
    return explode_as_flat_columns(
        df_data_with_resolved_slots.withColumn(
            _KEY_ENTITIES,
            extract_entities_udf(resolved_slots_colname, start_split_idx=0)
        ),
        col_to_explode=_KEY_ENTITIES,
        explode_colname_or_prefix=entity_colname_prefix,
        explode_colname_suffix=entity_colname_suffix
    )


def spark_explode_entities_from_resolved_slots_or_nlu_hypothesis(
        df_data_with_resolved_slots_and_nlu_hypothesis: DataFrame,
        resolved_slots_colname: str,
        nlu_hypothesis_colname: str,
        entity_colname_prefix: str = None,
        entity_colname_suffix: str = None,
        num_slots_colname: str = None,
        num_entities_colname: str = None,
        target_slot_types: Iterable[str] = None
):
    from boba_python_utils.spark_utils.common import get_internal_colname
    from boba_python_utils.spark_utils.data_transform import explode_as_flat_columns
    from boba_python_utils.production_utils._nlu.target_slots import get_num_target_slot_types_udf

    _df = df_data_with_resolved_slots_and_nlu_hypothesis
    _KEY_ENTITIES = get_internal_colname(KEY_ENTITIES)

    _df = _df.withColumn(
        _KEY_ENTITIES,
        extract_entities_udf(resolved_slots_colname, start_split_idx=0)
    )

    # only 1% queries do not have ER-resolved entities, but with non-empty slots
    _df = _df.withColumn(
        _KEY_ENTITIES,
        F.when(
            ((F.size(_KEY_ENTITIES) == 0) & (F.size(F.split(nlu_hypothesis_colname, r"\|")) > 2)),
            extract_entities_udf(nlu_hypothesis_colname),
        ).otherwise(F.col(_KEY_ENTITIES)),
    )

    if num_slots_colname:
        _df = _df.withColumn(num_slots_colname, F.size(_KEY_ENTITIES))

    if num_entities_colname:
        if target_slot_types is None:
            from boba_python_utils.production_utils._nlu.target_slots import NON_STOP_SLOT_TYPES
            target_slot_types = NON_STOP_SLOT_TYPES
        _df = _df.withColumn(num_entities_colname, get_num_target_slot_types_udf(
            _KEY_ENTITIES,
            entity_type_colname=KEY_ENTITY_TYPE,
            target_slot_types=target_slot_types,
        ))

    return explode_as_flat_columns(
        _df,
        col_to_explode=_KEY_ENTITIES,
        explode_colname_or_prefix=entity_colname_prefix,
        explode_colname_suffix=entity_colname_suffix
    )


def extract_entity_pairs(hypothesis1, hypothesis2, exclude_identity=True, target_entity_types=None):
    entities1 = extract_entities(hypothesis1)
    entities2 = extract_entities(hypothesis2)

    out = []
    for (entity_type1, entity_val1), (entity_type2, entity_val2) in product(entities1, entities2):
        if exclude_identity and entity_val1 == entity_val2 and entity_type1 == entity_type2:
            continue
        if target_entity_types is not None:
            if callable(target_entity_types):
                if not (target_entity_types(entity_type1) and target_entity_types(entity_type2)):
                    continue
            elif not (entity_type1 in target_entity_types and entity_type2 in target_entity_types):
                continue
        out.append(
            {
                'entity_type_first': entity_type1,
                'entity_first': entity_val1,
                'entity_type_second': entity_type2,
                'entity_second': entity_val2,
            }
        )

    if not out:
        return None
    return out


def extract_entity_pairs_udf(
        hypothesis1, hypothesis2, exclude_identity=True, target_entity_types=None
):
    return F.udf(
        partial(
            extract_entity_pairs,
            exclude_identity=exclude_identity,
            target_entity_types=target_entity_types,
        ),
        returnType=ArrayType(
            StructType(
                [
                    StructField(KEY_ENTITY_TYPE_FIRST, StringType(), True),
                    StructField(KEY_ENTITY_FIRST, StringType(), True),
                    StructField(KEY_ENTITY_TYPE_SECOND, StringType(), True),
                    StructField(KEY_ENTITY_SECOND, StringType(), True),
                ]
            )
        ),
    )(hypothesis1, hypothesis2)


def concat_entity_type_and_value(entity_type: str, entity_value: str) -> str:
    return f'{entity_type}:{entity_value}'


@F.udf(returnType=StringType())
def concat_entity_type_and_value_udf(entity_type, entity_value):
    return F.concat(F.col(entity_type), F.lit(':'), F.col(entity_value))

# endregion
