from functools import partial
from typing import Union, Callable, Iterable

from pyspark.sql import functions as F
from pyspark.sql.column import Column
from pyspark.sql.types import BooleanType, IntegerType, StringType

from boba_python_utils.spark_utils.typing import NameOrColumn
from boba_python_utils.production_utils._nlu.hypothesis_parsing import get_slots_dict_from_hypothesis

BASIC_ENTITY_SLOT_TYPES = [
    'Name',
    'Item',
    'CallSign',
    'Content',
    'Ingredient'
]

_ENTITY_SLOT_TYPES = [
    'Country',
    'State',
    'City',
    'Place',
    'Player',
    'Device',
    'Appliance',
    'Location',
    'Lyrics',
    'ContactName'
]

ENTITY_SLOT_TYPES = [
    *BASIC_ENTITY_SLOT_TYPES,
    *_ENTITY_SLOT_TYPES
]

_FOR_COMPARISON_SLOT_TYPES = [
    'Phrase',
    'Number',
    'Time',
    'Date',
    'Duration',
    'Language',
    'Label',
    'Title',
    'Topic',
    'Query',
    'Question',
    'AnnouncementContent',
    'MessageContent'
]

FOR_COMPARISON_SLOT_TYPES = [
    *ENTITY_SLOT_TYPES,
    *_FOR_COMPARISON_SLOT_TYPES
]

_NON_STOP_SLOT_TYPES = [
    'Type',
    'Setting',
    'Value'
]

NON_STOP_SLOT_TYPES = [
    *FOR_COMPARISON_SLOT_TYPES,
    *_NON_STOP_SLOT_TYPES
]

_NON_BASIC_ENTITY_SLOT_TYPES = _ENTITY_SLOT_TYPES + _FOR_COMPARISON_SLOT_TYPES + _NON_STOP_SLOT_TYPES
_NON_ENTITY_SLOT_TYPES = _FOR_COMPARISON_SLOT_TYPES + _NON_STOP_SLOT_TYPES


# region general target slot type functions

def is_target_slot_type(entity_type: str, target_slot_types: Iterable[str]) -> bool:
    if entity_type is not None and isinstance(entity_type, str):
        return any((_token in entity_type) for _token in target_slot_types)


def is_target_slot_type_udf(entity_type: NameOrColumn, target_slot_types: Iterable[str]) -> Column:
    return F.udf(
        partial(
            is_target_slot_type,
            target_slot_types=target_slot_types
        ),
        returnType=BooleanType()
    )(entity_type)


def _get_entity_type(entity, entity_type_fieldname):
    if isinstance(entity, (list, tuple)):
        return entity[0]
    else:
        return entity[entity_type_fieldname]


def has_target_slot_type(
        entities: Iterable,
        entity_type_fieldname: str,
        target_slot_types: Iterable[str],
        excluded_target_slot_types: Iterable[str] = None
) -> bool:
    return entities is not None and any(
        (
                is_target_slot_type(
                    _get_entity_type(entity, entity_type_fieldname),
                    target_slot_types=target_slot_types
                ) and
                (
                    (not is_target_slot_type(
                        _get_entity_type(entity, entity_type_fieldname),
                        target_slot_types=excluded_target_slot_types
                    ))
                    if excluded_target_slot_types is not None
                    else True
                )
        )
        for entity in entities
    )


def has_target_slot_type_udf(
        entities: NameOrColumn,
        entity_type_colname: str,
        target_slot_types: Iterable[str],
        excluded_target_slot_types: Iterable[str] = None
) -> Column:
    return F.udf(
        partial(
            has_target_slot_type,
            entity_type_fieldname=entity_type_colname,
            target_slot_types=target_slot_types,
            excluded_target_slot_types=excluded_target_slot_types
        ),
        returnType=BooleanType()
    )(entities)


def get_num_target_slot_types(
        entities: Iterable,
        entity_type_fieldname: str,
        target_slot_types: Iterable[str],
        excluded_target_slot_types: Iterable[str] = None
) -> int:
    return sum(
        int(
            entity is not None and
            is_target_slot_type(
                _get_entity_type(entity, entity_type_fieldname),
                target_slot_types=target_slot_types
            ) and
            (
                (not is_target_slot_type(
                    _get_entity_type(entity, entity_type_fieldname),
                    target_slot_types=excluded_target_slot_types
                ))
                if excluded_target_slot_types is not None
                else True
            )
        )
        for entity in entities
    ) if entities is not None else 0


def get_num_target_slot_types_udf(
        entities: NameOrColumn,
        entity_type_colname: str,
        target_slot_types: Iterable[str],
        excluded_target_slot_types: Iterable[str] = None
) -> Column:
    return F.udf(
        partial(
            get_num_target_slot_types,
            entity_type_fieldname=entity_type_colname,
            target_slot_types=target_slot_types,
            excluded_target_slot_types=excluded_target_slot_types
        ),
        returnType=IntegerType()
    )(entities)


# endregion

# region non-stop slot types
def is_non_stop_slot_type(entity_type: str) -> bool:
    return is_target_slot_type(entity_type, target_slot_types=NON_STOP_SLOT_TYPES)


def is_non_stop_slot_type_udf(entity_type: NameOrColumn) -> Column:
    return is_target_slot_type_udf(entity_type, target_slot_types=NON_STOP_SLOT_TYPES)


def has_non_stop_slot_type(entities: Iterable, entity_type_fieldname: str) -> bool:
    return has_target_slot_type(
        entities=entities,
        entity_type_fieldname=entity_type_fieldname,
        target_slot_types=NON_STOP_SLOT_TYPES
    )


def has_non_stop_slot_type_udf(entities: NameOrColumn, entity_type_colname: str) -> Column:
    return has_target_slot_type_udf(
        entities=entities,
        entity_type_colname=entity_type_colname,
        target_slot_types=NON_STOP_SLOT_TYPES
    )


def get_num_non_stop_slot_types(entities: Iterable, entity_type_fieldname: str) -> int:
    return get_num_target_slot_types(
        entities=entities,
        entity_type_fieldname=entity_type_fieldname,
        target_slot_types=NON_STOP_SLOT_TYPES
    )


def get_num_non_stop_slot_types_udf(entities: NameOrColumn, entity_type_colname: str) -> Column:
    return get_num_target_slot_types_udf(
        entities=entities,
        entity_type_colname=entity_type_colname,
        target_slot_types=NON_STOP_SLOT_TYPES
    )


# endregion

# region for comparison slot types
def is_for_comparison_slot_type(entity_type: str) -> bool:
    return is_target_slot_type(entity_type, target_slot_types=FOR_COMPARISON_SLOT_TYPES)


def is_for_comparison_slot_type_udf(entity_type: NameOrColumn) -> Column:
    return is_target_slot_type_udf(entity_type, target_slot_types=FOR_COMPARISON_SLOT_TYPES)


def has_for_comparison_slot_type(entities: Iterable, entity_type_fieldname: str) -> bool:
    return has_target_slot_type(
        entities=entities,
        entity_type_fieldname=entity_type_fieldname,
        target_slot_types=FOR_COMPARISON_SLOT_TYPES
    )


def has_for_comparison_slot_type_udf(entities: NameOrColumn, entity_type_colname: str) -> Column:
    return has_target_slot_type_udf(
        entities=entities,
        entity_type_colname=entity_type_colname,
        target_slot_types=FOR_COMPARISON_SLOT_TYPES
    )


def get_num_for_comparison_slot_types(entities: Iterable, entity_type_fieldname: str) -> int:
    return get_num_target_slot_types(
        entities=entities,
        entity_type_fieldname=entity_type_fieldname,
        target_slot_types=FOR_COMPARISON_SLOT_TYPES
    )


def get_num_for_comparison_slot_types_udf(entities: NameOrColumn, entity_type_colname: str) -> Column:
    return get_num_target_slot_types_udf(
        entities=entities,
        entity_type_colname=entity_type_colname,
        target_slot_types=FOR_COMPARISON_SLOT_TYPES
    )


# endregion

# region basic entity slot types
def is_basic_entity_slot_type(entity_type: str) -> bool:
    return (
            is_target_slot_type(entity_type, target_slot_types=BASIC_ENTITY_SLOT_TYPES) and
            (not is_target_slot_type(entity_type, target_slot_types=_NON_BASIC_ENTITY_SLOT_TYPES))
    )


def is_basic_entity_slot_type_udf(entity_type: NameOrColumn) -> Column:
    return (
            is_target_slot_type_udf(entity_type, target_slot_types=BASIC_ENTITY_SLOT_TYPES) &
            (~is_target_slot_type_udf(entity_type, target_slot_types=_NON_BASIC_ENTITY_SLOT_TYPES))
    )


def has_basic_entity_slot_type(entities: Iterable, entity_type_fieldname: str) -> bool:
    return (
        has_target_slot_type(
            entities=entities,
            entity_type_fieldname=entity_type_fieldname,
            target_slot_types=BASIC_ENTITY_SLOT_TYPES,
            excluded_target_slot_types=_NON_BASIC_ENTITY_SLOT_TYPES
        )
    )


def has_basic_entity_slot_type_udf(entities: NameOrColumn, entity_type_colname: str) -> Column:
    return has_target_slot_type_udf(
        entities=entities,
        entity_type_colname=entity_type_colname,
        target_slot_types=BASIC_ENTITY_SLOT_TYPES,
        excluded_target_slot_types=_NON_BASIC_ENTITY_SLOT_TYPES
    )


def get_num_basic_entity_slot_types(entities: Iterable, entity_type_fieldname: str) -> int:
    return get_num_target_slot_types(
        entities=entities,
        entity_type_fieldname=entity_type_fieldname,
        target_slot_types=BASIC_ENTITY_SLOT_TYPES,
        excluded_target_slot_types=_NON_BASIC_ENTITY_SLOT_TYPES
    )


def get_num_basic_entity_slot_types_udf(entities: NameOrColumn, entity_type_colname: str) -> Column:
    return get_num_target_slot_types_udf(
        entities=entities,
        entity_type_colname=entity_type_colname,
        target_slot_types=BASIC_ENTITY_SLOT_TYPES,
        excluded_target_slot_types=_NON_BASIC_ENTITY_SLOT_TYPES
    )


# endregion

# region entity slot types
def is_entity_slot_type(entity_type: str) -> bool:
    """

    Args:
        entity_type:

    Returns:

    Examples:
        >>> is_entity_slot_type('AnnouncementContent')
        False
    """
    return is_target_slot_type(entity_type, target_slot_types=ENTITY_SLOT_TYPES) and (
        not is_target_slot_type(entity_type, target_slot_types=_NON_ENTITY_SLOT_TYPES)
    )


def is_entity_slot_type_udf(entity_type: NameOrColumn) -> Column:
    return (
            is_target_slot_type_udf(entity_type, target_slot_types=ENTITY_SLOT_TYPES) &
            (~is_target_slot_type_udf(entity_type, target_slot_types=_NON_ENTITY_SLOT_TYPES))
    )


def has_entity_slot_type(entities: Iterable, entity_type_fieldname: str) -> bool:
    return has_target_slot_type(
        entities=entities,
        entity_type_fieldname=entity_type_fieldname,
        target_slot_types=ENTITY_SLOT_TYPES
    )


def has_entity_slot_type_udf(entities: NameOrColumn, entity_type_colname: str) -> Column:
    return has_target_slot_type_udf(
        entities=entities,
        entity_type_colname=entity_type_colname,
        target_slot_types=ENTITY_SLOT_TYPES
    )


def get_num_entity_slot_types(entities: Iterable, entity_type_fieldname: str) -> int:
    return get_num_target_slot_types(
        entities=entities,
        entity_type_fieldname=entity_type_fieldname,
        target_slot_types=ENTITY_SLOT_TYPES
    )


def get_num_entity_slot_types_udf(entities: NameOrColumn, entity_type_colname: str) -> Column:
    return get_num_target_slot_types_udf(
        entities=entities,
        entity_type_colname=entity_type_colname,
        target_slot_types=ENTITY_SLOT_TYPES
    )


# endregion


def normalize_utterance_text_by_slot_values(
        utterance1,
        utterance2,
        nlu_hypothesis1,
        nlu_hypothesis2,
        slot_type_filter: Union[Callable, Iterable] = is_non_stop_slot_type,
        remove_overlap: bool = False
):
    """

    Examples:
        >>> normalize_utterance_text_by_slot_values(
        ...   "pen kids' song",
        ...   "open baby song",
        ...   "Music|PlayMusicIntent|GenreName:kids'|MediaType:song",
        ...   "GeneralMedia|LaunchNativeAppIntent|AppName:baby song"
        ... )
        ("kids' song", 'baby song')
        >>> normalize_utterance_text_by_slot_values(
        ...   "the post",
        ...   "play the pulse on siriusxm",
        ...   "amzn1.ask.skill.cb01deae-b670-400f-be87-20b868275e18_Live|PlayAudio",
        ...   "Music|PlayStationIntent|ServiceName:siriusxm|StationName:the pulse"
        ... )
        ('', 'siriusxm the pulse')
        >>> normalize_utterance_text_by_slot_values(
        ...   "alexi",
        ...   "play no juice by boosie",
        ...   "Knowledge|QAIntent|Question:alexi",
        ...   "Music|PlayMusicIntent|ArtistName:boosie|SongName:no juice"
        ... )
        ('alexi', 'boosie no juice')
        >>> normalize_utterance_text_by_slot_values(
        ...   "cancel alarm in gemma's room",
        ...   "cancel alarm in chase's room",
        ...   "Notifications|CancelNotificationIntent|DeviceLocation:room|DeviceName:gemma's|OnType:alarm",
        ...   "Notifications|CancelNotificationIntent|DeviceLocation:room|DeviceName:chase's|OnType:alarm",
        ...   slot_type_filter=None,
        ...   remove_overlap=True
        ... )
        ("gemma's", "chase's")
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

    shared_non_stop_slot_types = set(slots1) & set(slots2)
    text1, text2 = [], []
    for slot_type in shared_non_stop_slot_types:
        if (not remove_overlap) or slots1[slot_type] != slots2[slot_type]:
            text1.append(slots1[slot_type])
            text2.append(slots2[slot_type])

    for slot_type, slot_value in slots1.items():
        if slot_type not in shared_non_stop_slot_types:
            if (not remove_overlap) or (slot_value not in utterance2):
                text1.append(slot_value)

    for slot_type, slot_value in slots2.items():
        if slot_type not in shared_non_stop_slot_types:
            if (not remove_overlap) or (slot_value not in utterance1):
                text2.append(slot_value)

    return ' '.join(text1), ' '.join(text2)


get_non_stop_slot_type_texts_with_slot_align_udf = F.udf(normalize_utterance_text_by_slot_values)
