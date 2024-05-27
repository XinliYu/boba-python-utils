from typing import List, Union, Iterable

from attr import attrs, attrib

from boba_python_utils.general_utils.nlp_utility.common import Languages
from boba_python_utils.general_utils.nlp_utility.metrics.edit_distance import EditDistanceOptions
from boba_python_utils.general_utils.nlp_utility.string_sanitization import StringSanitizationOptions, StringSanitizationConfig
from boba_python_utils.production_utils._nlu.entity_pairing import ENTITY_STOP_PREFIX_TOKENS
from boba_python_utils.production_utils._nlu.slot_value_change import SlotChangeOptions

DEFAULT_SANITIZATION_ACTIONS_FOR_RISK_SLOT_CHANGE = (
    StringSanitizationOptions.REMOVE_PREFIX,
    StringSanitizationOptions.REMOVE_ACRONYMS_PERIODS_AND_SPACES,
    StringSanitizationOptions.REMOVE_PUNCTUATIONS_EXCEPT_FOR_HYPHEN,
    StringSanitizationOptions.MAKE_FUZZY
)
DEFAULT_SANITIZATION_CONFIG_FOR_RISK_SLOT_CHANGE = StringSanitizationConfig(
    actions=DEFAULT_SANITIZATION_ACTIONS_FOR_RISK_SLOT_CHANGE,
    prefixes_to_sanitize=ENTITY_STOP_PREFIX_TOKENS
)


@attrs(slots=True)
class RiskSlotChangeConfig:
    slot_value_similarity_threshold = attrib(type=float)
    slot_types = attrib(type=List[str], default=None)
    slot_change_options = attrib(type=SlotChangeOptions, default=None)

    # region utterance similarity
    utterance_similarity_threshold = attrib(type=float, default=None)
    enable_utterance_similarity_for_selected_slot_types = attrib(type=List[str], default=None)
    # endregion

    # region edit distance config
    sanitization_config = attrib(type=Union[StringSanitizationConfig, Iterable[StringSanitizationOptions]], default=None)
    edit_distance_consider_sorted_tokens = attrib(type=bool, default=True)
    edit_distance_consider_same_num_tokens = attrib(type=bool, default=True)
    edit_distance_options = attrib(type=EditDistanceOptions, default=None)

    # endregion

    def __attrs_post_init__(self):
        if not self.edit_distance_options:
            self.edit_distance_options = EditDistanceOptions()


def get_conflict_slot_change_configs(language: Languages = Languages.English):
    return [
        RiskSlotChangeConfig(
            slot_types=['Question'],
            slot_value_similarity_threshold=0.55,
            slot_change_options=SlotChangeOptions(
                allows_slot_type_change=False,
                allows_rewrite_slot_value_being_substr=False
            ),
            sanitization_config=[
                StringSanitizationOptions.REMOVE_ACRONYMS_PERIODS_AND_SPACES,
                StringSanitizationOptions.REMOVE_COMMON_TOKENS,
                StringSanitizationOptions.MAKE_FUZZY
            ]
        ),
        RiskSlotChangeConfig(
            slot_types=['ItemName'],
            slot_value_similarity_threshold=0.6,
            slot_change_options=SlotChangeOptions(
                allows_slot_type_change=False,
                allows_rewrite_slot_value_being_substr=False
            ),
            edit_distance_options=EditDistanceOptions(
                weight_distance_if_strs_have_common_start=0.8,
                weight_distance_if_str1_is_substr=0.8,
                weight_distance_by_comparing_start=language,
                weight_distance_by_comparing_end=language,
                weight_distance_for_short_strs=0.6
            ),
            sanitization_config=[
                StringSanitizationOptions.REMOVE_ACRONYMS_PERIODS_AND_SPACES,
                StringSanitizationOptions.MAKE_FUZZY
            ]
        ),
        RiskSlotChangeConfig(
            slot_types=['EventTitle'],
            slot_value_similarity_threshold=0.75,
            slot_change_options=SlotChangeOptions(
                allows_slot_type_change=False,
                allows_rewrite_slot_value_being_substr=False
            ),
            sanitization_config=[
                StringSanitizationOptions.REMOVE_ACRONYMS_PERIODS_AND_SPACES,
                StringSanitizationOptions.MAKE_FUZZY
            ]
        ),
        RiskSlotChangeConfig(
            slot_types=['ChannelName', 'StationName', 'ArtistName', 'ProgramName'],
            slot_value_similarity_threshold=0.4,
            slot_change_options=SlotChangeOptions(
                allows_slot_type_change=True,
                allows_rewrite_slot_value_being_substr=True
            )
        ),
        RiskSlotChangeConfig(
            slot_types=['CallSign', 'AppName', 'StationNumber'],
            slot_value_similarity_threshold=0.75,
            slot_change_options=SlotChangeOptions(
                allows_slot_type_change=True,
                allows_rewrite_slot_value_being_substr=False
            ),
            sanitization_config=[
                StringSanitizationOptions.REMOVE_ACRONYMS_PERIODS_AND_SPACES
            ]
        ),
    ]


def get_risk_slot_change_configs(language: Languages = Languages.English):
    return [
        RiskSlotChangeConfig(
            slot_types=[
                'ArtistName',
                'SongName',
                'AlbumName',
                'VideoName',
                'AppName',
                'PlaylistName',
                'GenreName',
                'BookName'
            ],
            slot_value_similarity_threshold=0.4,
            slot_change_options=SlotChangeOptions(
                allows_slot_type_change=True,
                allows_slot_type_change_to_unspecified_slot_type=True
            ),
            enable_utterance_similarity_for_selected_slot_types=['VideoName', 'BookName'],
            utterance_similarity_threshold=0.7,
            edit_distance_options=EditDistanceOptions(
                weight_distance_if_strs_have_common_start=0.9,
                weight_distance_if_str1_is_substr=0.8,
                no_distance_for_empty_str1=True,
                weight_distance_by_comparing_start=language,
                weight_distance_by_comparing_end=language,
                weight_distance_for_short_strs=0.6
            )
        ),
        RiskSlotChangeConfig(
            slot_types=['AppName'],
            slot_value_similarity_threshold=0.5,
            slot_change_options=SlotChangeOptions(
                allows_slot_type_change=True,
                allows_slot_type_change_to_unspecified_slot_type=True,
                allows_rewrite_slot_value_being_substr=False
            ),
            sanitization_config=[
                StringSanitizationOptions.REMOVE_ACRONYMS_PERIODS_AND_SPACES,
                StringSanitizationOptions.REMOVE_PUNCTUATIONS_EXCEPT_FOR_HYPHEN,
                StringSanitizationOptions.REMOVE_COMMON_PREFIX,
                StringSanitizationOptions.REMOVE_COMMON_SUFFIX,
                StringSanitizationOptions.MAKE_FUZZY
            ],
            edit_distance_consider_sorted_tokens=False,
            edit_distance_consider_same_num_tokens=False,
            edit_distance_options=EditDistanceOptions(
                weight_distance_if_strs_have_common_start=0.9,
                no_distance_for_empty_str1=True
            )
        ),
        RiskSlotChangeConfig(
            slot_types=['BookName', 'AppName'],
            slot_value_similarity_threshold=0.6,
            slot_change_options=SlotChangeOptions(
                allows_slot_type_change=False
            ),
            sanitization_config=[
                StringSanitizationOptions.REMOVE_ACRONYMS_PERIODS_AND_SPACES,
                StringSanitizationOptions.REMOVE_PUNCTUATIONS_EXCEPT_FOR_HYPHEN,
                StringSanitizationOptions.REMOVE_COMMON_TOKENS,
                StringSanitizationOptions.MAKE_FUZZY,
                StringSanitizationOptions.REMOVE_SPACES
            ],
            edit_distance_options=EditDistanceOptions(
                weight_distance_if_strs_have_common_start=0.9,
                weight_distance_if_str1_is_substr=0.9,
                no_distance_for_empty_str1=True
            )
        ),
        RiskSlotChangeConfig(
            slot_types=['StationName'],
            slot_value_similarity_threshold=0.5,
            slot_change_options=SlotChangeOptions(
                allows_slot_type_change=False,
                allows_rewrite_slot_value_being_substr=False
            ),
            sanitization_config=[
                StringSanitizationOptions.REMOVE_ACRONYMS_PERIODS_AND_SPACES,
                StringSanitizationOptions.REMOVE_PUNCTUATIONS_EXCEPT_FOR_HYPHEN,
                StringSanitizationOptions.REMOVE_COMMON_PREFIX,
                StringSanitizationOptions.MAKE_FUZZY
            ],
            edit_distance_options=EditDistanceOptions(
                weight_distance_if_strs_have_common_start=0.9,
                no_distance_for_empty_str1=True
            )
        ),
        RiskSlotChangeConfig(
            slot_types=['PlaylistName'],
            slot_value_similarity_threshold=0.2,
            slot_change_options=SlotChangeOptions(
                allows_slot_type_change=True,
                allows_slot_type_change_to_unspecified_slot_type=True
            )
        ),
        RiskSlotChangeConfig(
            slot_types=['ChannelName', 'CallSign', 'StationName', 'StationNumber', 'ProgramName'],
            slot_value_similarity_threshold=0.4,
            utterance_similarity_threshold=0.9,
            slot_change_options=SlotChangeOptions(
                allows_slot_type_change=True
            )
        ),
        RiskSlotChangeConfig(
            slot_types=['StationName', 'AppName'],
            slot_value_similarity_threshold=0.75,
            slot_change_options=SlotChangeOptions(
                allows_slot_type_change=True,
                allows_rewrite_slot_value_being_substr=False
            )
        ),
        RiskSlotChangeConfig(
            slot_types=['StreetAddress'],
            slot_value_similarity_threshold=0.4,
            slot_change_options=SlotChangeOptions(
                allows_slot_type_change=True,
                allows_slot_type_change_to_unspecified_slot_type=True
            )
        ),
        RiskSlotChangeConfig(
            slot_types=['TopicName'],
            slot_value_similarity_threshold=0.5,
            slot_change_options=SlotChangeOptions(
                allows_slot_type_change=True,
                allows_slot_type_change_to_unspecified_slot_type=True
            )
        ),
        RiskSlotChangeConfig(
            slot_types=['ListName'],
            slot_value_similarity_threshold=0.5,
            slot_change_options=SlotChangeOptions(
                allows_slot_type_change=True,
                allows_slot_type_change_to_unspecified_slot_type=True
            )
        ),
        RiskSlotChangeConfig(
            slot_types=['DishName', 'DrinkItem'],
            slot_value_similarity_threshold=0.5,
            slot_change_options=SlotChangeOptions(
                allows_slot_type_change=True
            ),
            edit_distance_options=EditDistanceOptions(
                weight_distance_if_strs_have_common_start=0.9
            )
        ),
        RiskSlotChangeConfig(
            slot_types=['EventName', 'AccountProviderName'],
            slot_value_similarity_threshold=0.6
        ),
        RiskSlotChangeConfig(
            slot_types=['EchoText'],
            slot_value_similarity_threshold=0.7
        ),
        RiskSlotChangeConfig(
            slot_types=[
                'NotificationLabel',
                'ContentSourceDeviceLocation',
                'DeviceName',
                'ContactName',
                'Device',
                'SettingValue',
                'Question'
            ],
            slot_value_similarity_threshold=0.4,
            slot_change_options=SlotChangeOptions(
                allows_slot_type_change=True
            )
        ),
        RiskSlotChangeConfig(
            slot_types=['ContactName', 'Participant'],
            slot_value_similarity_threshold=0.5,
            slot_change_options=SlotChangeOptions(
                consider_utterance=False,
                allows_slot_type_change=True,
                allows_rewrite_slot_value_being_substr=False
            ),
            sanitization_config=[
                StringSanitizationOptions.REMOVE_ACRONYMS_PERIODS_AND_SPACES,
                StringSanitizationOptions.REMOVE_PUNCTUATIONS_EXCEPT_FOR_HYPHEN,
                StringSanitizationOptions.REMOVE_COMMON_PREFIX,
                StringSanitizationOptions.REMOVE_COMMON_SUFFIX,
                StringSanitizationOptions.MAKE_FUZZY
            ],
            edit_distance_consider_sorted_tokens=False,
            edit_distance_consider_same_num_tokens=False
        ),
        *get_conflict_slot_change_configs(language=language)
    ]


RISK_SLOT_DROP_WHEN_ALL_OTHER_SLOTS_CARRY_OVER = [
    'ArtistName',
    'SongName',
    'AlbumName',
    'AppName',
    'PlaylistName',
    'ChannelName',
    'GenreName',
    'VideoName',
    'CallSign'
]

RISKY_SLOT_ADDITION = [
    'SettingValue',
    'Duration',
    'Date',
    'Time',
    'ContactName',
    'Destination',
    'Place',
    'ItemName',
    'Event'
]

RISKY_SLOT_DROP = [
    'DeviceLocation',
    'SettingValue',
    'Duration',
    'Date',
    'Time',
    'ShortWeatherDetail',
    'ContactName',
    'Destination'
]
