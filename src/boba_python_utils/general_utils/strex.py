import re
from collections import defaultdict
from typing import Tuple, Mapping

import boba_python_utils.string_utils.regex as rex
from boba_python_utils.common_utils.set_helper import compare_sets
from boba_python_utils.general_utils.nlp_utility.common import Languages
from boba_python_utils.string_utils.prefix_suffix import *


def get_processed_strings_and_map(
        strs: Iterable[str],
        str_proc: Callable,
        filter: Callable = None,
        keep_the_identical_in_map: bool = False,
        allow_duplicates_in_map: bool = False,
        **kwargs
) -> Tuple[List[str], Union[Mapping[str, str], Mapping[str, List[str]]]]:
    """
    Applies string processing function `str_proc` to `strs`, returns the processed strings
        and an map between processed strings and the original strings.
    If `filter` is specified, a processed string needs to pass the filter to return.

    Args:
        strs: the input strings.
        str_proc: the string processing function.
        filter: a processed string is returned if it passes this filter function
                (i.e. the function returns True).
        keep_the_identical_in_map: True to keep all processed string to
                original string mapping even if they are identical.
        allow_duplicates_in_map: True to allow one-to-many processed string
                to original string mapping;
            in this case the it is a mapping between the processed string
                and a list of original strings.
        **kwargs: named arguments for `str_proc`.

    Returns: the list of processed strings, and a processed string to original string mapping
            (can be one-one mapping or one-to-many mapping dependent on `allow_duplicates_in_map`)

    """

    processed_strs = []
    process_str_to_original_str_map = defaultdict(list) if allow_duplicates_in_map else {}
    for s in strs:
        s_processed = str_proc(s, **kwargs)
        if filter is None or filter(s):
            processed_strs.append(s_processed)
            if keep_the_identical_in_map or s_processed != s:
                if allow_duplicates_in_map:
                    process_str_to_original_str_map[s_processed].append(s)
                else:
                    process_str_to_original_str_map[s_processed] = s
    return processed_strs, process_str_to_original_str_map


# region advanced applications
def get_token_inflection_regex(
        token: str,
        optional_space=True,
        augmentation=True,
        language: Union[str, Languages] = Languages.English,
) -> str:
    """
    Generate a regex pattern that matches the input token and its inflections.

    Args:
        token: The input token for which the regex pattern is generated.
        optional_space: If True, spaces in the token are replaced with optional spaces.
        augmentation: If True, the regex pattern will match the token's inflections.
        language: The language of the token.

    Returns:
        str: A regex pattern that matches the input token and its inflections.

    Examples:
        >>> get_token_inflection_regex("dog")
        "(?:dog(?:'s)?)|(?:dogs'?)|(?:dogg?est)|(?:dogg?er'?s?)|(?:dogg?or(?:'s?)?)|(?:dogg?ing)|(?:dogness)"

        >>> get_token_inflection_regex("dog", optional_space=False, augmentation=False)
        'dog'

        >>> get_token_inflection_regex("box", optional_space=False, augmentation=True)
        "(?:box(?:'s)?)|(?:boxs'?)|(?:boxx?est)|(?:boxx?er'?s?)|(?:boxx?or(?:'s?)?)|(?:boxx?ing)|(?:boxness)"

    Raises:
        ValueError: If the provided language is not supported.

    """

    token = re.escape(token)
    if optional_space:
        token = token.replace(' ', r's?')  # `re.escape` already adds a '\' before space
    if augmentation:
        group = []
        if language == Languages.English:  # TODO support other languages
            if token[-1] == 's':
                group.append(f'(?:{token})')
                group.append(f"(?:{token}'?)")  # Xs'
            else:
                group.append(f"(?:{token}(?:'s)?)")  # X's
                group.append(f"(?:{token}s'?)")  # Xs'

            if 'a' <= token[-1] <= 'z':
                if token[-1] in ('s', 'h'):
                    group.append(f"(?:{token}es'?)")  # Xses, Xhes, Xses', Xhes'
                if token[-1] == 'y':
                    group.append(f"(?:{token[:-1]}ies'?)")  # Xies, Xies'
                if len(token) > 2 and token not in ('off',):
                    if not token.endswith('est'):
                        group.append(f'(?:{token}{token[-1]}?est)')  # Xest
                        if token[-1] == 'y':
                            group.append(f'(?:{token[:-1]}iest)')  # Xiest
                    if not token.endswith('er'):
                        group.append(f"(?:{token}{token[-1]}?er'?s?)")  # Xer, Xers, Xer's
                        if token[-1] == 'y':
                            group.append(f"(?:{token[:-1]}ier'?s?)")  # Xiers, Xier's
                    if not token.endswith('or'):
                        group.append(f"(?:{token}{token[-1]}?or(?:'?s?)?)")  # Xor, Xors, Xor's
                    if not token.endswith('ing'):
                        group.append(f'(?:{token}{token[-1]}?ing)')  # Xing
                    if not token.endswith('ness'):
                        group.append(f'(?:{token}ness)')  # Xness
                        if token[-1] == 'y':
                            group.append(f'(?:{token[:-1]}iness)')  # Xiness
        else:
            raise ValueError(f"language '{language}' is not supported")

        return '|'.join(group)
    else:
        return token


def _construct_regex_from_token_tup(
        tokens,
        add_word_boundary=True,
        optional_space=True,
        token_augmentation=True,
        language='en'
):
    regex = []
    for token in tokens:
        if isinstance(token, str):
            token_regex = get_token_inflection_regex(
                token,
                optional_space=optional_space,
                augmentation=token_augmentation,
                language='en'
            )
            regex.append(f'({token_regex})')
        else:
            group = []
            for _token in token:
                token_regex = get_token_inflection_regex(
                    _token,
                    optional_space=optional_space,
                    augmentation=token_augmentation,
                    language='en'
                )
                group.append(f'(?:{token_regex})')
            regex.append(f"({'|'.join(group)})")

    regex = '|'.join(regex)
    if add_word_boundary:
        return fr'(?:\b|^)({regex})(?:\b|\s|$)'
    else:
        return regex


def num_overlap_tokens(
        str1: str,
        str2: str,
        tokenizer=None
) -> int:
    if not tokenizer:
        str1 = set(str1.split())
        str2 = set(str2.split())
    elif isinstance(tokenizer, str):
        str1 = set(str1.split(tokenizer))
        str2 = set(str2.split(tokenizer))
    elif callable(tokenizer):
        str1 = set(tokenizer(str1))
        str2 = set(tokenizer(str2))
    return len(str1.intersection(str2))


def solve_conflict_keywords_config(
        conflict_keywords: Union[Iterable[Tuple], Mapping[Languages, Iterable[Tuple]]],
        language: Languages = Languages.English,
        always_include_en: bool = True
) -> Mapping:
    """
    Solve the conflict keywords configuration.

    This function takes a conflict_keywords input and resolves it based on the provided language
    and `always_include_en` options. It returns the conflict keywords for the specified language
    and, if `always_include_en` is True, includes the English conflict keywords as well.

    Args:
        conflict_keywords: A mapping from languages to conflict keywords setup or an iterable of tuples.
        language: The language of the text. Default is English.
        always_include_en: If True, always include the English conflict keywords.

    Returns:
        Mapping: The resolved conflict keywords for the specified language and the English language
                 if always_include_en is True.

    Examples:
        >>> conflict_keywords = {
        ...     Languages.English: [('red', 'blue'), (('car', 'vehicle'), 'plane')],
        ...     Languages.Spanish: [('rojo', 'azul'), (('coche', 'vehículo'), 'avión')]
        ... }
        >>> solve_conflict_keywords_config(conflict_keywords, Languages.Spanish, always_include_en=True)
        [('rojo', 'azul'), (('coche', 'vehículo'), 'avión'), ('red', 'blue'), (('car', 'vehicle'), 'plane')]

        >>> solve_conflict_keywords_config(conflict_keywords, Languages.Spanish, always_include_en=False)
        [('rojo', 'azul'), (('coche', 'vehículo'), 'avión')]

        >>> solve_conflict_keywords_config(conflict_keywords, Languages.English, always_include_en=True)
        [('red', 'blue'), (('car', 'vehicle'), 'plane')]
    """
    if isinstance(conflict_keywords, Mapping):
        if language is None:
            language = Languages.English
        if language not in conflict_keywords:
            raise ValueError(f"language '{language}' is not supported")
        _conflict_keywords = conflict_keywords[language]
        if always_include_en and language != Languages.English:
            _conflict_keywords = [*_conflict_keywords, *conflict_keywords[Languages.English]]
        conflict_keywords = _conflict_keywords
    elif language is not None:
        if (not always_include_en) and language != Languages.English:
            raise ValueError(f"specified language '{language}', "
                             f"then 'conflict_keywords' must be a mapping from languages to "
                             f"conflict keywords setup")
    return conflict_keywords


def has_conflict_keywords_allowing_shared_keywords(
        str1,
        str2,
        conflict_keywords: Iterable[Tuple],
        optional_space=False,
        token_augmentation=False,
        allows_add: bool = True,
        allows_drop: bool = True,
        language: Union[str, Languages] = Languages.English,
        always_include_en=True
):
    """

    Args:
        str1:
        str2:
        conflict_keywords:
        optional_space:
        token_augmentation:
        allows_add:
        allows_drop:
        language:
        always_include_en:

    Returns:

    >>> has_conflict_keywords_allowing_shared_keywords(
    ...    "please would you check the mode setting for wash cycle on washer",
    ...    'device endpointId is 45, name is "living room", appliance type is light, actions are SmartHome.adjustBrightness, SmartHome.decreaseColorTemperature, SmartHome.getBrightness, SmartHome.getColor, SmartHome.getColorTemperature, SmartHome.getConnectivity, SmartHome.getPower, SmartHome.increaseColorTemperature, SmartHome.setBrightness, SmartHome.setColor, SmartHome.setColorTemperature, SmartHome.turnOff and SmartHome.turnOn',
    ...    conflict_keywords=[[
    ...        ['bed room', 'bedroom'],
    ...        ['living room', 'common room', 'reception', 'parlor'],
    ...        ['bath room', 'bathroom', 'restroom', 'toilet'],
    ...        ['kitchen', 'dining', 'washer'],
    ...        ['den', 'study', 'office', 'study room'],
    ...        ['front porch', 'front door', 'front yard', 'outside'],
    ...        ['back door', 'back yard', 'backyard', 'outside'],
    ...        ['side door', 'outside'],
    ...        ['balcony', 'outside'],
    ...        ['family room'],
    ...        ['game room', 'play room', 'playroom'],
    ...        ['lobby', 'lounge'],
    ...        ['garbage', 'trash'],
    ...        ['garage', 'storage'],
    ...        ['basement', 'storage'],
    ...        ['laundry', 'washer'],
    ...        ['pantry'],
    ...        ['dresser']
    ...    ]]
    ... )

    """
    if str1 and str2:
        conflict_keywords = solve_conflict_keywords_config(
            conflict_keywords=conflict_keywords,
            language=language,
            always_include_en=always_include_en
        )

        hit_kws1, hit_kws2 = set(), set()
        for conflict_keywords_groups in conflict_keywords:
            for keywords_groups in conflict_keywords_groups:
                kw_group_regex = _construct_regex_from_token_tup(
                    keywords_groups,
                    add_word_boundary=True,
                    optional_space=optional_space,
                    token_augmentation=token_augmentation,
                    language=language
                )
                if re.search(kw_group_regex, str1) is not None:
                    hit_kws1.update(keywords_groups)
                if re.search(kw_group_regex, str2) is not None:
                    hit_kws2.update(keywords_groups)

        if compare_sets(
                hit_kws1,
                hit_kws2,
                allows_add=allows_add,
                allows_drop=allows_drop
        ):
            return True

    elif str1:
        return not allows_drop
    elif str2:
        return not allows_add
    return False


def has_conflict_keywords(
        str1: str,
        str2: str,
        conflict_keywords: Iterable[Tuple],
        optional_space=True,
        token_augmentation=True,
        allows_add: bool = True,
        allows_drop: bool = True,
        language: Union[str, Languages] = Languages.English,
        always_include_en=True
) -> bool:
    """

    Args:
        str1:
        str2:
        conflict_keywords:
        optional_space:
        token_augmentation:
        allows_add:
        allows_drop:
        language:
        always_include_en:

    Returns:


    """
    conflict_keywords = solve_conflict_keywords_config(
        conflict_keywords=conflict_keywords,
        language=language,
        always_include_en=always_include_en
    )
    if str1 and str2:
        for _conflict_tokens_tup in conflict_keywords:
            if not isinstance(_conflict_tokens_tup, (tuple, list)):
                raise ValueError(f"one set of conflict tokens must be stored "
                                 f"in a list or tuple; got {_conflict_tokens_tup}")
            token_tup_regex = _construct_regex_from_token_tup(
                _conflict_tokens_tup,
                add_word_boundary=True,
                optional_space=optional_space,
                token_augmentation=token_augmentation,
                language=language
            )
            token_group_indexes1 = rex.get_regex_match_group_indexes(
                pattern=token_tup_regex, string=str1, start_group_index=1
            )
            token_group_indexes2 = rex.get_regex_match_group_indexes(
                pattern=token_tup_regex, string=str2, start_group_index=1
            )

            if compare_sets(
                    token_group_indexes1,
                    token_group_indexes2,
                    allows_add=allows_add,
                    allows_drop=allows_drop
            ):
                return True
    elif str1:
        return not allows_drop
    elif str2:
        return not allows_add
    return False


def _has_token_drop(
        str1: str,
        str2: str,
        token: str,
        optional_space=True,
        augmentation=True,
        language='en'
):
    _tokens = token.split()
    for _token in _tokens:
        token_regex = get_token_inflection_regex(
            token=_token,
            optional_space=optional_space,
            augmentation=augmentation,
            language=language
        )

        token_match1 = re.search(pattern=token_regex, string=str1)
        token_match2 = re.search(pattern=token_regex, string=str2)
        if token_match1 is not None and token_match2 is None:
            return False
    return True


def has_token_drop(
        str1: str,
        str2: str,
        tokens=Iterable[Tuple],
        optional_space=True,
        augmentation=True,
        language='en'
) -> bool:
    for token in iter__(tokens):
        if isinstance(token, str):
            if _has_token_drop(
                    str1,
                    str2,
                    token,
                    optional_space=optional_space,
                    augmentation=augmentation,
                    language=language
            ):
                return True
        else:
            for _token in token:
                if _has_token_drop(
                        str1,
                        str2,
                        _token,
                        optional_space=optional_space,
                        augmentation=augmentation,
                        language=language
                ):
                    return True

    return False


def has_token(
        s: str,
        tokens=Iterable[Tuple],
        optional_space=True,
        augmentation=True,
        language='en'
) -> bool:
    def _has_token(_token):
        token_regex = get_token_inflection_regex(
            token=_token,
            optional_space=optional_space,
            augmentation=augmentation,
            language=language
        )

        return re.search(pattern=token_regex, string=s) is not None

    for token in iter__(tokens):
        if isinstance(token, str):
            if _has_token(token):
                return True
        else:
            for _token in token:
                if _has_token(_token):
                    return True

    return False

# endregion
