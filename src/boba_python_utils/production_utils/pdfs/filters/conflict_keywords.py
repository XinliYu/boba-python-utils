from functools import partial
from typing import Mapping

from pyspark.sql.functions import udf
from pyspark.sql.types import BooleanType

from boba_python_utils.general_utils import strex as strex
from boba_python_utils.general_utils.nlp_utility.common import Languages
from boba_python_utils.general_utils.nlp_utility.punctuations \
    import remove_acronym_periods_and_spaces as _remove_acronym_periods_and_spaces
from boba_python_utils.string_utils.common import contains_any
from boba_python_utils.production_utils._nlu.hypothesis_parsing import \
    get_slots_dict_from_hypothesis
from boba_python_utils.production_utils.pdfs.filters._configs.conflict_keywords import (
    CONFLICT_KEYWORD_PAIRS, SLOT_SPECIFIC_CONFLICT_KEYWORD_PAIRS
)


def has_conflict_keywords(
        utterance1,
        utterance2,
        conflict_keywords=CONFLICT_KEYWORD_PAIRS,
        language='en',
        always_include_en=True,
        **kwargs
):
    """
    Checkes if two utterances contain conflict keywords,
    using the conflict keywords defined in `CONFLICT_KEYWORD_PAIRS` by default.

    Examples:
        >>> assert has_conflict_keywords("turn off my room", 'turn off playroom')
        >>> assert has_conflict_keywords("turn off my room", "turn off joe's room")
        >>> assert has_conflict_keywords('tell me a joke about moms', 'tell me a joke about dads')
        >>> assert has_conflict_keywords('play c. n. n.', 'play msnbc')
        >>> assert has_conflict_keywords('play p. b. s.', 'play fox news')
        >>> assert has_conflict_keywords('what is the weather today', 'what is the weather tomorrow')
        >>> assert has_conflict_keywords("play enemy", 'open music quiz')
        >>> assert has_conflict_keywords("play youtube", 'play n. b. c. news')
        >>> assert has_conflict_keywords('get me c. n. b. c. from tunein', 'play msnbc from tunein')
        >>> assert has_conflict_keywords('play music by l. o. f. i. z. e.', 'play music by l. o. f. i. v. e.')
        >>> assert not has_conflict_keywords("turn off my room", 'turn off playroom', language=Languages.German, always_include_en=False)
        >>> assert not has_conflict_keywords("turn off play room", 'turn off playroom')
        >>> assert not has_conflict_keywords("play quiz game", 'play name quiz')
        >>> assert not has_conflict_keywords('what is the weather today tomorrow', 'what is the weather tomorrow')
        >>> assert not has_conflict_keywords('what is the weather now', 'what is the weather today')
        >>> assert has_conflict_keywords("turn off my room", 'turn off playroom', language=Languages.German, always_include_en=True)
        >>> assert has_conflict_keywords("schalte mein zimmer aus", 'spielzimmer ausschalten', language=Languages.German)
    """
    return strex.has_conflict_keywords(
        str1=_remove_acronym_periods_and_spaces(utterance1),
        str2=_remove_acronym_periods_and_spaces(utterance2),
        conflict_keywords=conflict_keywords,
        language=language if language in conflict_keywords else 'en',
        always_include_en=always_include_en,
        **kwargs
    )


def has_conflict_keywords_udf(
        utterance1_colname,
        utterance2_colname,
        conflict_keywords=CONFLICT_KEYWORD_PAIRS,
        language='en',
        always_include_en=True,
        **kwargs
):
    return udf(
        partial(
            has_conflict_keywords,
            conflict_keywords=conflict_keywords,
            language=language,
            always_include_en=always_include_en,
            **kwargs
        ),
        returnType=BooleanType()
    )(utterance1_colname, utterance2_colname)


def slot_has_conflict_keywords(
        nlu_hypothesis1: str,
        nlu_hypothesis2: str,
        conflict_keywords: Mapping = SLOT_SPECIFIC_CONFLICT_KEYWORD_PAIRS,
        language: Languages = Languages.English,
        always_include_en=True,
        **kwargs
):
    if not nlu_hypothesis1 or not nlu_hypothesis2:
        return False

    slots1 = get_slots_dict_from_hypothesis(nlu_hypothesis1)
    if not slots1:
        return False
    slots2 = get_slots_dict_from_hypothesis(nlu_hypothesis2)
    if not slots2:
        return False

    slot_types_to_check = set(
        filter(
            lambda x: x in slots2 and contains_any(x, conflict_keywords.keys()),
            slots1.keys()
        )
    )

    if not slot_types_to_check:
        return False

    return any(
        any(
            strex.has_conflict_keywords(
                slots1[slot_type], slots2[slot_type],
                conflict_keywords=_conflict_tokens,
                language=language if language in _conflict_tokens else Languages.English,
                always_include_en=always_include_en,
                **kwargs
            )
            for slot_type_keyword, _conflict_tokens in conflict_keywords.items()
            if slot_type_keyword in slot_type
        )
        for slot_type in slot_types_to_check
    )


def slot_has_conflict_keywords_udf(
        nlu_hypothesis1_colname: str,
        nlu_hypothesis2_colname: str,
        conflict_keywords: Mapping = SLOT_SPECIFIC_CONFLICT_KEYWORD_PAIRS,
        language: Languages = Languages.English,
        always_include_en=True,
        **kwargs
):
    return udf(
        partial(
            slot_has_conflict_keywords,
            conflict_keywords=conflict_keywords,
            language=language,
            always_include_en=always_include_en,
            **kwargs
        ),
        returnType=BooleanType()
    )(nlu_hypothesis1_colname, nlu_hypothesis2_colname)


if __name__ == '__main__':
    assert slot_has_conflict_keywords(
        nlu_hypothesis1='Weather|GetWeatherDetailsIntent|Date:today',
        nlu_hypothesis2='Weather|GetWeatherDetailsIntent|Date:tomorrow'
    )
    assert slot_has_conflict_keywords(
        nlu_hypothesis1='Weather|GetWeatherDetailsIntent|CityName:seattle uptown',
        nlu_hypothesis2='Weather|GetWeatherDetailsIntent|CityName:seattle downtown'
    )
    assert not slot_has_conflict_keywords(
        nlu_hypothesis1='Weather|GetWeatherDetailsIntent|Location:seattle',
        nlu_hypothesis2='Weather|GetWeatherDetailsIntent|Location:seattle west'
    )
    assert slot_has_conflict_keywords(
        nlu_hypothesis1='Global|SetPreferenceIntent|Value:this week',
        nlu_hypothesis2='Global|SetPreferenceIntent|Value:next week'
    )
    assert not slot_has_conflict_keywords(
        nlu_hypothesis1='Global|SetPreferenceIntent|Value:this week',
        nlu_hypothesis2='Global|SetPreferenceIntent|Value:this day'
    )

    assert not slot_has_conflict_keywords(
        nlu_hypothesis1='Global|SetPreferenceIntent|Value:esta semana',
        nlu_hypothesis2='Global|SetPreferenceIntent|Value:próxima semana'
    )
    assert slot_has_conflict_keywords(
        nlu_hypothesis1='Global|SetPreferenceIntent|Value:esta semana',
        nlu_hypothesis2='Global|SetPreferenceIntent|Value:próxima semana',
        language=Languages.Spanish
    )
    assert not slot_has_conflict_keywords(
        nlu_hypothesis1='Global|SetPreferenceIntent|Value:this week',
        nlu_hypothesis2='Global|SetPreferenceIntent|Value:this day',
        language=Languages.Spanish,
        always_include_en=False
    )
