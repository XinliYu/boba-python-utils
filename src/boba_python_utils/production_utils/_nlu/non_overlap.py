import re
import warnings
try:
    from Levenshtein import ratio
except:
    warnings.warn(f"Failed to load 'Levenshtein' module")
from boba_python_utils.general_utils.nlp_utility.punctuations import remove_acronym_periods_and_spaces
import boba_python_utils.general_utils.nlp_utils as nlpu

# region constants

NON_OVERLAP_IGNORE_WORDS_ESSENTIAL = {
    'hey',
    'please',
    'alexa',
    'echo',
    'music',
    'song',
    'songs',
    'play',
    'replay',
    'turn',
    'show',
    'shuffle',
    'shuffling',
    'put',
    'set',
    'give',
    'tell',
    'open',
    'shut',
    'close',
    'stop',
    'disconnect',
    'switch',
    'i',
    'my',
    'you',
    'your',
    'us',
    'it',
    'me',
    'a',
    'an',
    'the',
    'this',
    'that',
    'best',
    'by',
    'from',
}
NON_OVERLAP_IGNORE_WORDS_EXTRA = {
    'music',
    'playlist',
    'song',
    'songs',
    'movie',
    'movies',
    'video',
    'videos',
    'sound',
    'sounds',
    'recipe',
    'recipes',
    'message',
    'messages',
    'album',
    'temperature',
    'weather',
    'forecast',
    'at',
    'in',
    'on',
    'out',
    'to',
    'for',
    'of',
    'off',
    'up',
    'with',
    'about',
    'as',
    'or',
    'and',
    'yes',
    'no',
    'just',
    'can',
    'is',
    'do',
    "doesn't",
    "it's",
    'what',
    "what's",
    'whats',
    'when',
    'where',
    'how',
    'too',
    'more',
    "i'm",
    "ain't",
}
NON_OVERLAP_REPLACEMENT = {
    'songs by': 'songs',
    'music by': 'music',
    'room': '',
    'alexa': '',
    'echo': '',
    "'s": '',
    "look up recipes": '',
    "the day": 'today',
    "alarm": "timer",
    "super": 'super ',
}

NON_OVERLAP_REPLACEMENT_LEGACY = {
    'songs by': 'songs',
    'music by': 'music',
    'room': '',
    'alexa': '',
    'echo': '',
    'x': 'ks',
    "'s": '',
    'ic ': 'yc ',
    'ing': 'innn',
    "look up recipes": '',
    "the day": 'today',
    "ye ": "the ",
    "alarm": "timer",
    "super": 'super ',
}

VOWEL_LETTERS = 'aeiou'
VOWEL_LETTER_SET = set(VOWEL_LETTERS)

NON_OVERLAP_IGNORE_WORDS = NON_OVERLAP_IGNORE_WORDS_ESSENTIAL.union(NON_OVERLAP_IGNORE_WORDS_EXTRA)
# endregion

_word_sep_pattern = re.compile(r'[\s\-]')
_by_from_pattern = re.compile(r'\b(by|from)\b')
sdx = nlpu.SoundexD()


def _remove_ending_s(s):
    if len(s) > 3 and s[-1] == 's' and s[-2] not in 'aeious':
        return s[:-1]
    return s


def _is_two_letter_with_a_vowel(s):
    return len(s) == 2 and s[1] in 'aeiou'


def _likely_short_replacement_error(text1, text2):
    return (
            ('list' in text1 or 'list' in text2)
            or ('how do you say ' in text1 or 'how do you say ' in text2)
            or ('what is ' in text1 or 'what is ' in text2)
            or (' radio' in text1 or ' radio' in text2)
    )


def phonetic_transform_for_single_token_pair(token1, token2):
    if 'oo' in token2:
        token1 = token1.replace('ju', 'droo')
    elif 'oo' in token1:
        token2 = token2.replace('ju', 'droo')
    if token1.startswith('un') and token2.startswith('un'):
        return token1[2:], token2[2:]
    if token1.endswith('est') and token2.endswith('est'):
        return token1[:-3], token2[:-3]
    elif token1.endswith('er') and token2.endswith('er'):
        return token1[:-2], token2[:-2]
    if token1 and (token1[0] == 'a' or token1[0] == 'i'):
        token1 = 'e' + token1[1:]
    if token2 and (token2[0] == 'a' or token2[0] == 'i'):
        token2 = 'e' + token2[1:]
    return token1, token2


def _get_tokens_for_utterance_non_overlap(utt1, utt2):
    utt1_after_by_tks = utt2_after_by_tks = set()
    utt1_match, utt2_match = _by_from_pattern.search(utt1), _by_from_pattern.search(utt2)
    if utt1_match is None and utt2_match is not None:
        utt2_after_by_tks = set(_word_sep_pattern.split(string=utt2[utt2_match.end():]))
        utt2 = utt2[: utt2_match.start()]
    elif utt1_match is not None and utt2_match is None:
        utt1_after_by_tks = set(_word_sep_pattern.split(string=utt1[utt1_match.end():]))
        utt1 = utt1[: utt1_match.start()]
    return (
        [x for x in _word_sep_pattern.split(string=utt1.strip()) if x not in utt2_after_by_tks],
        [x for x in _word_sep_pattern.split(string=utt2.strip()) if x not in utt1_after_by_tks]
    )


def get_utterance_non_overlap(utt1, utt2):
    # TODO: improvement with new utilities
    utt1 = remove_acronym_periods_and_spaces(utt1)
    utt2 = remove_acronym_periods_and_spaces(utt2)

    for src, trg in NON_OVERLAP_REPLACEMENT.items():
        utt1 = utt1.replace(src, trg)
        utt2 = utt2.replace(src, trg)

    tks1, tks2 = _get_tokens_for_utterance_non_overlap(utt1, utt2)
    tks1_unique = set(tks1) | set(_tk + 's' for _tk in tks1)
    tks2_unique = set(tks2) | set(_tk + 's' for _tk in tks2)

    no_tks1 = [
        x
        for x in tks1
        if x not in tks2_unique
           and (x + 's' not in tks2_unique)
           and x not in NON_OVERLAP_IGNORE_WORDS_ESSENTIAL
    ]
    no_tks2 = [
        x
        for x in tks2
        if x not in tks1_unique
           and (x + 's' not in tks1_unique)
           and x not in NON_OVERLAP_IGNORE_WORDS_ESSENTIAL
    ]
    if len(no_tks1) != 1 or len(no_tks2) != 1:
        no_tks1 = filter(lambda x: x not in NON_OVERLAP_IGNORE_WORDS_EXTRA, no_tks1)
        no_tks2 = filter(lambda x: x not in NON_OVERLAP_IGNORE_WORDS_EXTRA, no_tks2)

    no_tks1 = list(map(_remove_ending_s, no_tks1))
    no_tks2 = list(map(_remove_ending_s, no_tks2))

    if len(no_tks1) == 1 and len(no_tks2) == 1:
        return no_tks1[0], no_tks2[0]
    else:
        return (
            ' '.join(no_tks1).strip(),
            ' '.join(no_tks2).strip()
        )


def utterance_non_overlap_filter(utterance1, utterance2):
    if 'time' in utterance1 and 'weather' in utterance2:
        return False
    if (
            utterance1.startswith('what day') or utterance1.startswith('what song')
    ) and utterance2.startswith('what time'):
        return False
    if (
            utterance2
            in (
            'turn off',
            'turn on',
            'power on',
            'power off',
            'what time is it',
            'what is the time',
            "what's today's date",
            'what is the date',
    )
            and utterance1.startswith(utterance2)
    ):
        return False
    if utterance1 in ('turn on ' + utterance2, 'turn off ' + utterance2):
        return False

    non_overlap1, non_overlap2 = get_utterance_non_overlap(utterance1, utterance2)
    single_word = ' ' not in non_overlap1 and ' ' not in non_overlap2
    if single_word:
        if (non_overlap1, non_overlap2) in {
            ('left', 'right'),
            ('right', 'left'),
            ('up', 'down'),
            ('down', 'up'),
            ('off', 'on'),
            ('stop', 'start'),
            ('indoor', 'outdoor'),
            ('outdoor', 'indoor'),
        }:
            return False
        non_overlap1, non_overlap2 = phonetic_transform_for_single_token_pair(
            non_overlap1, non_overlap2
        )
    if not non_overlap1 and non_overlap2:
        if 'play' in utterance1 and 'play' not in utterance2 and len(non_overlap2) < 6:
            return False

    if non_overlap1 and non_overlap2:
        non_overlap1_init_sounds = set(nlpu.get_init_sound_tag(_tk) for _tk in non_overlap1.split())
        non_overlap2_init_sounds = set(nlpu.get_init_sound_tag(_tk) for _tk in non_overlap2.split())
        no_init_sound_overlap = (
                len(non_overlap1_init_sounds.intersection(non_overlap2_init_sounds)) == 0
        )
        s1_th_add = 0
        # s1_th_add = 0.1 if no_init_sound_overlap else 0

        vowel_extra1, vowel_extra2, vowel_overlap = nlpu.get_vowel_diff(non_overlap1, non_overlap2)
        len_vowel_extra1, len_vowel_extra2 = len(vowel_extra1), len(vowel_extra2)
        len_non_overlap1, len_non_overlap2 = len(non_overlap1), len(non_overlap2)
        min_non_overlap_len = min(len_non_overlap1, len_non_overlap2)
        if single_word:
            if min_non_overlap_len < 4:
                s1_th_add += 0.2
            if min_non_overlap_len < 5:
                s1_th_add += max(len_vowel_extra1, len_vowel_extra2) * 0.1

        likely_short_replacement_error = _likely_short_replacement_error(utterance1, utterance2)
        s1 = ratio(non_overlap1, non_overlap2)
        text_overlap_len_ratio1 = len(utterance1) / len_non_overlap1
        text_overlap_len_ratio2 = len(utterance2) / len_non_overlap2

        if (
                (len_non_overlap1 <= 2 and len_non_overlap2 <= 2)
                or (len_non_overlap1 == 1 or len_non_overlap2 == 1)
                or (
                len_non_overlap1 + len_non_overlap2 <= 7
                and abs(len_non_overlap1 - len_non_overlap2) == 1
        )
                and not likely_short_replacement_error
        ):
            return True
        elif len_non_overlap1 > 2 or len_non_overlap2 > 2:
            phoneme1 = sdx.phonetics(non_overlap1)
            phoneme2 = sdx.phonetics(non_overlap2)
            if phoneme1 and phoneme2:
                initial_vowel_diff1 = (non_overlap1[0] in 'aeiou') != (non_overlap2[0] in 'aeiou')
                if (
                        len(phoneme1) == 2
                        and len(phoneme2) == 2
                        and (
                        phoneme1[0] == phoneme2[0]
                        or (phoneme1[1] == phoneme2[1])
                        and phoneme1[1] != '7'
                )
                        and (not single_word or not initial_vowel_diff1)
                ):
                    return True

                initial_vowel_diff2 = nlpu.init_sound_diff(non_overlap1, non_overlap2)
                phoneme1_set, phoneme2_set = set(phoneme1), set(phoneme2)
                s2 = ratio(phoneme1, phoneme2)
                max_phoneme_len = max(len(phoneme1_set), len(phoneme2_set))
                s3 = len(phoneme1_set.intersection(phoneme2_set)) / max_phoneme_len
                if max_phoneme_len >= 5:
                    s3 *= 0.8
                if s1 == 0 and s2 == 0 and s3 == 0:
                    return False

                s1_th_add += 0.1 if (s2 < 0.45 or s3 < 0.4) else 0
                ending_er_diff = (non_overlap1.endswith('er') or non_overlap1.endswith('or')) != (
                        non_overlap2.endswith('er') or non_overlap2.endswith('or')
                )
                ending_sound_diff = nlpu.ending_sound_diff(non_overlap1, non_overlap2)
                if ending_sound_diff:
                    s1_th_add += 0.05

                if (len(phoneme1_set) > 4 and len(phoneme2_set) > 4) and s2 < 0.8 and s3 < 0.8:
                    s1_th_add += 0.1
                    s3_th_add = 0.05
                else:
                    s3_th_add = 0

                if single_word and not initial_vowel_diff2:
                    phoneme_set_intersection = phoneme2_set.intersection(phoneme1_set)
                    if (
                            len(phoneme_set_intersection) in (len(phoneme2_set), len(phoneme1_set))
                            and s2 >= 0.5
                            and s3 >= 0.5
                    ):
                        s1_th_add -= 0.05

                    if (
                            likely_short_replacement_error
                            and (text_overlap_len_ratio1 < 3 or text_overlap_len_ratio2 < 3)
                            and s1 < 0.4
                    ):
                        return False
                    if (
                            len(non_overlap1) < 8
                            and len(non_overlap2) < 8
                            and text_overlap_len_ratio1 > 4.5
                            and text_overlap_len_ratio2 > 4.5
                            and not likely_short_replacement_error
                    ):
                        return True

                    if non_overlap1[0] == non_overlap2[0] and non_overlap1[0] in 'aeiou':
                        s1_th_add -= 0.15
                    if ending_er_diff:
                        s1_th_add = 0.25 if (len_non_overlap1 < 7 or len_non_overlap2 < 7) else 0.15
                    return s1 > (0.4 + s1_th_add) or s2 > 0.7 or s3 > (0.8 + s3_th_add)
                elif s2 > 0.7 or s3 > (0.8 + s3_th_add):
                    if no_init_sound_overlap:
                        return s1 > (0.45 + s1_th_add)
                    else:
                        return s1 > (0.2 + s1_th_add)
                elif s2 <= 0.5 and s3 <= 0.5:
                    return s1 > 0.55
                elif initial_vowel_diff2:
                    return s1 > (0.55 + s1_th_add)
                else:
                    return s1 > (0.4 + s1_th_add)

        return s1 > 0.3 + s1_th_add
    else:
        return True
