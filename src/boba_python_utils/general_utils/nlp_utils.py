import warnings
from itertools import groupby
from typing import Mapping, Dict
import re
import warnings

from boba_python_utils.general_utils.nlp_utility.punctuations import remove_acronym_periods_and_spaces

try:
    from nltk import edit_distance
except:
    warnings.warn("unable to load module 'nltk'")
import boba_python_utils.general_utils.strex as strex


class Text:
    def __init__(
        self,
        text,
        tokenizer=None,
        ignore_acronym_periods_and_spaces=False,
        reconnect_tokens_by_space_as_text=True,
        re_split_tokens_by_space=True,
        lower=False,
    ):
        self.text = (
            remove_acronym_periods_and_spaces(text)
            if ignore_acronym_periods_and_spaces
            else text
        )
        if tokenizer is None:
            self.tokens = self.text.split()
        else:
            self.tokens = _solve_tokens(self.text, tokenizer=tokenizer)

        if lower:
            self.tokens = [x.lower() for x in self.tokens]

        self.bow_tokens = sorted(self.tokens)
        self.bow_text = ' '.join(self.bow_tokens)
        self.token_set = set(self.tokens)

        if reconnect_tokens_by_space_as_text:
            self.text = ' '.join(self.tokens)
        elif lower:
            self.text = self.text.lower()

        if re_split_tokens_by_space:
            self.tokens = self.text.split()
            self.bow_tokens = self.bow_text.split()


def tokenize_with_token_map(text, token_map, tokenizer=None):
    tokens = _solve_tokens(text, tokenizer)
    out = []
    for token in tokens:
        if token in token_map:
            out.append(token_map[token])
        else:
            out.append(token)
    return out


def _solve_tokens(text, tokenizer=None, bow=False):
    if isinstance(text, Text):
        return text.bow_tokens if bow else text.tokens
    else:
        if isinstance(text, str):
            text = text.strip()
            if tokenizer is not None:
                if isinstance(tokenizer, Mapping):
                    token_map = tokenizer
                    tokens = tokenize_with_token_map(text, token_map)
                elif callable(tokenizer):
                    tokens = tokenizer(text)
                else:
                    raise ValueError()
            else:
                tokens = text.split()
        else:
            tokens = text
        if bow:
            tokens = sorted(tokens)
        return tokens


def _solve_text(text, tokenizer=None, bow=False, ignore_spaces=False):
    if isinstance(text, Text):
        out = text.bow_text if bow else text.text
    else:
        out = ('' if ignore_spaces else ' ').join(_solve_tokens(text, tokenizer, bow))
    return out.replace(' ', '') if ignore_spaces else out


def char_edit_distance(
    text1, text2, normalized=False, tokenizer=None, bow=False, ignore_spaces=False
):
    text1, text2 = _solve_text(text1, tokenizer, bow, ignore_spaces), _solve_text(
        text2, tokenizer, bow, ignore_spaces
    )
    dis = edit_distance(text1, text2)
    if normalized:
        return dis / max(len(text1), len(text2))
    else:
        return dis


# region phonemes

VOWEL_LETTERS = {'a', 'e', 'i', 'o', 'u'}


def get_init_sound_tag(s):
    s0 = s[0]
    if s0 == 'w' and len(s) > 1 and s[1] in VOWEL_LETTERS:
        s0 = s[1]
    if s0 == 'a' or s0 == 'u' or s0 == 'o':
        return 1
    elif s0 == 'e' or s0 == 'i':
        return 2
    elif s0 == 's' or s0 == 'x':
        return 3
    elif s0 == 'h' or s0 == 'f' or s.startswith('wh'):
        return 4
    elif s0 == 'l' or s0 == 'm' or s0 == 'n' or s0 == 'r':
        return 5
    elif s0 in 'cgkq':
        return 6
    elif s0 in 'y' or s.startswith('ja'):
        return 7
    elif s0 in 'dtjz':
        return 8
    elif s0 in 'wbp':
        return 9
    else:
        return 0


def get_vowel_diff(s1, s2):
    s1 = set(s1).intersection(VOWEL_LETTERS)
    s2 = set(s2).intersection(VOWEL_LETTERS)
    return (s1 - s2), (s2 - s1), (s1 & s2)


def init_sound_diff(text1, text2):
    tag1 = get_init_sound_tag(text1)
    tag2 = get_init_sound_tag(text2)
    if (text1[0] == 'e' and text2[0] == 'a') or (text1[0] == 'a' and text2[0] == 'e'):
        return False
    if 'oo' in text1[:3]:
        return tag2 != 1 and tag1 != tag2
    if 'oo' in text2[:3]:
        return tag1 != 1 and tag1 != tag2
    return tag1 != tag2


def ending_sound_diff(text1, text2):
    if not text1 or not text2:
        return True

    s0 = text1[-1]
    s1 = text2[-1]
    return s0 in 'aeiou' and s1 in 'aeiou' and s0 != s1


class SoundexD:
    INDEX_FOR_Y = '7'

    def __init__(self):
        self.translations = self._translate_soundex(
            'AEIOUYWHBPFVCSKGJQXZDTLMNR'.lower(), '00000700111122222222334556'
        )

    def phonetics(self, word: str) -> str:
        """
        Gets the soundex translation of a given word. Applies to English only.
        For example, "service" will be translated into a sequent of numbers into "2612".

        Args:
            word: the input word

        Returns: a string of numbers representing the soundex transformation of the given word
        :type word: object

        """
        if word[-1] == 'y' and len(word) > 1 and word[-2] in VOWEL_LETTERS:
            word = word[:-1]
        return self._squeeze(re.sub(r'[^a-z]', r'', word).translate(self.translations))

    def _translate_soundex(self, src: str, trg: str, drop='0') -> Dict[str, str]:
        if drop:
            return {ord(x): (ord(y) if y != drop else None) for x, y in zip(src, trg)}
        else:
            return {ord(x): ord(y) for x, y in zip(src, trg)}

    def _squeeze(self, s: str) -> str:
        return ''.join(x[0] for x in groupby(s))


# endregion
