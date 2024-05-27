import re


def _create_word_bounded_number_pattern(word):
    _NUMBERS = 'one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety'
    return re.compile(r'\b{} ({})\b'.format(word, _NUMBERS))


def _turn_pair_has_bound_num_change(text1, text2, bound_word):
    pattern = _create_word_bounded_number_pattern(bound_word)
    match1 = pattern.search(text1)
    match2 = pattern.search(text2)
    bounded_num1 = bounded_num2 = None
    if match1:
        bounded_num1 = match1.group(1)
    if match2:
        bounded_num2 = match2.group(1)
    return bounded_num1 != bounded_num2


def turn_pair_has_bounded_number_change(turn1_utterance, turn2_utterance, bound_words):
    return any(
        _turn_pair_has_bound_num_change(turn1_utterance, turn2_utterance, bound_word)
        for bound_word in bound_words
    )


