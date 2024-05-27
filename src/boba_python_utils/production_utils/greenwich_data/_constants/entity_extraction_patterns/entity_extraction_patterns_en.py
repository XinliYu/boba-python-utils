from boba_python_utils.string_utils.regex import RegexFactory, RegexFactoryItem

EN_QUERY_COMMON_CONTROL_VERBS = ('stop', 'resume', 'continue')

EN_QUERY_COMMON_ACTION_VERBS = ('playing', 'replay', 'shuffle play', 'play', 'sing', 'hear', 'shuffling', 'shuffle', 'search', 'go to', 'show me', 'show', 'find me', 'find', 'turn back on', 'turn on', 'turn off', 'turn', 'put on', 'put', 'tune to', 'open', 'remove', 'delete')

_en_query_prefix = r'(?:hey\s)?(?:(?:alexa|echo|computer|ziggy|amazon|siri|google)\s)?(?:(?:can you|could you|i would like to|i\'d like to)\s)?'
_en_query_prefix_entertainment = rf'{_en_query_prefix}(?:(?:{"|".join(EN_QUERY_COMMON_CONTROL_VERBS)})\s)?'
_en_entity_constraint_prefix = r'(?:the\s)?(?:new\s)?'
_en_entity_prefix = '(?:the most popular|next|more)?(?:the (.+) version of)?'

_en_query_prefix_music_video_with_verb = rf'{_en_query_prefix_entertainment}(?:(?:{"|".join(EN_QUERY_COMMON_ACTION_VERBS)})\s)(?:(?:{_en_entity_prefix})\s)?'
_en_query_prefix_music_video = fr'{_en_query_prefix_entertainment}(?:(?:{_en_entity_prefix})\s)?'
_en_query_suffix_music_video = r'(?:\s(?:explicit|remix|acoustic|instrumental))?'
_en_query_prefix_book_with_verb = fr'{_en_query_prefix_entertainment}(?:(?:reading|read)\s)(?:(?:{_en_entity_prefix})\s)?'
_en_query_prefix_book = fr'{_en_query_prefix}(?:(?:{_en_entity_prefix})\s)?'

EN_ENTITY_EXTRACTION_PATTERNS = RegexFactory(
    patterns=[
        RegexFactoryItem(
            pattern_prefix=[fr"{_en_query_prefix}(?:(?:show|show me|tell me)\s)?"],
            main_pattern=[
                r"when (?:was|is) (.+)(?:\'s?) birthday",
                "what style of music is (.+)",
                r"who\'s the director of (.+)",
                "who is the director of (.+)",
                r"who\'s the author of (.+)",
                "who is the author of (.+)",
                r"what is (.+)(?:\'s?) (?:new|latest|top) (?:song|album) called",
                r"(?:what is|what\'s) (.+)(?:\'s?)? (?:new|newest|best known|top|best|latest|next|most popular) (?:movie|song)",
                r"(?:(?:how\'s|what\'s|how is|what is)\s)(?:the\s)?(?:weather|temperature|weekly) (?:forecast\s)?(?:(?:going to be|gonna be)\s)?(?:like\s)?(?:for\s)?(?:(?:today|tomorrow)\s)?(?:in|for) (.+)",
                "how do you make (.+) out of (.+)",
                "how do you make (.+)",
                "more about (.+)",
                "about (.+)",
                r"what\'s (.+)"
                "what is (.+)"
            ]
        ),
        RegexFactoryItem(
            pattern_prefix=[_en_query_prefix],
            main_pattern=[
                "turn (?:on|off)? (.+)",
                "turn (.+) (?:on|off)$",
                r"add (.+) to the (?:(?:shopping|grocery|todo|walmart)\s)?(?:list|cart|favorite)",
                r"(?:show|show me|what are) (?:(?:the|some)\s)?(?:recipe|recipes) for (.+)"
            ]
        ),
        RegexFactoryItem(
            pattern_prefix=[_en_query_prefix_music_video_with_verb, _en_query_prefix_music_video],
            main_pattern=[
                "(.+) radio on (.+)",
                "(.+) station on (.+)",
                "(.+) radio",
                "(.+) station",
                "my (.+) playlist",
                "(.+) playlist"

                "(.+) album by (.+) on (.+)",
                f"{_en_entity_constraint_prefix}album (.+) by (.+) on (.+)",
                f"{_en_entity_constraint_prefix}album (.+) from (.+) on (.+)",
                f"{_en_entity_constraint_prefix}music by (.+) from (.+) on (.+)",
                f"{_en_entity_constraint_prefix}songs by (.+) from (.+) on (.+)",
                f"{_en_entity_constraint_prefix}songs from (.+) by (.+) on (.+)",
                f"{_en_entity_constraint_prefix}songs (.+) by (.+) on (.+)",
                f"{_en_entity_constraint_prefix}song (.+) by (.+) on (.+)",
                "from (.+) by (.+) on (.+)",
                "(.+) by (.+) on (.+)",

                "(.+) album by (.+)",
                f"{_en_entity_constraint_prefix}album (.+) by (.+)",
                f"{_en_entity_constraint_prefix}albums from (.+)",
                f"{_en_entity_constraint_prefix}music by (.+) from (.+)",
                f"{_en_entity_constraint_prefix}songs by (.+) from (.+)",
                "from (.+) by (.+)",
                f"{_en_entity_constraint_prefix}songs from (.+) by (.+)",
                f"{_en_entity_constraint_prefix}song (.+) by (.+)",
                f"{_en_entity_constraint_prefix}album (.+)",
                f"{_en_entity_constraint_prefix}music by (.+)",
                f"{_en_entity_constraint_prefix}music from (.+)",
                f"{_en_entity_constraint_prefix}music (.+)",
                f"{_en_entity_constraint_prefix}song called (.+)",
                f"{_en_entity_constraint_prefix}songs from (.+)",
                f"{_en_entity_constraint_prefix}songs by (.+)",
                f"{_en_entity_constraint_prefix}song by (.+)",
                f"{_en_entity_constraint_prefix}song (.+)",
                "(.+) by (.+)",
                "(.+) from (.+)",
                "(.+) music from (.+)",
                "(.+) music (.+)",
                "(.+) for kids"
                "(.+) album",
                "a song called (.+)",
                "(.+) songs",
                "(.+) music",
                "(.+) from (.+)",
                "(.+) playlist on (.+)",
                r"(.+) (?:(?<!(put)\s)(?<!(turn)\s)(?<!(going)\s)(?<!(go)\s)(?<!(play)\s)on) (.+)",

                "a book called (.+) by (.+)",
                f"{_en_entity_constraint_prefix}book called (.+) by (.+)",
                f"{_en_entity_constraint_prefix}book (.+) by (.+) for kids",
                f"{_en_entity_constraint_prefix}book (.+) by (.+)",
                f"{_en_entity_constraint_prefix}book by (.+)",
                f"{_en_entity_constraint_prefix}books by (.+)",
                "a book called (.+)",
                f"{_en_entity_constraint_prefix}book called (.+)",
                f"{_en_entity_constraint_prefix}book (.+)",

                f"{_en_entity_constraint_prefix}(?:video|movie) (.+)",
                f"{_en_entity_constraint_prefix}(?:video|movie) called (.+)",
                f"a (?:video|movie) (.+)",
                "a (?:video|movie) called (.+)",
                "a (?:video|movie) on (.+)",
                "a (?:video|movie) for (.+)",
                f"{_en_entity_constraint_prefix}videos on (.+)",
                f"{_en_entity_constraint_prefix}videos for (.+)"
            ],
            pattern_suffix=[_en_query_suffix_music_video]
        ),
        RegexFactoryItem(
            pattern_prefix=[_en_query_prefix_book_with_verb, _en_query_prefix_book],
            main_pattern=[
                f"{_en_entity_constraint_prefix}book (.+) by (.+) for kids",
                f"{_en_entity_constraint_prefix}book (.+) by (.+)",
                "(.+) for kids"
                f"{_en_entity_constraint_prefix}book by (.+)",
                f"{_en_entity_constraint_prefix}books by (.+)",
                f"{_en_entity_constraint_prefix}book (.+)",
                "(.+) book by (.+)",
                "(.+) book",
                "(.+) by (.+)"
            ]
        ),
        # region fallback patterns
        f"{_en_query_prefix_music_video_with_verb}(.+)",
        f"{_en_query_prefix_book_with_verb}(.+){_en_query_suffix_music_video}",
        RegexFactoryItem(
            pattern_prefix=[_en_query_prefix],
            main_pattern=[
                "show (.+)",
                "search (.+)",
                "show me (.+)",
                "tell me (.+)",
                "did (.+) win tonight"
            ]
        ),
        fr'{_en_query_prefix_entertainment}(?:(?:{_en_entity_prefix})\s)?(.+)'
        # endregion
    ]
)

if __name__ == '__main__':
    from boba_python_utils.spark_utils.specialized.nlp_utility.spark_ner import regex_extract_all_groups, SparkER, PreDefinedErMethods

    self = SparkER(
        spark=None,
        er_text_colname=None,
        er_result_colname=None,
        er_method=PreDefinedErMethods.PreDefinedPatterns,
        er_args=EN_ENTITY_EXTRACTION_PATTERNS
    )

    assert regex_extract_all_groups('can you tell me more about ashford', pattern=self._er_args[1]) == ['ashford']
    assert regex_extract_all_groups('play this is what a heart break feels like on spotify', pattern=self._er_args[1]) == ['this is what a heart break feels like', 'spotify']
    assert regex_extract_all_groups('hey siri play teen things i hate about you by lia kate', pattern=self._er_args[1]) == ['teen things i hate about you', 'lia kate']
    assert regex_extract_all_groups('play the piano version of when till i found you by stephen sanchez', pattern=self._er_args[1]) == ['piano', 'when till i found you', 'stephen sanchez']
    assert regex_extract_all_groups('search we have always lived in a castle', pattern=self._er_args[1]) == ['we have always lived in a castle']
    assert regex_extract_all_groups('play dave matthews live from madison square garden', pattern=self._er_args[1]) == ['dave matthews live', 'madison square garden']
    assert regex_extract_all_groups('can you play o. four thousand times by david crowder again', pattern=self._er_args[1]) == ['o. four thousand times', 'david crowder again']
    assert regex_extract_all_groups('play a spooky scary skeleton remix', pattern=self._er_args[1]) == ['a spooky scary skeleton remix']
    assert regex_extract_all_groups('read the hobbits book', pattern=self._er_args[1]) == ['the hobbits']
    assert regex_extract_all_groups('play the movie loud house', pattern=self._er_args[1]) == ['loud house']
    assert regex_extract_all_groups('play the nas collection', pattern=self._er_args[1]) == ['the nas collection']
    assert regex_extract_all_groups("how's the weather gonna be today in lancaster", pattern=self._er_args[1]) == ['lancaster']
    assert regex_extract_all_groups("what are some recipes for a whole eye of ground", pattern=self._er_args[1]) == ['a whole eye of ground']
    assert regex_extract_all_groups("what's the weather forecast for today in greenville illinois", pattern=self._er_args[1]) == ['greenville illinois']
    assert regex_extract_all_groups("did the san francisco forty niners win tonight", pattern=self._er_args[1]) == ['the san francisco forty niners']
    assert regex_extract_all_groups("what is anthony hopkins next movie", pattern=self._er_args[1]) == ['anthony hopkins']
    assert regex_extract_all_groups("what is the weekly forecast for the finger lakes", pattern=self._er_args[1]) == ['the finger lakes']
    assert regex_extract_all_groups("turn on liza's office", pattern=self._er_args[1]) == ["liza's office"]
    assert regex_extract_all_groups("could you put on the gummy bear song and", pattern=self._er_args[1]) == ['the gummy bear song and']
    assert regex_extract_all_groups("play the i love you song from", pattern=self._er_args[1]) == ['the i love you song from']
    assert regex_extract_all_groups("play we're going on a dragon hunt", pattern=self._er_args[1]) == ["we're going on a dragon hunt"]
    assert regex_extract_all_groups("i'd like to hear music from carrie underwood", pattern=self._er_args[1]) == ["carrie underwood"]
    assert regex_extract_all_groups("play on top of the world", pattern=self._er_args[1]) == ["on top of the world"]
    assert regex_extract_all_groups("what is boy with duke's new song called", pattern=self._er_args[1]) == ['boy with duke']
    assert regex_extract_all_groups("when is olivia rodrigo's birthday", pattern=self._er_args[1]) == ['olivia rodrigo']
    assert regex_extract_all_groups("play romantic comes on", pattern=self._er_args[1]) == ['romantic comes on']


