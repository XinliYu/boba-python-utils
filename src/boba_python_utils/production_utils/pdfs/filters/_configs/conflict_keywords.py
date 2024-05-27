from boba_python_utils.general_utils.nlp_utility.common import Languages

# region locations
_KEYWORDS_HOUSE_LOCATIONS = (
    'bath room',
    'bed room',
    'family room',
    'den',
    'living room',
    'master room',
    'kitchen',
    'play room',
    'dining room',
    'study room',
    ('lobby', 'lounge'),
    'elevator',
    ('garbage', 'garbage room'),
    'outside',
    "'s room",
    ('this room', 'my room'),
    'everywhere'
)

_KEYWORDS_HOUSE_LOCATIONS_DE = (
    ('bad', 'badezimmer'),
    'schlafzimmer',
    'wohnzimmer',
    ('höhle', 'hohle'),
    # 'wohnzimmer',
    'hauptzimmer',
    ('küche', 'kuche'),
    'spielzimmer',
    'esszimmer',
    'comedor',
    ('empfangshalle', 'salon'),
    'aufzug',
    ('müll', 'mull', 'müll raum', 'mull raum'),
    ('außen', 'auben'),
    "s zimmer",
    ('dieser raum', 'mein zimmer')
)

_KEYWORDS_HOUSE_LOCATIONS_ES = (
    ('baño', 'bao'),
    ('dormitorio', 'recámara', 'recamara', 'alcoba'),
    'cuarto familiar',
    ('guarida', 'estudio', 'madriguera'),
    ('sala', 'el living'),
    'habitación principal',
    'cocina',
    ('juegos', 'cuarto de jugar'),
    'estudio',
    ('vestíbulo', 'pasillo', 'salón', 'salon'),
    ('ascensor', 'elevador'),
    ('basura', 'cuarto de basura'),
    ('fuera', 'patio delantero'),
    # "cuarto de",
    ('esta habitación', 'esta habitacion', 'mi habitación', 'mi habitacion')
)
# endregion

# region weekdays
_KEYWORDS_TODAY = ('today', 'now')
_KEYWORDS_TODAY_DE = ('heute', 'heutzutage', 'gegenwärtig', 'gegenwartig', 'jetzt')
_KEYWORDS_TODAY_ES = ('este dia', 'hoy', 'ahora', 'actualmente', 'entonces')
_KEYWORDS_WEEKDAYS = (
    'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday',
    _KEYWORDS_TODAY
)
_KEYWORDS_WEEKDAYS_DE = (
    'montag', 'dienstag', 'mittwoch', 'donnerstag', 'freitag', 'samstag', 'sonntag',
    _KEYWORDS_TODAY_DE
)
_KEYWORDS_WEEKDAYS_ES = (
    'lunes', 'martes', ('miércoles', 'miercoles'), 'jueves', 'viernes', ('sábado', 'sabado'), 'domingo',
    _KEYWORDS_TODAY_ES
)
# endregion

# region color
_KEYWORDS_COLOR = (
    'red',
    ('blue', 'bluish'),
    ('green', 'greenish'),
    ('yellow', 'yellowish'),
    ('black', 'dark'),
    'white',
    ('pink', 'rose'),
    ('purple', 'violet'),
    'brown',
    ('gold', 'golden'),
    'grey',
    'rainbow',
    'teal',
    'vinyl',
    'sapphire'
)
_KEYWORDS_COLOR_DE = (
    'rot',
    ('blau', 'bläulich', 'blaulich'),
    ('grün', 'grun', 'grünlich', 'grunlich'),
    ('gelb', 'gelblich'),
    ('schwarz', 'dunkel', 'finster'),
    ('weiß', 'weib'),
    ('gold', 'golden'),
    'rosa',
    'violett',
    'braun',
    'grau',
    'regenbogen',
    ('blaugrün', 'blaugrun'),
    'vinyl'
    'saphir'
)

_KEYWORDS_COLOR_ES = (
    ('roja', 'rojo'),
    ('azul', 'azulado', 'azulada'),
    'verde',
    ('amarilla', 'amarillo'),
    ('negra', 'negro', 'oscura', 'oscuro'),
    ('blanca', 'blanco'),
    ('rosado', 'rosada'),
    ('morada', 'morado', 'violeta', 'violeto'),
    ('dorado', 'dorada', 'oro'),
    ('marrón', 'marron'),
    'gris',
    ('arcoíris', 'arcoiris'),
    'verde azulado',
    ('vinilo', 'vinila'),
    ('zafiro', 'zafira')
)
# endregion

# region animal
_KEYWORDS_ANIMAL = ('fish', 'bird', 'dog', 'cat', 'donkey', 'shark', 'duck', 'cricket')

_KEYWORDS_ANIMAL_ES = (
    'pez', ('pájaro', 'pajaro'), ('perra', 'perro'), ('gata', 'gato')
)
# endregion

# region devices
_KEYWORDS_DEVICES = (
    'fan',
    ('light', 'lamp'),
    ('air con', 'air conditioner', 'a. c.'),
    'desk',
    'shutter',
    ('tv', 'television'),
    ('alarm', 'timer', 'reminder', 'appointment'),
    ('display', 'monitor', 'screen'),
    'headphone',
    ('soundbar', 'sound bar'),
    ('iphone', 'phone'),
    ('car', 'vehicle', 'tesla'),
    'oven',
    'water',
    'garbage',
    'xbox',
    'camera',
    'echo',
    'kindle',
    'audible',
    'ipad',
    'bluetooth',
    'noise',
)

_KEYWORDS_DEVICES_ES = (
    'ventilador',
    ('ligera', 'ligero', 'lámpara', 'lampara'),
    'escritorio',
    ('tv', 'televisión', 'television'),
    ('alarma', 'temporizador', 'recordatorio', 'cita', 'nombramiento'),
    ('monitora', 'monitor'),
    'auricular',
    ('barra de sonido', 'barra sonido'),
    ('teléfono', 'telefono', 'iphone'),
    ('coche', 'vehículo', 'vehiculo', 'tesla'),
    'horno',
    'agua',
    'basura',
)
# endregion

# region gender
_KEYWORDS_GENDER = ('male', 'female')
_KEYWORDS_GENDER_ES = (('masculina', 'masculino'), ('femenina', 'femenino'))
# endregion

_KEYWORDS_SOUND = (
    'snow',
    ('fire', 'fireplace'),
    'waterfall',
    ('thunder', 'thunderstorm'),
    ('rainforest', 'forest', 'tree'),
    ('ocean', 'wave', 'beach'),
    ('airplane', 'airport'),
    ('rain', 'shower'),
    ('wind', 'tornado'),
    'winter',
    'summer',
    'spring',
    *_KEYWORDS_ANIMAL,
    *_KEYWORDS_COLOR,
    *_KEYWORDS_DEVICES
)

_KEYWORDS_CHANNELS = ('pbs', 'cbs', 'tbs', 'msnbc', 'abc', 'nbc', 'bbc', 'cnn', 'fox', 'itv', 'cnbc', 'lbc', 'npr')

_KEYWORDS_APP = (
    'netflix',
    'spotify',
    'audible',
    'xbox',
    'tunein',
    ('tube', 'youtube'),
    'hulu',
    'hbo max',
    'disney',
    ('xfinity', 'comcast'),
    'tivo',
    'satellite',
    'cable',
    'dish',
    'prime',
    'shopping',
    'deal',
    'home',
    'mcdonald',
    'starbuck',
    'tubi',
    'vudu',
    'peacock',
    'freevee',
    'apple',
    'google',
    'roku',
    'dyson',
    *_KEYWORDS_DEVICES,
    *_KEYWORDS_CHANNELS
)

_KEYWORDS_GENRE = (
    'rock',
    'rap',
    'electronic',
    'metal',
    'pop',
    'r. and b.',
    'jazz',
    'acoustic',
    'dance',
    'piano',
    'country',
    ('classic', 'classical'),
    ('kid', 'baby'),
    ('sleep', 'sleepy', 'peaceful', 'sooth', 'comfort', 'relax', 'night'),
    ('spa', 'bath', 'dance'),
    ('pet', 'dog', 'cat', 'animal'),
    'clarinet',
    ('oboe', 'sax', 'saxophone'),
    ('battle', 'war', 'patriot'),
    'violin',
    'guitar',
    'flute',
    'drum',
    'trumpet',
    'harp',
    ('latin', 'latino', 'tropical'),
)

_KEYWORDS_SIGN = (
    ('aries', 'ram'), ('taurus', 'bull'), ('gemini', 'twins'), ('cancer', 'crab'),
    ('leo', 'lion'), ('virgo', 'virgin'), ('libra', 'balance'), ('scorpius', 'scorpion'),
    ('sagittarius', 'archer'), ('capricornus', 'goat'),
    ('aquarius', 'water bearer', 'bearer'), ('pisces', 'fish')
)

CONFLICT_KEYWORD_PAIRS = {
    Languages.English: [
        # region opposite meaning
        ('save', 'skip'),
        ('left', 'right'),
        ('up', 'down'),
        ('indoor', 'outdoor'),
        ('inside', 'outside'),
        ('front', 'back'),
        ('front', 'side'),
        ('morning', 'night'),
        ('light', 'dark'),
        ('light', 'heavy'),
        (('earliest', 'first', 'preceding', 'pilot'), ('last', 'latest'), 'previous', 'next'),
        ("rise", "set"),
        (("bad", "shit"), "good"),
        ("high", "low"),
        ("upstairs", "downstairs"),
        ("boyfriend", "girlfriend"),
        ("north", "south"),
        ('her', 'his'),
        ('weather', 'news', ('time', 'hour')),
        (("mother", "mom", "mommy"), ("father", "dad", "daddy", "papa"), 'grandpa', 'grandma', 'uncle', 'aunt', 'boss'),
        ('cold', 'hot'),
        (('large', 'big'), 'small'),
        (('today', 'now'), 'tomorrow', 'weekend', 'weekday'),
        # endregion
        # region on/off conflict
        ("pause", "unpause"),
        ("arm", "disarm"),
        ("mute", "unmute"),
        ("start", "stop"),
        (('open', 'start', 'resume', 'play', 'replay', 'unlock', 'show'), ('stop', 'pause', 'close', 'lock')),
        ('off', ('start', 'open', 'on')),
        ('turn off', 'turn up'),
        # endregion
        # region other confusions
        (('moon', 'moon light'), ('sun', 'sun light', 'sunshine')),
        ('home page', 'app'),
        _KEYWORDS_SOUND,
        ('spanish', 'english', 'french', 'italian', 'chinese', 'japanese', 'korean'),
        ("make", "spell"),
        ("poop", "pee"),
        ('how', ('when', 'where'), 'who'),
        _KEYWORDS_CHANNELS,
        _KEYWORDS_DEVICES,
        _KEYWORDS_HOUSE_LOCATIONS,
        _KEYWORDS_COLOR,
        _KEYWORDS_WEEKDAYS,
        _KEYWORDS_ANIMAL,
        _KEYWORDS_APP,
        ('nba', 'nfl'),
        (('song', 'music'), ('noise', 'sound'), 'poem', 'fact', 'joke', 'game'),
        (
            ('play', 'sing', 'turn on', 'put on'),
            ('what', 'who', 'do you'),
            'change',
            'repeat',
            'quiz',
            ('like', 'thumb up', 'thumbs up'),
            ('unlike', 'thumb down', 'thumbs down'),
            'fart'
        ),
        _KEYWORDS_SIGN,
        ('episode', 'season', ('song', 'music')),
        ('cocomelon', 'pokemon', 'pinkfong', 'ping pong'),
        ('parking', 'station'),
        _KEYWORDS_GENRE,
        ('web', 'weather'),
        ('rating', 'weather'),
        ('fireplace', 'reading'),
        ('sump', 'shelf'),
        ('baby t. rex', 'baby sleep', 'baby shark'),
        ('facetime', 'call'),
        ('lofive', 'lofize'),
        ('fairy tale story', 'bedtime story'),
        ('basketball', 'baseball'),
        ('bruno', 'anymore')
        # endregion
    ],
    Languages.German: [
        _KEYWORDS_HOUSE_LOCATIONS_DE,
        _KEYWORDS_WEEKDAYS_DE,
        _KEYWORDS_COLOR_DE
    ],
    Languages.Spanish: [
        _KEYWORDS_HOUSE_LOCATIONS_ES,
        _KEYWORDS_WEEKDAYS_ES,
        _KEYWORDS_COLOR_ES,
        _KEYWORDS_ANIMAL_ES,
        _KEYWORDS_DEVICES_ES
    ]
}

_SLOT_SPECIFIC_CONFLICT_KEYWORD_PAIRS_DATETIME_EN = (
    ('a.m.', 'p.m.'),
    (_KEYWORDS_TODAY, 'tonight', 'tomorrow', 'weekend', 'weekday'),
    ('this', 'next')
)
_SLOT_SPECIFIC_CONFLICT_KEYWORD_PAIRS_DATETIME_DE = (
    (_KEYWORDS_TODAY_DE, ('heute abend', 'heute nacht', 'diese nacht'), 'morgen', 'wochenende', 'wochentag'),
    (('dieser', 'dies'), ('nächste', 'nachste'))
)
_SLOT_SPECIFIC_CONFLICT_KEYWORD_PAIRS_DATETIME_ES = (
    (
        _KEYWORDS_TODAY_ES,
        ('mañana', 'manana'),
        ('fin de semana', 'fin semana'), 'laborable'
    ),
    (
        ('esta', 'esto'),
        ('siguiente', 'próximo', 'proximo', 'próxima', 'proxima', 'entrante')
    )
)

SLOT_SPECIFIC_CONFLICT_KEYWORD_PAIRS = {
    'Time': {
        'en': (
            ('a.m.', 'p.m.'),
        )
    },
    'Setting': {
        'en': _SLOT_SPECIFIC_CONFLICT_KEYWORD_PAIRS_DATETIME_EN,
        'de': _SLOT_SPECIFIC_CONFLICT_KEYWORD_PAIRS_DATETIME_DE,
        'es': _SLOT_SPECIFIC_CONFLICT_KEYWORD_PAIRS_DATETIME_ES
    },
    'Value': {
        'en': _SLOT_SPECIFIC_CONFLICT_KEYWORD_PAIRS_DATETIME_EN,
        'de': _SLOT_SPECIFIC_CONFLICT_KEYWORD_PAIRS_DATETIME_DE,
        'es': _SLOT_SPECIFIC_CONFLICT_KEYWORD_PAIRS_DATETIME_ES
    },
    'Date': {
        'en': _SLOT_SPECIFIC_CONFLICT_KEYWORD_PAIRS_DATETIME_EN,
        'de': _SLOT_SPECIFIC_CONFLICT_KEYWORD_PAIRS_DATETIME_DE,
        'es': _SLOT_SPECIFIC_CONFLICT_KEYWORD_PAIRS_DATETIME_ES
    },
    'Location': (
        ('east', 'west', 'north', 'south'),
        (('upper', 'uptown'), ('lower', 'downtown'))
    ),
    'City': (
        ('east', 'west', 'north', 'south'),
        (('upper', 'uptown'),
         ('lower', 'downtown'))
    ),
    'Gender': {
        'en': (
            _KEYWORDS_GENDER,
        ),
        'es': (
            _KEYWORDS_GENDER_ES,
        )
    },
    'OnType': (
        ('timer', 'reminder'),
    )
}
