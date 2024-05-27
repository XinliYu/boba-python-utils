from boba_python_utils.production_utils.common._constants.path import SupportedLocales

IGNORED_HYPOTHESIS_FOR_QUERIES = set()
IGNORED_HYPOTHESIS_FOR_QUERIES_BY_LOCALE = {
    SupportedLocales.EN_US: {
        'Game|LaunchGameIntent|MediaType:game',
        'Game|LaunchGameIntent|MediaType:games',
        'Game|LaunchGameIntent',
        'Game|LaunchGameIntent|ActiveUserTrigger:i|MediaType:game',
        'Game|LaunchGameIntent|ActiveUserTrigger:i|MediaType:games'
    },
    SupportedLocales.EN_GB: {
        'Books|ReadBookIntent|GenreName:bedtime story',
        'Books|ReadBookIntent|GenreName:bedtime',
        'Books|ReadBookIntent|GenreName:bedtime|MediaType:story'
    }
}
IGNORED_HYPOTHESIS_FOR_REWRITES = set()
IGNORED_HYPOTHESIS_FOR_REWRITES_BY_LOCALE = {
    SupportedLocales.EN_US: {
        'Game|LaunchGameIntent|MediaType:game',
        'Game|LaunchGameIntent|MediaType:games',
        'Game|LaunchGameIntent',
        'Game|LaunchGameIntent|ActiveUserTrigger:i|MediaType:game',
        'Game|LaunchGameIntent|ActiveUserTrigger:i|MediaType:games'
    },
    SupportedLocales.EN_GB: {
        'Books|ReadBookIntent|GenreName:bedtime story',
        'Books|ReadBookIntent|GenreName:bedtime',
        'Books|ReadBookIntent|GenreName:bedtime|MediaType:story'
    }
}

CONFLICT_HYPOTHESIS_DROP = [
    ('BookingsAndReservations', 'CreateBookingIntent'),
    ('Shopping', 'SubmitOrderIntent'),
    ('order', ('taxi', 'car')),
    ('Calendar', 'CreateEventIntent'),
    ('Shopping', 'ItemName'),
    ('ConnectDeviceIntent', 'ContactName')
]

RISK_HYPOTHESIS_DROP = [
    *CONFLICT_HYPOTHESIS_DROP,
    ('StationName', 'StationNumber'),  # rewrite drops both 'StationName' and 'StationNumber'
    ('ChannelName', 'CallSign'),
    ('StationName', 'CallSign'),
    ('Quantifier:all',),
    ('TargetDevice', ('xbox', 'fire tv')),
    (
        'AppName',
        (
            'nextflix',
            'hulu',
            'disney',
            'tube',
            'xfinity',
            'h. b. o. max',
            'pandora',
            'spotify',
            'heart',
            'sleep',
            'rain'
        )
    ),
    ('SortType', 'like'),
    ('ShortWeatherDetail',),
    ('GenreName', 'MediaType:game')
]

CONFLICT_HYPOTHESIS_ADDITION = [
]

RISK_HYPOTHESIS_ADDITION = [
    *CONFLICT_HYPOTHESIS_ADDITION,
    ('ContactName', 'home'),
    ('Device', 'everywhere'),
    ('EndCallIntent',),
    ('OriginalContent', 'ContentType:joke'),
    ('Help', ('not what i', 'i did not', "i don't", 'i do not', "i'm not", 'i am not', 'not you'))
]

CONFLICT_HYPOTHESIS_CHANGE = [
    (
        (('*SetNotificationIntent', ), '!*Time', '!*Duration'),  # request-side condition
        (('*Time', '*Duration'),)  # conditions request fails but rewrite meets
    ),
    (
        ('*Status:hide',),
        ('*BrowseGalleryIntent',)
    ),
    (
        ('*Type:package',),
        ('*ItemName',)
    ),
    (
        ('*CancelReminderIntent',),
        ('*NotificationLabel',)
    ),
]

RISK_HYPOTHESIS_CHANGE = [
    *CONFLICT_HYPOTHESIS_CHANGE,
    (
        ('!*QAIntent', '!*amzn1'),  # request-side condition
        (('*Detail', '*Feedback', '*Robot'),)  # conditions request fails but rewrite meets
    ),
    (
        ('!*RepeatIntent',),
        ('*Help',)
    ),
    (
        ('*PlayMusicIntent', '*SongName'),
        ('*LaunchNativeAppIntent', '*quiz')
    ),
    (
        ('*SongName', '*ArtistName', ('*AppName', '*ServiceName')),
        ('!*SongName', '!*ArtistName', ('*AppName', '*ServiceName'))
    ),
    (
        ('*SongName', '*ArtistName',),
        ('!*SongName', '!*ArtistName', '*VideoName')
    ),
    (
        ('*SongName', '*ArtistName',),
        ('!*SongName', '!*ArtistName', '*BookName')
    ),
    (
        ('*PlaylistName',),
        ('*ActiveUserTrigger:my',)
    ),
    (
        ('*ActiveUserTrigger:my',),
        ('*PlaylistName',)
    ),
    (
        ('*GetApparelRecommendationIntent',),
        (('!*EventName', '!*Preference'),)
    ),
    (
        ('!*shark',),
        ('*baby shark',)
    ),
]
