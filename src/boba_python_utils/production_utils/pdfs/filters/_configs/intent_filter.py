from boba_python_utils.production_utils.common._constants.path import SupportedLocales

RISK_INTENT_SPECIFIC_REQUEST_KEYWORDS = {
    'QAIntent': ['spell'],
    'amzn': ['spell']
}

IGNORED_INTENTS_FOR_REPHRASE = {
    'StopIntent',
    'NextIntent',
    'CancelIntent',
    'PauseIntent',
    'VolumeDownIntent',
    'WhatTimeIntent',
    'SetVolumeIntent',
    'NoExpressedIntent',
    'GetMusicDetailsIntent',
    'AMAZON.NextIntent',
    'PreviousIntent',
    'AMAZON.PreviousIntent',
    'MoreIntent',
    'SelectIntent',
    'VolumeUpIntent',
    'FeedbackIntent',
    'InitiateFeedbackIntent'
    'PhaticIntent',
    'AMAZON.ResumeIntent',
    'StartOverIntent',
    'AMAZON.StartOverIntent',
    'ExitCallIntent',
    'AboutAlexaQAIntent',
    'PleasantryIntent',
    'FarewellIntent',
    'GreetingsIntent',
    'WhoAmIIntent',
    'PersonalityQAIntent',
    'NegativeFeedbackIntent',
    'TeachAlexaIntent',
    'ConversationIntent',
    'TranslateUtteranceIntent',
}

INVALID_INTENT_PAIRS = {
    # Music
    ('AddToPlaylistIntent', 'PlayMusicIntent'),
    ('AddToPlaylistIntent', 'PlayStationIntent'),
    ('AddToPlaylistIntent', 'ContentOnlyIntent'),
    ('AddToPlaylistIntent', 'MusicControlIntent'),
    ('MusicControlIntent', 'GetMusicDetailsIntent'),
    ('AddToPlaylistIntent', 'GetMusicDetailsIntent'),
    ('ExpressMusicPreferenceIntent', 'GetMusicDetailsIntent'),
    ('ExpressPreferenceIntent', 'GetMusicDetailsIntent'),
    ('AddToPlaylistIntent', 'ExpressMusicPreferenceIntent'),
    ('RemoveFromPlayQueueIntent', 'PlayMusicIntent'),
    ('PlayMusicIntent', 'FindMusicIntent'),
    ('PlayMusicIntent', 'GetMusicDetailsIntent'),
    ('PlayMusicIntent', 'ExpressMusicPreferenceIntent'),
    ('PlayMusicIntent', 'NavigateIntent'),
    ('PlayMusicIntent', 'LaunchGameIntent'),
    ('PlayMusicIntent', 'GetContentIntent'),
    ('PlayStationIntent', 'EnableDoNotDisturbIntent'),
    ('GetPlayQueueIntent', 'GetMusicDetailsIntent'),

    # HomeAutomation
    ('TurnOnApplianceIntent', 'TurnOffApplianceIntent'),
    ('TurnOnApplianceIntent', 'CloseApplianceIntent'),
    ('TurnOnApplianceIntent', 'GetMusicDetailsIntent'),
    ('TurnOnApplianceIntent', 'DisplayVideoFeedIntent'),
    ('TurnOnApplianceIntent', 'GetContentIntent'),
    ('TurnOffApplianceIntent', 'OpenApplianceIntent'),
    ('TurnOffApplianceIntent', 'StopIntent'),
    ('TurnOffApplianceIntent', 'GetContentIntent'),
    ('TurnOffApplianceIntent', 'SilenceNotificationIntent'),
    ('SetUpIntent', 'TurnOffApplianceIntent'),
    ('GetSettingsDetailsIntent', 'TurnOnApplianceIntent'),
    ('TurnOnApplianceIntent', 'SetValueIntent'),
    ('TurnOffApplianceIntent', 'SetValueIntent'),
    ('TurnOffApplianceIntent', 'NavigateIntent'),
    ('UnlockApplianceIntent', 'LockApplianceIntent'),
    ('UnmuteApplianceIntent', 'MuteApplianceIntent'),
    ('OpenApplianceIntent', 'CloseApplianceIntent'),
    ('CloseApplianceIntent', 'InvokeRoutineIntent'),
    ('OpenAutoApplianceIntent', 'CloseAutoApplianceIntent'),
    ('ApplianceSettingDownIntent', 'TurnOnApplianceIntent'),
    ('ApplianceSettingUpIntent', 'TurnOffApplianceIntent'),
    ('ApplianceSettingUpIntent', 'ApplianceSettingDownIntent'),
    ('VolumeUpIntent', 'VolumeDownIntent'),
    ('PairDeviceIntent', 'UnpairDeviceIntent'),
    ('SecureIntent', 'UnsecureIntent'),
    ('ConnectDeviceIntent', 'DisconnectDeviceIntent'),
    ('PairDeviceIntent', 'DisconnectDeviceIntent'),
    ('DisconnectDeviceIntent', 'GetContentIntent'),
    ('PairDeviceIntent', 'GetContentIntent'),
    ('DisplayVideoFeedIntent', 'StopVideoFeedIntent'),
    ('FindMyItemIntent', 'ConnectDeviceIntent'),
    ('StopVideoFeedIntent', 'DisplayVideoFeedIntent'),
    ('RemoveFromHomeScreenIntent', 'AddToHomeScreenIntent'),
    ('GrantAccessIntent', 'DenyAccessIntent'),
    ('RemoveFromGroupIntent', 'AddToGroupIntent'),
    ('MuteIntent', 'VolumeUpIntent'),
    ('MuteIntent', 'GetVolumeSettingIntent'),
    ('MuteIntent', 'PlayIntent'),
    ('UnMuteIntent', 'GetContentIntent'),
    ('UnMuteIntent', 'WhatDayIntent'),
    ('VolumeDownIntent', 'StopIntent'),
    ('VolumeDownIntent', 'TurnOnApplianceIntent'),
    ('VolumeDownIntent', 'TurnOffApplianceIntent'),
    ('CancelIntent', 'SetValueIntent'),

    # Books
    ('RemoveBookmarkIntent', 'AddBookmarkIntent'),
    ('ReadBookIntent', 'BrowseBookIntent'),

    # Calendar
    ('RemoveEventIntent', 'CreateEventIntent'),
    ('RemoveFromEventIntent', 'AddToEventIntent'),
    ('BrowseCalendarIntent', 'CreateEventIntent'),
    ('RemoveEventIntent', 'BrowseCalendarIntent'),

    # Communication
    ('CallIntent', 'NavigateIntent'),
    ('CallIntent', 'AcceptCallIntent'),
    ('UnpairDeviceIntent', 'PairDeviceIntent'),
    ('InstantConnectIntent', 'DisableCommsIntent'),
    ('InstantConnectIntent', 'EndCallIntent'),
    ('ExitCallIntent', 'CallIntent'),
    ('EndCallIntent', 'CallIntent'),
    ('CallIntent', 'SendMessageIntent'),
    ('UnmuteMyVoiceIntent', 'MuteMyVoiceIntent'),
    ('ResumeReadingMessageIntent', 'PauseReadingMessageIntent'),
    ('GetMessageIntent', 'CancelMessageIntent'),
    ('StopTranscriptionIntent', 'StartTranscriptionIntent'),
    ('RepeatMessageIntent', 'RemoveMessageIntent'),
    ('ResumeCallIntent', 'EndCallIntent'),
    ('IgnoreCallIntent', 'CallIntent'),
    ('ResumeReadingMessageIntent', 'RestartReadingMessageIntent'),
    ('ResumeCallIntent', 'RejectCallIntent'),
    ('DisconnectDeviceIntent', 'ConnectDeviceIntent'),
    ('ResumeTranscriptionIntent', 'PauseTranscriptionIntent'),
    ('StopReadingMessageIntent', 'RestartReadingMessageIntent'),
    ('RemoveContactIntent', 'AddContactIntent'),
    ('RemoveEffectIntent', 'AddEffectIntent'),
    ('EnableCommsIntent', 'DisableCommsIntent'),
    ('DisconnectDeviceIntent', 'PairDeviceIntent'),
    ('JoinCallIntent', 'InvokeRoutineIntent'),
    ('AcceptCallIntent', 'RejectCallIntent'),
    ('AcceptCallIntent', 'EndCallIntent'),
    ('TurnOffApplianceIntent', 'CallIntent'),
    ('SendMessageIntent', 'GetMessageIntent'),

    # DailyBriefing
    ('EnableNewsNotificationIntent', 'DisableNewsNotificationIntent'),

    # Gallery
    ('BrowseGalleryIntent', 'SearchVideoIntent'),
    ('SearchVideoIntent', 'ContentOnlyIntent'),
    ('StopGalleryContentIntent', 'PlayGalleryContentIntent'),
    ('RemoveFromGalleryIntent', 'AddToGalleryIntent'),
    ('RemoveGalleryDetailsIntent', 'AddGalleryDetailsIntent'),
    ('ResumeGalleryIntent', 'PauseGalleryIntent'),
    ('StopRecordingVideoIntent', 'StartRecordingVideoIntent'),

    # GeneralMedia
    ('RemoveNativeAppIntent', 'AddNativeAppIntent'),
    ('EnableAppIntent', 'DisableAppIntent'),
    ('LaunchNativeAppIntent', 'DisableAppIntent'),
    ('LaunchNativeAppIntent', 'CallIntent'),
    ('LaunchNativeAppIntent', 'CancelIntent'),

    # Global
    ('StopIntent', 'RestartIntent'),
    ('EnableAutoMoveIntent', 'DisableAutoMoveIntent'),
    ('EnableCleanModeIntent', 'DisableCleanModeIntent'),
    ('StopIntent', 'PlayIntent'),
    ('StartMembershipIntent', 'EndMembershipIntent'),
    ('PlayIntent', 'PauseIntent'),
    ('EnableTerseModeIntent', 'DisableTerseModeIntent'),
    ('StopRecordingIntent', 'StartRecordingIntent'),
    ('EnableDoNotDisturbIntent', 'DisableDoNotDisturbIntent'),
    ('PreviousIntent', 'NextIntent'),
    ('MuteIntent', 'UnmuteIntent'),
    ('ResumeIntent', 'PauseIntent'),
    ('ResumeIntent', 'StopIntent'),
    ('ResumeIntent', 'RestartIntent'),
    ('ResumeIntent', 'TurnOffApplianceIntent'),
    ('ZoomOutIntent', 'ZoomInIntent'),
    ('StopSharingLocationIntent', 'StartSharingLocationIntent'),
    ('StopWorkIntent', 'StartWorkIntent'),
    ('FastForwardIntent', 'RewindIntent'),
    ('MuteIntent', 'RewindIntent'),
    ('LaunchNativeAppIntent', 'FastForwardIntent'),
    ('GetDetailsIntent', 'ExplainActivityHistoryIntent'),

    # HealAndFitness
    ('ResumeFitnessEventIntent', 'PauseFitnessEventIntent'),
    ('StopMedicationManagementIntent', 'StartMedicationManagementIntent'),
    ('ResumeHealthEventIntent', 'PauseHealthEventIntent'),
    ('StopFitnessEventIntent', 'StartFitnessEventIntent'),
    ('StartFitnessEventIntent', 'RestartFitnessEventIntent'),
    ('StopHealthEventIntent', 'StartHealthEventIntent'),
    ('ResumeHealthEventIntent', 'RemoveHealthEventIntent'),
    ('StopFitnessEventIntent', 'RestartFitnessEventIntent'),

    # HelpApplianceSettingDownIntent
    ('StopTutorialIntent', 'StartTutorialIntent'),

    # Knowledge
    ('SetInfoNotificationIntent', 'CancelInfoNotificationIntent'),
    ('RememberIntent', 'BrowseMemoryIntent'),

    # LocalSearch
    ('IsOpenIntent', 'IsClosedIntent'),
    ('MapUnmuteIntent', 'MapMuteIntent'),
    ('CancelWaypointIntent', 'AddWaypointIntent'),

    # Music
    ('RemoveFromPlayQueueIntent', 'AddToPlayQueueIntent'),
    ('EnableMusicInfoModeIntent', 'DisableMusicInfoModeIntent'),
    ('EnableCleanMusicModeIntent', 'DisableCleanMusicModeIntent'),
    ('RemoveFromPlayQueueIntent', 'AddToPlaylistIntent'),

    # Notifications
    ('ResumeNotificationIntent', 'PauseNotificationIntent'),
    ('SetReminderIntent', 'CancelReminderIntent'),
    ('CancelReminderIntent', 'ExpressMusicPreferenceIntent'),
    ('PlayMusicIntent', 'CancelNotificationIntent'),
    ('CancelReminderIntent', 'ExpressPreferenceIntent'),
    ('SetNotificationIntent', 'CancelNotificationIntent'),
    ('SetNotificationIntent', 'SilenceNotificationIntent'),
    ('SetNotificationIntent', 'BrowseCalendarIntent'),
    ('BrowseNotificationIntent', 'SilenceNotificationIntent'),
    ('BrowseNotificationIntent', 'SetNotificationIntent'),
    ('BrowseNotificationIntent', 'CancelNotificationIntent'),
    ('BrowseNotificationIntent', 'RemoveNotificationIntent'),
    ('PauseNotificationIntent', 'CancelNotificationIntent'),
    ('EditNotificationIntent', 'SetNotificationIntent'),
    ('GetNotificationVolumeSettingIntent', 'NotificationVolumeUpIntent'),
    ('SnoozeNotificationIntent', 'SetNotificationIntent'),

    # Recipes
    ('StopRecipeIntent', 'PlayRecipeIntent'),
    ('RemoveFromRecipeListIntent', 'AddToRecipeListIntent'),
    ('NavigateRecipeIntent', 'AddToRecipeListIntent'),
    ('BrowseRecipeListIntent', 'AddToRecipeListIntent'),
    ('PlayRecipeIntent', 'NextRecipeIntent'),
    ('PrepareFoodIntent', 'TurnOffApplianceIntent'),

    # Robot
    ('RobotTurnOnDeviceIntent', 'RobotTurnOffDeviceIntent'),
    ('RestrictAccessIntent', 'GiveAccessIntent'),
    ('EndHomeTourIntent', 'HomeTourIntent'),
    ('UnsetLocationNameIntent', 'SetLocationNameIntent'),
    ('StartExplorationIntent', 'EndExplorationIntent'),

    # Routines
    ('EnableRoutineIntent', 'DisableRoutineIntent'),

    # Shopping
    ('SubmitOrderIntent', 'CancelOrderIntent'),
    ('ReturnItemIntent', 'BuyItemIntent'),
    ('RemoveItemFromShoppingListIntent', 'AddToListIntent'),
    ('CheckOrderStatusIntent', 'BrowseShoppingContainerIntent'),

    # SocialExperiences
    ('RemoveConnectionIntent', 'AddConnectionIntent'),

    # Translation
    ('StopLiveTranslationIntent', 'StartLiveTranslationIntent'),
    ('RemoveBilingualAnswerIntent', 'AddBilingualAnswerIntent'),

    # Utilities
    ('UnlinkBillingAccountIntent', 'LinkBillingAccountIntent'),

    # Video
    ('EnableSubtitlesIntent', 'DisableSubtitlesIntent'),
    ('PlayVideoIntent', 'PauseVideoIntent'),
    ('PauseVideoIntent', 'GetContentIntent'),
    ('VideoRemoveFromListIntent', 'VideoAddToListIntent'),
    ('RestrictVideoContentIntent', 'ExpandVideoContentIntent'),
    ('StopRecordingVideoProgramIntent', 'StartRecordingVideoProgramIntent'),
    ('ResumeVideoIntent', 'PauseVideoIntent'),
    ('GetVideoDetailsIntent', 'GetMusicDetailsIntent'),

    # Weather
    ('SetWeatherNotificationIntent', 'CancelWeatherNotificationIntent'),

    # Knowledge
    ('QAIntent', 'WhatTimeIntent'),
    ('QAIntent', 'WhatDayIntent'),
    ('SubjectiveQAIntent', 'WhatTimeIntent'),
    ('SubjectiveQAIntent', 'WhatDayIntent'),
    ('QAIntent', 'RepeatIntent'),
    ('QAIntent', 'LaunchNativeAppIntent'),
    ('QAIntent', 'GetVoiceHistoryIntent'),

    # Misc
    ('EchoIntent', 'EchoIntent'),
    ('GetGalleryDetailsIntent', 'GetGalleryDetailsIntent'),
    ('CapturePictureIntent', 'PlayGalleryContentIntent')
}

INVALID_INTENT_TRANSITION = {
    ('PlayVideoIntent', 'QAIntent'),
    ('PlayMusicIntent', 'QAIntent'),
    ('PlayMusicIntent', 'TurnOnApplianceIntent'),
    ('PlayMusicIntent', 'TurnOffApplianceIntent')
}

INVALID_INTENT_PAIRS_BY_LOCALE = {
    SupportedLocales.EN_GB: {
        ('ReadBookIntent', 'NavigateBooksIntent')
    }
}

RISKY_INTENT_PAIRS = {
    # Music
    ('SetMusicNotificationIntent', 'PlayMusicIntent'),
    ('PlayMusicIntent', 'GetVoiceHistoryIntent'),

    # HomeAutomation
    ('ToggleApplianceIntent', 'TurnOnApplianceIntent'),
    ('ToggleApplianceIntent', 'TurnOffApplianceIntent'),
    ('OpenApplianceIntent', 'ConnectDeviceIntent'),
    ('SetValueIntent', 'InvokeRoutineIntent'),

    # Notifications
    ('EditNotificationIntent', 'BrowseNotificationIntent'),
    ('SetNotificationIntent', 'GetWeatherForecastIntent'),
    ('CancelNotificationIntent', 'ExtendNotificationIntent'),
    ('CancelReminderIntent', 'SilenceNotificationIntent'),
    ('CancelReminderIntent', 'BrowseReminderIntent'),
}
