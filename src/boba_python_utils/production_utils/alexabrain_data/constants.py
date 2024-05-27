from boba_python_utils.production_utils.alexabrain_data._situated_context.constants import *

from boba_python_utils.production_utils.common.constants import SupportedRegions, AGGREGATION_PATH_PATTERN_WITH_HOUR_SUFFIX

GREENWICH_PATH_PATTERN_SUFFIX = '/{year}/{month}/{day}/{hour}/{file}'
_ALEXA_BRAIN_PATH_PREFIX = 'abml://ssdg-data-store-{dataset_id}/{dataset_id}/canonical/*'
ALEXA_BRAIN_PATH_PREFIX_NA = _ALEXA_BRAIN_PATH_PREFIX.format(dataset_id='56da5cf8-c91f-432b-acfc-2e0f36739a9c') + AGGREGATION_PATH_PATTERN_WITH_HOUR_SUFFIX
ALEXA_BRAIN_PATH_PREFIX_EU = _ALEXA_BRAIN_PATH_PREFIX.format(dataset_id='565a64f3-47ec-414c-9ee3-5b78215051df') + AGGREGATION_PATH_PATTERN_WITH_HOUR_SUFFIX
ALEXA_BRAIN_PATH_PREFIX_FE = _ALEXA_BRAIN_PATH_PREFIX.format(dataset_id='0629e736-be3c-459c-868a-51c261060520') + AGGREGATION_PATH_PATTERN_WITH_HOUR_SUFFIX

ALEXA_BRAIN_DATA_PATH_PATTERNS = {
    SupportedRegions.NA: ALEXA_BRAIN_PATH_PREFIX_NA,
    SupportedRegions.EU: ALEXA_BRAIN_PATH_PREFIX_EU,
    SupportedRegions.FE: ALEXA_BRAIN_PATH_PREFIX_FE
}

AB_KEY_SEGMENTS = 'dataSetOutput.value.segments'
AB_SEGMENT_KEY_SIMPLIFIED_REQUEST = 'aus.GenerateAlternativeUtterancesV2.traceRecord.simplifiedRequest'
AB_SEGMENT_KEY_ORIGINAL_UTTERANCES = 'aus.GenerateAlternativeUtterancesV2.traceRecord.simplifiedRequest.originalUtterances'
AB_SEGMENT_KEY_CUSTOMER_ID = 'dialog.activeUserState.activeUser.customerId'
AB_SEGMENT_KEY_DEVICE_TYPE = 'device_arbitration.deviceType'
AB_SEGMENT_KEY_SITUATED_CONTEXT = f'{AB_SEGMENT_KEY_SIMPLIFIED_REQUEST}.{KEY_SITUATED_CONTEXT}'

AB_KEY_LOCALE = 'dataSetOutput.value.stream.audio.locale'
AB_KEY_SIMPLIFIED_REQUEST = f'{AB_KEY_SEGMENTS}.{AB_SEGMENT_KEY_SIMPLIFIED_REQUEST}'
AB_KEY_SITUATED_CONTEXT = f'{AB_KEY_SIMPLIFIED_REQUEST}.{KEY_SITUATED_CONTEXT}'

KEY_SEGMENTS = 'segments'
