from enum import Enum

from boba_python_utils.production_utils.common.constants import SupportedRegions

GREENWICH_UNIFIED_PATH_PREFIX = 's3abml://abml-workspaces-na/flare_unified_metrics'
GREENWICH_NEXTGEN_PATH_PREFIX = 's3abml://abml-workspaces-na/greenwich-nextgen/metrics'
GREENWICH3_PATH_PREFIX = 's3abml://abml-workspaces-na/greenwich/metrics/3.0.0'
GREENWICH_NEXTGEN_PATH_PREFIX_EU = 's3abml://abml-workspaces-eu/greenwich-nextgen/metrics'
GREENWICH_UNIFIED_PATH_PREFIX_EU = 's3abml://abml-workspaces-eu/flare_unified_metrics'
GREENWICH3_PATH_PREFIX_EU = 's3abml://abml-workspaces-eu/greenwich/metrics/3.0.0'

GREENWICH_PATH_PATTERN_SUFFIX = '/{year}/{month}/{day}/{hour}/{file}'
GREENWICH_UNIFIED_PATH_PATTERN = GREENWICH_UNIFIED_PATH_PREFIX + GREENWICH_PATH_PATTERN_SUFFIX
GREENWICH_NEXTGEN_PATH_PATTERN = GREENWICH_NEXTGEN_PATH_PREFIX + GREENWICH_PATH_PATTERN_SUFFIX
GREENWICH3_PATH_PATTERN = GREENWICH3_PATH_PREFIX + GREENWICH_PATH_PATTERN_SUFFIX
GREENWICH_UNIFIED_PATH_PATTERN_EU = GREENWICH_UNIFIED_PATH_PREFIX_EU + GREENWICH_PATH_PATTERN_SUFFIX
GREENWICH_NEXTGEN_PATH_PATTERN_EU = GREENWICH_NEXTGEN_PATH_PREFIX_EU + GREENWICH_PATH_PATTERN_SUFFIX
GREENWICH3_PATH_PATTERN_EU = GREENWICH3_PATH_PREFIX_EU + GREENWICH_PATH_PATTERN_SUFFIX



class GwVersion(str, Enum):
    GREENWICH3 = 'greenwich3'
    GREENWICH_NEXTGEN = 'greenwich_nextgen',
    GREENWICH_UNIFIED = 'greenwich_unified'


GREENWICH_PATH_PATTERNS = {
    GwVersion.GREENWICH3: {
        SupportedRegions.NA: GREENWICH3_PATH_PATTERN,
        SupportedRegions.EU: GREENWICH3_PATH_PATTERN_EU,
    },
    GwVersion.GREENWICH_NEXTGEN: {
        SupportedRegions.NA: GREENWICH_NEXTGEN_PATH_PATTERN,
        SupportedRegions.EU: GREENWICH_NEXTGEN_PATH_PATTERN_EU,
    },
    GwVersion.GREENWICH_UNIFIED: {
        SupportedRegions.NA: GREENWICH_UNIFIED_PATH_PATTERN,
        SupportedRegions.EU: GREENWICH_UNIFIED_PATH_PATTERN_EU,
    }
}

