from enum import Enum
from types import MappingProxyType

AGGREGATION_ROOT_PATH_PATTERN = '{prefix}/{workspace}/{data}/{version}/{locale}'
AGGREGATION_PATH_PATTERN_SUFFIX = '/{year}/{month}/{day}/{file}'
AGGREGATION_PATH_PATTERN_WITH_HOUR_SUFFIX = '/{year}/{month}/{day}/{hour}/{file}'

# region workspace, region, locale

class PreDefinedWorkspaces(str, Enum):
    PROD = 'prod'
    MASTER = 'master'
    SCIENCE = 'science'


class SupportedRegions(str, Enum):
    NA = 'na'
    EU = 'eu'
    FE = 'fe'


class SupportedLocales(str, Enum):
    EN_US = 'en_US'
    EN_GB = 'en_GB'
    DE_DE = 'de_DE'
    PT_BR = 'pt_BR'
    IT_IT = 'it_IT'
    EN_IN = 'en_IN'
    EN_CA = 'en_CA'
    ES_MX = 'es_MX'
    ES_US = 'es_US'
    ES_ES = 'es_ES'
    FR_FR = 'fr_FR'
    ALL ='all'


# endregion

# region buckets, owner, path prefixes

class PreDefinedBuckets(str, Enum):
    AbmlWorkspaces = 'abml-workspaces'
    BluetrainDatasetsLive = 'bluetrain-datasets-live'
    BluetrainDatasetsLiveMirror = 'bluetrain-datasets-live-mirror'


DEFAULT_BUCKET = PreDefinedBuckets.AbmlWorkspaces

CROSS_REGION_BUCKETS = {
    (SupportedRegions.NA, SupportedRegions.EU):
        PreDefinedBuckets.BluetrainDatasetsLiveMirror
}

OWNER_NAME_INHOUSE = 'in_house'
OWNER_NAME_AESLU = 'aeslu'


class SupportedOwners(str, Enum):
    IN_HOUSE = OWNER_NAME_INHOUSE
    AESLU = OWNER_NAME_AESLU


AGGREGATION_ABML_S3_PREFIX_NA = 's3://abml-workspaces-na'
AGGREGATION_ABML_S3_PREFIX_EU = 's3://abml-workspaces-eu'
AGGREGATION_BDL_S3_PREFIX_NA = 's3://bluetrain-datasets-live'
AGGREGATION_BDL_S3_PREFIX_EU_MIRROR = 's3://bluetrain-datasets-live-mirror-eu-west-1'
AGGREGATION_BDL_S3_PREFIX_EU = 's3://bluetrain-eu-datasets-live'

AGGREGATION_S3_PREFIXS_ABML = {
    OWNER_NAME_INHOUSE: {
        SupportedRegions.NA: AGGREGATION_ABML_S3_PREFIX_NA,
        SupportedRegions.EU: AGGREGATION_ABML_S3_PREFIX_EU,
    }
}
AGGREGATION_S3_PREFIXS_BDL = {
    OWNER_NAME_INHOUSE: {
        SupportedRegions.NA: AGGREGATION_BDL_S3_PREFIX_NA,
        SupportedRegions.EU: AGGREGATION_BDL_S3_PREFIX_EU
    }
}
AGGREGATION_S3_PREFIXS_BDL_MIRROR = {
    OWNER_NAME_INHOUSE: {
        SupportedRegions.NA: AGGREGATION_BDL_S3_PREFIX_NA,
        SupportedRegions.EU: AGGREGATION_BDL_S3_PREFIX_EU_MIRROR,
    }
}
COMMON_PREFIX_DICT = MappingProxyType(
    {
        PreDefinedBuckets.AbmlWorkspaces: AGGREGATION_S3_PREFIXS_ABML,
        PreDefinedBuckets.BluetrainDatasetsLive: AGGREGATION_S3_PREFIXS_BDL,
        PreDefinedBuckets.BluetrainDatasetsLiveMirror: AGGREGATION_S3_PREFIXS_BDL_MIRROR
    }
)

# endregion

# region strategy
class PreDefinedStrategies(str, Enum):
    PROD = 'prod'
    DEVELOPMENT = 'dev'
    EXPERIMENTAL = 'lab'
# endregion
