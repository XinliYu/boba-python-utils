from enum import Enum
from os import environ

from boba_python_utils.production_utils.common.constants import (
    SupportedRegions, SupportedLocales, PreDefinedBuckets,
    AGGREGATION_BDL_S3_PREFIX_NA, AGGREGATION_BDL_S3_PREFIX_EU, AGGREGATION_BDL_S3_PREFIX_EU_MIRROR,
    AGGREGATION_ABML_S3_PREFIX_NA, AGGREGATION_ABML_S3_PREFIX_EU
)

DIR_NAME_WORKSPACE_ROOT = 'pdfs_workspace'
PATH_PDFS_LOCAL_WORKSPACE_ROOT = f'/efs-storage/{DIR_NAME_WORKSPACE_ROOT}'
S3PATH_PDFS_S3_WORKSPACE_ROOT_NA = f'{AGGREGATION_ABML_S3_PREFIX_NA}/{DIR_NAME_WORKSPACE_ROOT}'
S3PATH_PDFS_S3_WORKSPACE_ROOT_EU = f'{AGGREGATION_ABML_S3_PREFIX_EU}/{DIR_NAME_WORKSPACE_ROOT}'
S3PATH_PDFS_S3_BDL_WORKSPACE_ROOT_NA = f'{AGGREGATION_BDL_S3_PREFIX_NA}/{DIR_NAME_WORKSPACE_ROOT}'
S3PATH_PDFS_S3_BDL_WORKSPACE_ROOT_EU = f'{AGGREGATION_BDL_S3_PREFIX_EU}/{DIR_NAME_WORKSPACE_ROOT}'
S3PATH_PDFS_S3_BDL_WORKSPACE_ROOT_EU_MIRROR = f'{AGGREGATION_BDL_S3_PREFIX_EU_MIRROR}/{DIR_NAME_WORKSPACE_ROOT}'

EXP_NAME_PDFS_P1_L2 = 'pdfs_p1_l2'
DIRNAME_RESOURCES = 'resources'
DIRNAME_MODELS = 'models'
DIRNAME_CONFIGS = 'configs'
DIRNAME_PDFS_CODEBASE = environ.get('dir_pdfs_codebase', '')
DIRNAME_ARTIFACTS = 'artifacts'
DIRNAME_L2_FEATURE_LIST = 'l2_feature_list'
DIRNAME_L2_FEATURES = 'l2_features'
DIRNAME_L2_FEATURES_NUMPY = 'l2_features_numpy'
DIRNAME_L2_RESULTS = 'l2_results'
DIRNAME_FEATURE_DATA = 'feature_data'
FILENAME_L2_FULL_FEATURE_LIST = 'l2_feats.txt'
S3PATH_P1_L2_RESOURCES = 's3://abml-workspaces-na/zgchen/pdfs/resources_p1_l2'

S3PATHS_PDFS_S3_WORKSPACE_ROOT = {
    PreDefinedBuckets.AbmlWorkspaces:
        {
            SupportedRegions.NA: S3PATH_PDFS_S3_WORKSPACE_ROOT_NA,
            SupportedRegions.EU: S3PATH_PDFS_S3_WORKSPACE_ROOT_EU,
        },
    PreDefinedBuckets.BluetrainDatasetsLive:
        {
            SupportedRegions.NA: S3PATH_PDFS_S3_BDL_WORKSPACE_ROOT_NA,
            SupportedRegions.EU: S3PATH_PDFS_S3_BDL_WORKSPACE_ROOT_EU,
        },
    PreDefinedBuckets.BluetrainDatasetsLiveMirror:
        {
            SupportedRegions.NA: S3PATH_PDFS_S3_BDL_WORKSPACE_ROOT_NA,
            SupportedRegions.EU: S3PATH_PDFS_S3_BDL_WORKSPACE_ROOT_EU_MIRROR,
        }
}

BLOCKLIST_VERSIONS = {
    SupportedRegions.NA: {
        SupportedLocales.EN_US: {
            # 'prod': ['static_blacklist_2020_04_22', '2022_06_update_for_notifications'],
            'prod': ['static_blacklist_2020_04_22', 'pDFS_v1_blocklist_2021_06_update2_for_todos']
        },
        SupportedLocales.EN_CA: {
            'prod': ['global_count_1000-global_avg_defect_005-combine_existing_blocklist_en_US']
        }
    },
    SupportedRegions.EU: {
        SupportedLocales.EN_GB: {
            'prod': ['static_blacklist_2020_04_22', 'impression_1000-cpdr7_005-with_en_us']
        },
        SupportedLocales.EN_IN: {
            'prod': ['global_count_1000-global_avg_defect_005-combine_existing_blocklist_en_US']
        },
        SupportedLocales.DE_DE: {
            'prod': ['90days_impression_800-cpdr7_005-with_en_us']
        }
    }
}


class PreDefinedArtifactTypes(str, Enum):
    Models = 'models'
    Resources = 'resources'
    Blocklist = 'blocklist'
    Precompute = 'precompute'
