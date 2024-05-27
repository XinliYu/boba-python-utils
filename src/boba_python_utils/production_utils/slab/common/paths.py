from typing import Union

from boba_python_utils.path_utils.path_join import join_
from boba_python_utils.production_utils.common._constants.path import (
    PreDefinedBuckets,
    SupportedRegions,
    AGGREGATION_ABML_S3_PREFIX_NA,
    AGGREGATION_ABML_S3_PREFIX_EU,
    AGGREGATION_BDL_S3_PREFIX_NA,
    AGGREGATION_BDL_S3_PREFIX_EU,
    AGGREGATION_BDL_S3_PREFIX_EU_MIRROR,
    SupportedLocales
)

DIR_NAME_WORKSPACE_ROOT = 'slab_modeling'
PATH_SLAB_LOCAL_WORKSPACE_ROOT = f'/efs-storage/{DIR_NAME_WORKSPACE_ROOT}'
S3PATH_SLAB_S3_WORKSPACE_ROOT_NA = f'{AGGREGATION_ABML_S3_PREFIX_NA}/{DIR_NAME_WORKSPACE_ROOT}'
S3PATH_SLAB_S3_WORKSPACE_ROOT_EU = f'{AGGREGATION_ABML_S3_PREFIX_EU}/{DIR_NAME_WORKSPACE_ROOT}'
S3PATH_SLAB_S3_BDL_WORKSPACE_ROOT_NA = f'{AGGREGATION_BDL_S3_PREFIX_NA}/{DIR_NAME_WORKSPACE_ROOT}'
S3PATH_SLAB_S3_BDL_WORKSPACE_ROOT_EU = f'{AGGREGATION_BDL_S3_PREFIX_EU}/{DIR_NAME_WORKSPACE_ROOT}'
S3PATH_SLAB_S3_BDL_WORKSPACE_ROOT_EU_MIRROR = f'{AGGREGATION_BDL_S3_PREFIX_EU_MIRROR}/{DIR_NAME_WORKSPACE_ROOT}'

S3PATHS_SLAB_S3_WORKSPACE_ROOT = {
    PreDefinedBuckets.AbmlWorkspaces:
        {
            SupportedRegions.NA: S3PATH_SLAB_S3_WORKSPACE_ROOT_NA,
            SupportedRegions.EU: S3PATH_SLAB_S3_WORKSPACE_ROOT_EU,
        },
    PreDefinedBuckets.BluetrainDatasetsLive:
        {
            SupportedRegions.NA: S3PATH_SLAB_S3_BDL_WORKSPACE_ROOT_NA,
            SupportedRegions.EU: S3PATH_SLAB_S3_BDL_WORKSPACE_ROOT_EU,
        },
    PreDefinedBuckets.BluetrainDatasetsLiveMirror:
        {
            SupportedRegions.NA: S3PATH_SLAB_S3_BDL_WORKSPACE_ROOT_NA,
            SupportedRegions.EU: S3PATH_SLAB_S3_BDL_WORKSPACE_ROOT_EU_MIRROR,
        }
}


def get_slab_experiment_s3_root_path_prefix(
        region: Union[str, SupportedRegions],
        bucket: Union[str, PreDefinedBuckets] = PreDefinedBuckets.AbmlWorkspaces
) -> str:
    """
    Gets a pre-defined s3 path prefix to be used for a SLAB experiment;
    """
    return S3PATHS_SLAB_S3_WORKSPACE_ROOT[bucket][region]


def get_slab_experiment_s3_root_path(
        region: Union[str, SupportedRegions],
        locale: Union[str, SupportedLocales],
        bucket: Union[str, PreDefinedBuckets] = PreDefinedBuckets.AbmlWorkspaces,
        workspace=None
) -> str:
    return join_(
        get_slab_experiment_s3_root_path_prefix(
            region,
            (bucket or PreDefinedBuckets.AbmlWorkspaces)
        ),
        workspace,
        locale,
        ignore_empty=True
    )


def get_slab_experiment_local_root_path(
        locale: Union[str, SupportedLocales],
        workspace=None
) -> str:
    return join_(PATH_SLAB_LOCAL_WORKSPACE_ROOT, workspace, locale)
