from typing import Union, Iterable

from boba_python_utils.path_utils.path_join import join_
from boba_python_utils.production_utils.common.constants import (
    SupportedRegions, SupportedLocales, PreDefinedBuckets
)
from boba_python_utils.production_utils.common.path import get_cross_region_bucket
from boba_python_utils.production_utils.pdfs.experiment._constants.paths import (  # noqa: E501
    DIRNAME_ARTIFACTS,
    S3PATHS_PDFS_S3_WORKSPACE_ROOT,
    PATH_PDFS_LOCAL_WORKSPACE_ROOT,
    BLOCKLIST_VERSIONS,
    PreDefinedArtifactTypes
)


def get_pdfs_experiment_s3_root_path_prefix(
        region: str,
        bucket: Union[str, PreDefinedBuckets] = PreDefinedBuckets.AbmlWorkspaces
) -> str:
    """
    Gets a pre-defined s3 path prefix to be used for a pDFS experiment;
    """
    return S3PATHS_PDFS_S3_WORKSPACE_ROOT[bucket][region]


def get_pdfs_experiment_s3_root_path(
        region: Union[str, SupportedRegions],
        locale: Union[str, SupportedLocales],
        bucket: Union[str, PreDefinedBuckets] = PreDefinedBuckets.AbmlWorkspaces,
        workspace=None
) -> str:
    return join_(
        get_pdfs_experiment_s3_root_path_prefix(
            region,
            (bucket or PreDefinedBuckets.AbmlWorkspaces)
        ),
        workspace,
        locale,
        ignore_empty=True
    )


def get_pdfs_experiment_local_root_path(
        locale: Union[str, SupportedLocales],
        workspace=None
) -> str:
    return join_(PATH_PDFS_LOCAL_WORKSPACE_ROOT, workspace, locale)


def _get_artifact_path(
        root_path: str,
        artifact_type: str,
        artifact_family=None,
        artifact_name=None,
        artifact_version: Union[str, Iterable[str]] = None
):
    if not root_path:
        raise ValueError("empty artifact root path")

    return join_(
        root_path,
        DIRNAME_ARTIFACTS,
        artifact_type,
        artifact_family,
        artifact_name,
        artifact_version,
        ignore_empty=True
    )


def get_artifact_s3_path(
        region: Union[str, SupportedRegions],
        locale: Union[str, SupportedLocales],
        artifact_type: Union[str, PreDefinedArtifactTypes],
        artifact_family=None,
        artifact_name=None,
        artifact_version: Union[str, Iterable[str]] = None,
        bucket: Union[str, PreDefinedBuckets] = PreDefinedBuckets.AbmlWorkspaces,
        workspace=None
):
    pdfs_experiment_s3_root_path = get_pdfs_experiment_s3_root_path(
        region=region,
        locale=locale,
        bucket=bucket,
        workspace=workspace
    )

    return _get_artifact_path(
        root_path=pdfs_experiment_s3_root_path,
        artifact_type=artifact_type,
        artifact_family=artifact_family,
        artifact_name=artifact_name,
        artifact_version=artifact_version
    )


def get_artifact_local_path(
        locale: Union[str, SupportedLocales],
        artifact_type: str,
        artifact_family=None,
        artifact_name=None,
        artifact_version: Union[str, Iterable[str]] = None,
        workspace=None
):
    pdfs_experiment_s3_root_path = get_pdfs_experiment_local_root_path(
        locale=locale,
        workspace=workspace
    )

    return _get_artifact_path(
        root_path=pdfs_experiment_s3_root_path,
        artifact_type=artifact_type,
        artifact_family=artifact_family,
        artifact_name=artifact_name,
        artifact_version=artifact_version
    )


def get_blocklist_path(
        region: Union[str, SupportedRegions],
        locale: Union[str, SupportedLocales],
        version: Union[str, Iterable[str]],
        workspace=None,
        bucket: Union[str, PreDefinedBuckets] = None,
        target_region: Union[str, SupportedRegions] = None
):
    blocklist_version_dict = BLOCKLIST_VERSIONS[region][locale]
    if bucket is None:
        bucket = get_cross_region_bucket(source_region=region, target_region=target_region)

    if isinstance(version, str):
        if version in blocklist_version_dict:
            version = blocklist_version_dict[version]
    else:
        version = sum(
            (blocklist_version_dict.get(_version, [_version]) for _version in version),
            []
        )

    return get_artifact_s3_path(
        region=target_region or region,
        locale=locale,
        artifact_type=PreDefinedArtifactTypes.Blocklist,
        artifact_version=version,
        workspace=workspace,
        bucket=bucket
    )
