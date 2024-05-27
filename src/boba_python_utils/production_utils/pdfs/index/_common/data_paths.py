from datetime import datetime
from typing import Union, List
from os import path
from attr import attrib, attrs

from boba_python_utils.general_utils.strex import add_suffix
from boba_python_utils.production_utils.common.constants import (
    PreDefinedBuckets,
    SupportedOwners,
    SupportedRegions,
    PreDefinedWorkspaces,
    SupportedLocales
)
from boba_python_utils.production_utils.common.path import (
    get_aggregation_root_path,
    get_aggregation_path_prefix, get_dated_aggregation_path, MultiDayAggregationPath, DailyAggregationPath
)
from boba_python_utils.production_utils.pdfs.constants import (
    SLAB_AGGREGATION_S3_PREFIXS,
    PreDefinedDailyAggregationTypes,
    PreDefinedIndexDataTypes
)
from boba_python_utils.production_utils.pdfs.index._common.feature_flag import (  # noqa: F401,F403,E501
    FeatureFlag,
    SupportedCPDRVersions,
)
from boba_python_utils.production_utils.pdfs.index._constants.paths import PRODUCT_NAME_PDFS


# region aggregation path formation
def get_index_version(feature_flag: FeatureFlag) -> str:
    """
    Takes feature_flag object and determines the s3 output version folder
    Args:
        feature_flag: feature flag object indicating which features are enabled

    Returns: string representing the version to be used on s3 path
    """
    # ! DEPRECATED
    # ! science workspace is fixed to version 2.1.2
    if feature_flag.multi_source_index:
        if feature_flag.cpdr_version == SupportedCPDRVersions.V6:
            if feature_flag.add_signals:
                return "2.1.1"
            else:
                return "2.1.0"
        elif (
                feature_flag.cpdr_version == SupportedCPDRVersions.V7 or
                feature_flag.cpdr_version == SupportedCPDRVersions.UNIFIED
        ):
            return "2.1.2"
        elif feature_flag.cpdr_version == SupportedCPDRVersions.V3:
            return "2.0.0"
    else:
        return "1.0.0"


def get_slab_aggregation_path_prefix(
        region: str,
        product_name=PRODUCT_NAME_PDFS,
        owner: Union[str, SupportedOwners] = SupportedOwners.IN_HOUSE,
        bucket: [str, PreDefinedBuckets] = None,
) -> str:
    """
    Gets a pre-defined path prefix to be used for a SLAB aggregation;
        according to the `region` and `owner`.
        for example, "s3://abml-workspaces-na/pDFS".

    Args:
        region: e.g. 'na', 'eu'
        owner: e.g. 'in_house', 'aeslu'; see `SupportedOwners`.

    Returns:

    """
    return get_aggregation_path_prefix(
        product_name=product_name,
        bucket=bucket,
        owner=owner,
        region=region,
        path_prefix_dict=SLAB_AGGREGATION_S3_PREFIXS
    )


def get_slab_aggregation_root_path(
        prefix: str,
        workspace: str,
        data: str,
        locale: Union[str, SupportedLocales],
        feature_flag: Union[FeatureFlag, str]
) -> str:
    """
    Gets s3 root path for a pDFS aggregation.
    Args:
        prefix: s3 prefix specified for aggregation root path.
        workspace: the name of the folder right after the prefix to save the aggregation data
            (e.g. master, prod, debug).
        data: the name of the aggregation data (e.g. daily, 30days, entity_traffic_30days).
        locale: locale specified for aggregation.
        feature_flag: feature flag object indicating which features are enabled.

    Returns: s3 root path for some pDFS aggregation
    """
    version = None
    if isinstance(feature_flag, str):
        if feature_flag in SupportedCPDRVersions._value2member_map_:
            feature_flag = FeatureFlag(cpdr_version=feature_flag)
        else:
            version = feature_flag

    if version is None:
        version = 'common' if feature_flag is None else get_index_version(feature_flag)

    if (version <= '2.1.1' or version == '2.2.0') and locale == 'en_US':
        locale = 'en-us'

    if not isinstance(feature_flag, str) and feature_flag.secondary_version:
        version = add_suffix(version, feature_flag.secondary_version, sep='-')

    return get_aggregation_root_path(
        prefix=prefix,
        data=data,
        workspace=workspace,
        version=version,
        locale=locale
    )


def get_slab_aggregation_artifact_path(
        artifact_type: str,
        artifact_name: str,
        region: Union[str, SupportedRegions],
        locale: Union[str, SupportedLocales],
        workspace: Union[str, PreDefinedWorkspaces],
        feature_flag: Union[FeatureFlag, str],
        product_name=PRODUCT_NAME_PDFS,
        owner: Union[str, SupportedOwners] = SupportedOwners.IN_HOUSE,
        bucket: Union[str, PreDefinedBuckets] = PreDefinedBuckets.AbmlWorkspaces
):
    return get_slab_aggregation_root_path(
        prefix=get_aggregation_path_prefix(
            product_name=product_name,
            region=region,
            owner=owner,
            bucket=bucket,
            path_prefix_dict=SLAB_AGGREGATION_S3_PREFIXS
        ),
        workspace=workspace,
        data=path.join('artifacts', artifact_type, artifact_name),
        locale=locale,
        feature_flag=feature_flag
    )


ARTIFACT_TYPE_SCHEMA = 'schemas'


def get_slab_aggregation_schema_artifact_path(
        artifact_name: str,
        region: Union[str, SupportedRegions],
        locale: Union[str, SupportedLocales],
        workspace: Union[str, PreDefinedWorkspaces],
        feature_flag: Union[FeatureFlag, str],
        product_name=PRODUCT_NAME_PDFS,
        owner: Union[str, SupportedOwners] = SupportedOwners.IN_HOUSE,
        bucket: Union[str, PreDefinedBuckets] = PreDefinedBuckets.AbmlWorkspaces
):
    return get_slab_aggregation_artifact_path(
        artifact_type=ARTIFACT_TYPE_SCHEMA,
        artifact_name=artifact_name,
        region=region,
        locale=locale,
        workspace=workspace,
        feature_flag=feature_flag,
        product_name=product_name,
        owner=owner,
        bucket=bucket
    )


def get_dated_slab_aggregation_path(
        prefix: str,
        workspace: str,
        data: str,
        locale: Union[str, SupportedLocales],
        date: Union[str, datetime],
        feature_flag: Union[FeatureFlag, str] = None,
        num_days_backward: int = None,
        file=None,
) -> Union[str, List[str]]:
    """
    Gets s3 path to save some pDFS aggregation.
    Args:
        prefix: prefix specified for aggregation.
        workspace: the name of the folder right after the prefix to save the aggregation data
            (e.g. master, prod, debug).
        data: the name of the aggregation data (e.g. daily, 30days, entity_traffic_30days).
        locale: locale specified for aggregation.
        date: specific day to run daily pre-aggregation for.
        feature_flag: feature flag object indicating which features are enabled.
        num_days_backward: specifies a positive integer to get a list of paths,
            starting from 'date - num_days_backward' (exclusive) until 'date' (inclusive).
        file: if specified, this 'file' be appended to the end of the s3 path,
            so the returned will point to a file.

    Returns: s3 path to write some pDFS aggregation.
    """

    return get_dated_aggregation_path(
        root_path=get_slab_aggregation_root_path(
            prefix=prefix,
            workspace=workspace,
            data=data,
            locale=locale,
            feature_flag=feature_flag
        ),
        end_date=date,
        num_days=num_days_backward,
        file=file
    )


# endregion

# region daily aggregation and multi-day aggregation paths
@attrs(slots=True)
class SlabDailyAggregationPath(DailyAggregationPath):
    """
    Represents the path to a pDFS daily aggregation.

    See Also:
        :class:`DailyAggregationPath`.

    """

    product_name = attrib(type=str, default=PRODUCT_NAME_PDFS)
    feature_flag = attrib(type=Union[str, FeatureFlag], default=None)

    def __attrs_post_init__(self):
        if not self.product_name:
            self.product_name = PRODUCT_NAME_PDFS

    def _get_aggregation_root_path(self, data: str) -> str:
        return get_slab_aggregation_root_path(
            prefix=get_aggregation_path_prefix(
                product_name=self.product_name,
                region=self.region,
                owner=self.owner,
                bucket=self.bucket,
                path_prefix_dict=SLAB_AGGREGATION_S3_PREFIXS
            ),
            workspace=self.workspace,
            data=data,
            locale=self.locale,
            feature_flag=self.feature_flag
        )


def get_slab_daily_aggregation_path(
        region: Union[str, SupportedRegions],
        workspace: [str, PreDefinedWorkspaces],
        locale: Union[str, SupportedLocales],
        date: Union[str, datetime],
        feature_flag: Union[str, FeatureFlag] = None,
        file=None,
        num_days_backward: int = None,
        owner: [str, SupportedOwners] = SupportedOwners.IN_HOUSE,
        agg_type: [str, PreDefinedDailyAggregationTypes] = None,
        bucket: [str, PreDefinedBuckets] = None,
        product_name: str = None,
        path_obj: SlabDailyAggregationPath = None
) -> Union[str, List[str]]:
    """
    Gets one or more paths for pDFS daily aggregations.

    See Also:
        :class:`PdfsDailyAggregationPath` and :class:`DailyAggregationPath`.

    """
    if path_obj is None:
        path_obj = SlabDailyAggregationPath(
            num_days=num_days_backward,
            date=date,
            workspace=workspace,
            region=region,
            locale=locale,
            data_type=agg_type,
            file=file,
            feature_flag=feature_flag,
            owner=owner,
            bucket=bucket
        )

    return str(path_obj)


@attrs(slots=True)
class SlabMultiDayAggregationPath(MultiDayAggregationPath):
    """
    Represents the path to a pDFS multi-day aggregation.

    See Also:
        :class:`MultiDayAggregationPath`.

    """

    product_name = attrib(type=str, default=PRODUCT_NAME_PDFS)
    feature_flag = attrib(type=Union[str, FeatureFlag], default=None)

    def __attrs_post_init__(self):
        if not self.product_name:
            self.product_name = PRODUCT_NAME_PDFS

    def _get_aggregation_root_path(self, data: str) -> str:
        return get_slab_aggregation_root_path(
            prefix=get_aggregation_path_prefix(
                product_name=self.product_name,
                region=self.region,
                owner=self.owner,
                bucket=self.bucket,
                path_prefix_dict=SLAB_AGGREGATION_S3_PREFIXS
            ),
            workspace=self.workspace,
            data=data,
            locale=self.locale,
            feature_flag=self.feature_flag
        )


def get_slab_multi_day_aggregation_path(
        num_days: int,
        end_date: Union[str, datetime],
        region: Union[str, SupportedRegions],
        locale: Union[str, SupportedLocales],
        workspace: str,
        data_type: Union[str, PreDefinedIndexDataTypes],
        data_type_suffix: str = None,
        file: str = None,
        feature_flag: Union[FeatureFlag, str] = None,
        owner: [str, SupportedOwners] = SupportedOwners.IN_HOUSE,
        bucket: [str, PreDefinedBuckets] = None,
        product_name: str = None,
        path_obj: SlabMultiDayAggregationPath = None
) -> str:
    """
    Gets the path for a multiple-day aggregation data.

    See Also:
        :class:`MultiDayAggregationPath` and :class:`PdfsMultiDayAggregationPath`.

    """
    if path_obj is None:
        path_obj = SlabMultiDayAggregationPath(
            product_name=product_name,
            region=region,
            locale=locale,
            workspace=workspace,
            num_days=num_days,
            date=end_date,
            data_type=add_suffix(data_type, data_type_suffix, sep='-'),
            file=file,
            feature_flag=feature_flag,
            owner=owner,
            bucket=bucket
        )

    return str(path_obj)


# endregion

def get_pdfs_customer_index_path(
        region: Union[str, SupportedRegions],
        locale: Union[str, SupportedLocales],
        workspace: str,
        num_days: int,
        end_date: Union[str, datetime],
        feature_flag: FeatureFlag,
        data_type_suffix=None,
        file=None,
        owner: [str, SupportedOwners] = SupportedOwners.IN_HOUSE,
) -> str:
    """
    Gets the s3 paths for pDFS customer history index.
    See `get_slab_multi_day_aggregation_path` for argument definitions.

    """
    return get_slab_multi_day_aggregation_path(
        num_days=num_days,
        end_date=end_date,
        workspace=workspace,
        region=region,
        locale=locale,
        data_type=PreDefinedIndexDataTypes.CUSTOMER_HISTORY_INDEX,
        data_type_suffix=data_type_suffix,
        file=file,
        feature_flag=feature_flag,
        owner=owner
    )


def get_utterance_index_parent_path(
        region: Union[str, SupportedRegions],
        workspace: str,
        locale: Union[str, SupportedLocales],
        num_days: int,
        end_date: Union[str, datetime],
        feature_flag: FeatureFlag,
        file=None,
        owner: [str, SupportedOwners] = SupportedOwners.IN_HOUSE,
) -> str:
    index_parent_path = get_slab_multi_day_aggregation_path(
        num_days=num_days,
        end_date=end_date,
        workspace=workspace,
        region=region,
        locale=locale,
        data_type=PreDefinedIndexDataTypes.UTTERANCE_INDEX,
        file=file,
        feature_flag=feature_flag,
        owner=owner
    )
    return index_parent_path


def get_rewrite_index_parent_path(
        region: Union[str, SupportedRegions],
        workspace: str,
        locale: Union[str, SupportedLocales],
        num_days: int,
        end_date: Union[str, datetime],
        feature_flag: FeatureFlag,
        file=None,
        owner: [str, SupportedOwners] = SupportedOwners.IN_HOUSE,
) -> str:
    index_parent_path = get_slab_multi_day_aggregation_path(
        num_days=num_days,
        end_date=end_date,
        workspace=workspace,
        region=region,
        locale=locale,
        data_type=PreDefinedIndexDataTypes.REWRITE_INDEX,
        file=file,
        feature_flag=feature_flag,
        owner=owner
    )
    return index_parent_path


def get_entity_index_parent_path(
        region: Union[str, SupportedRegions],
        workspace: str,
        locale: Union[str, SupportedLocales],
        num_days: int,
        end_date: Union[str, datetime],
        feature_flag: FeatureFlag,
        file=None,
        owner: [str, SupportedOwners] = SupportedOwners.IN_HOUSE,
) -> str:
    index_parent_path = get_slab_multi_day_aggregation_path(
        num_days=num_days,
        end_date=end_date,
        workspace=workspace,
        region=region,
        locale=locale,
        data_type=PreDefinedIndexDataTypes.ENTITY_INDEX,
        file=file,
        feature_flag=feature_flag,
        owner=owner
    )
    return index_parent_path


def get_global_affinity_index_s3_path(
        region: Union[str, SupportedRegions],
        workspace: str,
        locale: Union[str, SupportedLocales],
        num_days: int,
        end_date: Union[str, datetime],
        feature_flag: FeatureFlag,
        file=None,
        owner: [str, SupportedOwners] = SupportedOwners.IN_HOUSE,
) -> str:
    return get_slab_multi_day_aggregation_path(
        num_days=num_days,
        end_date=end_date,
        workspace=workspace,
        region=region,
        locale=locale,
        data_type=PreDefinedIndexDataTypes.GLOBAL_AFFINITY_INDEX,
        file=file,
        feature_flag=feature_flag,
        owner=owner
    )


def get_customer_affinity_index_s3_path(
        region: Union[str, SupportedRegions],
        workspace: str,
        locale: Union[str, SupportedLocales],
        num_days: int,
        end_date: Union[str, datetime],
        feature_flag: FeatureFlag,
        file=None,
        owner: [str, SupportedOwners] = SupportedOwners.IN_HOUSE,
) -> str:
    return get_slab_multi_day_aggregation_path(num_days=num_days, end_date=end_date, workspace=workspace, region=region, locale=locale, data_type=PreDefinedIndexDataTypes.CUSTOMER_AFFINITY_INDEX, file=file, feature_flag=feature_flag, owner=owner)


def get_customer_history_affinity_index_s3_path(
        region: Union[str, SupportedRegions],
        workspace: str,
        locale: Union[str, SupportedLocales],
        num_days: int,
        end_date: Union[str, datetime],
        feature_flag: FeatureFlag,
        file=None,
        owner: [str, SupportedOwners] = SupportedOwners.IN_HOUSE,
) -> str:
    return get_slab_multi_day_aggregation_path(num_days=num_days, end_date=end_date, workspace=workspace, region=region, locale=locale, data_type=PreDefinedIndexDataTypes.CUSTOMER_HISTORY_AFFINITY_INDEX, file=file, feature_flag=feature_flag, owner=owner)


def get_pdfs_l2_features_s3_path(
        region: Union[str, SupportedRegions],
        workspace: str,
        locale: Union[str, SupportedLocales],
        num_days: int,
        end_date: Union[str, datetime],
        feature_flag: FeatureFlag,
        l2_feature_version='p1',
        file=None,
        owner: [str, SupportedOwners] = SupportedOwners.IN_HOUSE,
) -> str:
    return get_slab_multi_day_aggregation_path(num_days=num_days, end_date=end_date, workspace=workspace, region=region, locale=locale, data_type=f'l2_features_{l2_feature_version}', file=file, feature_flag=feature_flag, owner=owner)


def get_adjacent_turn_utterance_pair_index_path(
        region: Union[str, SupportedRegions],
        workspace: str,
        locale: Union[str, SupportedLocales],
        num_days: int,
        end_date: Union[str, datetime],
        feature_flag: FeatureFlag,
        file=None,
        owner: [str, SupportedOwners] = SupportedOwners.IN_HOUSE,
) -> str:
    return get_slab_multi_day_aggregation_path(
        num_days=num_days,
        end_date=end_date,
        workspace=workspace,
        region=region,
        locale=locale,
        data_type=PreDefinedIndexDataTypes.TURN_PAIR_INDEX,
        file=file,
        feature_flag=feature_flag,
        owner=owner
    )


def get_turn_pair_index_parent_path(
        region: Union[str, SupportedRegions],
        workspace: str,
        locale: Union[str, SupportedLocales],
        num_days: int,
        end_date: Union[str, datetime],
        feature_flag: FeatureFlag,
        file=None,
        owner: [str, SupportedOwners] = SupportedOwners.IN_HOUSE,
) -> str:
    return get_slab_multi_day_aggregation_path(num_days=num_days, end_date=end_date, workspace=workspace, region=region, locale=locale, data_type=PreDefinedIndexDataTypes.TURN_PAIR_INDEX, file=file, feature_flag=feature_flag, owner=owner)
