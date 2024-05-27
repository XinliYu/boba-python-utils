from abc import ABC
from datetime import datetime
from os import path
from types import MappingProxyType
from typing import Mapping
from typing import Union, List

from attr import attrib, attrs

from boba_python_utils.path_utils.path_with_date_time_info import get_path_with_year_month_day
from boba_python_utils.production_utils.common.constants import (
    DEFAULT_BUCKET,
    AGGREGATION_ROOT_PATH_PATTERN,
    COMMON_PREFIX_DICT,
    AGGREGATION_PATH_PATTERN_SUFFIX,
    PreDefinedBuckets,
    SupportedOwners,
    SupportedRegions,
    PreDefinedWorkspaces,
    SupportedLocales,
    CROSS_REGION_BUCKETS
)


# region aggregation path formation

def get_cross_region_bucket(source_region, target_region):
    if source_region and target_region and source_region != target_region:
        return CROSS_REGION_BUCKETS[(source_region, target_region)]


def get_aggregation_path_prefix(
        product_name: str,
        bucket: Union[str, PreDefinedBuckets],
        region: Union[str, SupportedRegions],
        owner: Union[str, SupportedOwners] = SupportedOwners.IN_HOUSE,
        path_prefix_dict: Mapping = None
) -> str:
    """
    Gets a pre-defined path prefix from the provided `path_prefix_dict`
    according to `bucket`, `owner` and `region`,
    and joint this prefix with `root_bucket_name`.

    Args:
        product_name: the product name for all aggregations under the returned prefix,
            e.g. pDFS, FIG. This retrieved path prefix from `path_prefix_dict` will join
            with the product name to form the final path prefix.
        region: the region of the aggregation, e.g. 'na', 'eu'.
        owner: aggregation owner, e.g. 'in_house', 'aeslu'; see `SupportedOwners`;
            aggregations from different owners can have
            different path formats and aggregation mechanism.
        bucket: the name of the bucket where the aggregation is saved.
        path_prefix_dict: a 3-layer dictionary registering path prefixes
            under different `bucket`, `owner` and `region`.

    Returns: a path prefix retrieved from `path_prefix_dict`, joint with `root_bucket_name`.

    """
    if not product_name:
        raise ValueError("must provide 'root_bucket_name'")
    if path_prefix_dict is None:
        path_prefix_dict = COMMON_PREFIX_DICT
    if not bucket:
        bucket = DEFAULT_BUCKET

    if bucket in path_prefix_dict:
        path_prefix_dict = path_prefix_dict[bucket]
        if owner not in path_prefix_dict:
            raise ValueError(f"the specified owner '{owner}' is not supported")
        path_prefix_dict = path_prefix_dict[owner]
        if region not in path_prefix_dict:
            raise ValueError(f"the specified region '{region}' is not supported yet")
        return path.join(path_prefix_dict[region], product_name)
    else:
        raise ValueError(f"does not support the specified bucket '{bucket}'")


def get_aggregation_root_path(
        prefix: str,
        workspace: Union[str, PreDefinedWorkspaces],
        data: str,
        locale: Union[str, SupportedLocales],
        version: str = None,
        root_path_pattern: str = AGGREGATION_ROOT_PATH_PATTERN
) -> str:
    """
    Gets root path for a data aggregation.

    Args:
        prefix: the path prefix for the aggregation. See `get_aggregation_path_prefix`.
        workspace: the name of the folder right after the prefix to save the aggregation data
            (e.g. master, prod, debug).
        data: the name for the aggregation data.
        locale: the locale of the aggregation.
        version: the version for the aggregation data; if not specified, the returned root path
            will not contain the version part.
        root_path_pattern: the format pattern of the root path,
            must contain keys 'prefix', 'data', 'workspace', 'version', 'locale';
            e.g. '{prefix}/{workspace}/{data}/{version}/{locale}'.

    Returns: the root path for a data aggregation.

    """
    if not version:
        root_path_pattern = root_path_pattern.replace((path.sep + '{version}'), '')

    return root_path_pattern.format(
        prefix=prefix,
        data=data,
        workspace=workspace,
        version=version,
        locale=locale
    )


def get_dated_aggregation_path(
        root_path: str,
        end_date: Union[str, datetime],
        num_days: int = None,
        file=None
) -> Union[str, List[str]]:
    """
    Gets one or more dated path to an aggregation.

    We joint `root_path` with sub path of format '/{year}/{month}/{day}/{file}',
    where 'year', 'month', 'day' are determined by `end_date`.

    If `num_days` is specified, we return a number of paths
    with consecutive dates starting from 'date - num_days_backward' (exclusive)
    until `date` (inclusive).
    """

    return get_path_with_year_month_day(
        path_pattern=root_path + AGGREGATION_PATH_PATTERN_SUFFIX,
        date=end_date,
        num_days_backward=num_days,
        file=(file or '')
    )


# endregion

# region daily aggregation & multi-day aggregation paths
@attrs(slots=True)
class AggregationPath:
    """
    Represents the path to a multi-day aggregation.

    Attributes:
        data_type: the data type of the aggregation, e.g. 'utterance_index', 'customer_history'.
        date: the interpretation depends on implementation of `get_path` method;
            usually is a boundary date of the time range the returned aggregation path should cover.
        num_days: number of days the returned aggregation path
            by `get_path` method should cover;
            the concrete interpretation depends on implementation of `get_path` method.
        workspace: the workspace for the aggregation (e.g. prod, master, science);
            workspace is used as one part of the aggregation path
            to identify data for difference purposes,
            for example 'prod' for production use,
            and 'science' for offline science use.
        region: the region for the aggregation (e.g. 'na', 'eu').
        locale: the locale for the aggregation (e.g. en_US, en_GB).
        file: if specified, this `file` will be the last path part.
        owner: aggregation owner, e.g. 'in_house', 'aeslu'; see `SupportedOwners`;
            aggregations from different owners can have
            different path formats and aggregation mechanism.
        bucket: the name of the bucket where the aggregation is saved.
        exists_path: a Callable function to determine if a path exists.
    """
    data_type = attrib(type=str, default=None)
    num_days = attrib(type=int, default=None)
    date = attrib(type=Union[str, datetime], default=None)
    workspace = attrib(type=Union[str, PreDefinedWorkspaces, Mapping[str, str]], default=None)
    region = attrib(type=Union[str, SupportedRegions], default=None)
    locale = attrib(type=Union[str, SupportedLocales], default=None)
    file = attrib(type=str, default=None)
    owner = attrib(type=Union[str, SupportedOwners], default=SupportedOwners.IN_HOUSE)
    bucket = attrib(type=[str, PreDefinedBuckets], default=None)

    def get_path(self):
        """
        Override this function to return a non-empty aggregation path.
        """
        pass

    def __str__(self):
        return self.get_path()


@attrs(slots=True)
class DailyAggregationPath(AggregationPath, ABC):
    """
    Represents one or more paths of a daily aggregation.

    Attributes:
        date: the date of the daily aggregation.
        num_days: is specified, returns a number of paths
            with consecutive dates starting from 'date - num_days_backward' (exclusive)
            until `date` (inclusive).

    See Also:
        :class:`AggregationPath`.
    """

    def _get_aggregation_root_path(self, data: str) -> str:
        raise NotImplementedError

    def get_path(self) -> str:
        data = path.join('daily', self.data_type)
        return get_dated_aggregation_path(
            root_path=self._get_aggregation_root_path(data),
            end_date=self.date,
            num_days=self.num_days,
            file=self.file
        )


@attrs(slots=True)
class MultiDayAggregationPath(AggregationPath, ABC):
    """
    Represents the path to a multi-day aggregation.

    Attributes:
        num_days: how many days the multi-day aggregation covers.
        end_date: the end date (inclusive) of the aggregation.

    See Also:
        :class:`AggregationPath`.
    """

    def _get_aggregation_root_path(self, data: str) -> str:
        raise NotImplementedError

    def get_path(self) -> str:
        num_days = '1day' if self.num_days == 1 else f'{self.num_days}days'
        data = path.join(self.data_type, num_days)
        return get_dated_aggregation_path(
            root_path=self._get_aggregation_root_path(data),
            end_date=self.date,
            num_days=None,
            file=self.file
        )


# endregion

# region misc
def get_language_from_locale(locale: Union[str, SupportedLocales]):
    return locale.split('_')[0]


def get_path_prefix_dict(
        *prefix_entries: Mapping,
        base_prefix_dict: Mapping = COMMON_PREFIX_DICT
):
    base_prefix_dict = dict(base_prefix_dict)
    for entry in prefix_entries:
        prefix_dict = base_prefix_dict
        bucket, owner, region, prefix = (
            entry['bucket'],
            entry['owner'],
            entry['region'],
            entry['prefix']
        )
        if bucket not in prefix_dict:
            prefix_dict[bucket] = {}
        prefix_dict = prefix_dict[bucket]
        if owner not in prefix_dict:
            prefix_dict[owner] = {}
        prefix_dict = prefix_dict[owner]
        prefix_dict[region] = prefix
    return MappingProxyType(base_prefix_dict)

# endregion
