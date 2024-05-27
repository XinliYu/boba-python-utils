from datetime import datetime
from typing import Union, List
from os import path

from boba_python_utils.production_utils.common._constants.path import PreDefinedBuckets, SupportedRegions
from boba_python_utils.production_utils.common.constants import SupportedLocales
from boba_python_utils.production_utils.common.path import (
    get_aggregation_root_path,
    get_dated_aggregation_path, get_aggregation_path_prefix
)
from boba_python_utils.production_utils.pdfs.index._common.feature_flag import (  # noqa: F401,F403,E501
    FeatureFlag,
    SupportedCPDRVersions,
)


def get_graph_aggregation_root_path(
        prefix: str,
        workspace: str,
        data: str,
        locale: Union[str, SupportedLocales],
        version: str = None,
        add_num_days_info_to_version: Union[int, bool] = False
) -> str:
    """
    Gets root path for a graph aggregation.
    Args:
        prefix: prefix specified for aggregation root path.
        workspace: the name of the folder right after the prefix to save the aggregation data
            (e.g. master, prod, debug).
        data: the name of the aggregation data (e.g. 'graph_triplets').
        locale: locale specified for aggregation.
        version: optionally provides a version string, which will be part of the returned root path,
            (e.g. 'recipe_graph/60days').
        add_num_days_info_to_version: specifies a non-negative integer to adds a path part
            indicating how many days' data the aggregation has to the `version` string;
            if this argument is 1, then 'daily' will be appended to the `version`
            (e.g. 'graph_triplets/daily');
            if this argument is larger than 1, then 'xdays' will be appended to the `version`
            (e.g. 'graph_triplets/60days').

    Returns: s3 root path for the graph aggregation.
    """

    return get_aggregation_root_path(
        prefix=prefix,
        data=data,
        workspace=workspace,
        version=version,
        locale=locale,
        add_num_days_info_to_version=add_num_days_info_to_version
    )


def get_graph_aggregation_path(
        workspace: str,
        region: Union[str, SupportedRegions],
        locale: Union[str, SupportedLocales],
        data: str,
        date: Union[str, datetime],
        version: str = None,
        multi_day_data: bool = True,
        num_days_backward: int = None,
        file=None,
) -> Union[str, List[str]]:
    """
    Gets a path to save the graph aggregation data.
    Args:
        prefix: prefix specified for aggregation.
        workspace: the name of the folder right after the prefix to save the aggregation data
            (e.g. master, prod, debug).
        data: the name of the aggregation data (e.g. 'graph_triplets').
        locale: locale specified for aggregation.
        date: specific day to run daily pre-aggregation for.
        version: optionally provides a version string, which will be part of the returned s3 path.
        num_days_backward: specifies a positive integer to get a list of paths,
            starting from 'date - num_days_backward' (exclusive) until 'date' (inclusive).
        file: if specified, this 'file' be appended to the end of the path,
            so the returned will point to a file.

    Returns: path to write graph aggregation data.
    """

    return get_dated_aggregation_path(
        root_path=get_graph_aggregation_root_path(
            prefix=get_aggregation_path_prefix(
                product_name='slab_graphs',
                bucket=PreDefinedBuckets.AbmlWorkspaces,
                region=region
            ),
            workspace=workspace,
            data=data,
            locale=locale,
            version=version,
            add_num_days_info_to_version=(num_days_backward or 1) if multi_day_data else False,
        ),
        end_date=date,
        num_days=None if multi_day_data else num_days_backward,
        file=file
    )
