from datetime import datetime
from typing import Mapping, Union

from boba_python_utils.production_utils.pdfs.constants import (
    GwVersion,
    GREENWICH_PATH_PATTERNS,
)
from boba_python_utils.production_utils.common.constants import SupportedRegions
from boba_python_utils.path_utils.path_with_date_time_info import (
    get_path_with_year_month_day,
)


def get_greenwich_s3_path(
    date: Union[str, datetime],
    version: GwVersion = GwVersion.GREENWICH3,
    region: Union[str, SupportedRegions] = SupportedRegions.NA,
    num_days_backward: int = None,
):
    """
    Gets Greenwich s3 path(S) to the specified `date`, `version` and `region`.
    If `num_days_backward` is specified,
        then a list of paths will be returned starting
        for date 'date - num_days_backward' (exclusive) until the 'date' (inclusive).

    Args:
        date: gets the s3 path to Greenwich data of this specified day;
            or a list of paths ending at this specified day if `num_days_backward` is specified;
            the parameter can be specified in a string of date format '%m/%d/%Y'.
        version: specifies one of the pre-defined Greenwich versions;
            currently support 'greenwich3', 'greenwich_nextgen'.
        region: specifies the region of the Greenwich data.
        num_days_backward: specifies a positive integer to get a list of paths,
                starting from 'date - num_days_backward' (exclusive) until 'date' (inclusive).

    Returns:

    """
    greenwich_path_patterns: Mapping = GREENWICH_PATH_PATTERNS
    if version not in greenwich_path_patterns:
        raise ValueError(f"the specified Greenwich version '{version}' is not supported")
    greenwich_path_patterns = greenwich_path_patterns[version]
    if region not in greenwich_path_patterns:
        raise ValueError(f"the specified region '{region}' is not supported yet")

    greenwich_path_pattern = greenwich_path_patterns[region]
    return get_path_with_year_month_day(
        path_pattern=greenwich_path_pattern,
        date=date,
        hour='*',
        file='*',
        num_days_backward=num_days_backward,
    )
