from datetime import datetime
from typing import Union

from boba_python_utils.path_utils.path_with_date_time_info import get_path_with_year_month_day
from boba_python_utils.production_utils.alexabrain_data.constants import ALEXA_BRAIN_DATA_PATH_PATTERNS
from boba_python_utils.production_utils.common.constants import SupportedRegions


def get_alexa_brain_data_path(
        date: Union[str, datetime],
        region: Union[str, SupportedRegions] = SupportedRegions.NA,
        num_days_backward: int = None,
):
    return get_path_with_year_month_day(
        path_pattern=ALEXA_BRAIN_DATA_PATH_PATTERNS[region],
        date=date,
        hour='*',
        file='*',
        num_days_backward=num_days_backward,
    )
