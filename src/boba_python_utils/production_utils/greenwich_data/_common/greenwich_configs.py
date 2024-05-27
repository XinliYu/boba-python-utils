from typing import Union, Iterable

from attr import attrib, attrs
from pyspark.sql import DataFrame

from boba_python_utils.common_utils.iter_helper import iter__
from boba_python_utils.spark_utils.spark_functions.common import first_non_null
from boba_python_utils.production_utils.greenwich_data._constants.common_data_keys import NAME_PREFIX_CUSTOMER_PAIR_AVG, NAME_PREFIX_GLOBAL_PAIR_AVG

from boba_python_utils.production_utils.greenwich_data.constants import (
    NAME_PREFIX_GLOBAL_AVG,
    NAME_PREFIX_CUSTOMER_AVG,
    KEY_CPDR7,
    KEY_DEFECT,
    KEY_CPDR7_1,
    KEY_SESSION_DEFECT,
    KEY_SESSION_DEFECT7,
    KEY_SESSION_DEFECT7_1,
    KEY_CPDR6,
    KEY_SESSION_DEFECT6,
    NAME_PREFIX_SUM,
    NAME_SUFFIX_FIRST,
    NAME_SUFFIX_SECOND,
)
from boba_python_utils.production_utils.pdfs.index._constants.customer_history import (
    DEFAULT_CUSTOMER_HISTORY_FILTER_MIN_GLOBAL_COUNT,
    DEFAULT_CUSTOMER_HISTORY_FILTER_MAX_GLOBAL_AVG_DEFECT,
    DEFAULT_CUSTOMER_HISTORY_FILTER_MAX_CUSTOMER_AVG_DEFECT_FOR_GLOBAL_SELECTION,
    DEFAULT_CUSTOMER_HISTORY_FILTER_MIN_CUSTOMER_COUNT_FOR_PERSONALIZED_SELECTION_LEVEL2,
    DEFAULT_CUSTOMER_HISTORY_FILTER_MAX_CUSTOMER_AVG_DEFECT,
    DEFAULT_CUSTOMER_HISTORY_FILTER_MAX_GLOBAL_AVG_DEFECT_FOR_PERSONALIZED_SELECTION,
    DEFAULT_CUSTOMER_HISTORY_FILTER_MIN_GLOBAL_COUNT_EXT_PERSONALIZED_SELECTION,
    DEFAULT_CUSTOMER_HISTORY_FILTER_MAX_GLOBAL_AVG_DEFECT_EXT_PERSONALIZED_SELECTION,
    DEFAULT_CUSTOMER_HISTORY_FILTER_MAX_SIZE,
    DEFAULT_CUSTOMER_HISTORY_BLOCKED_PROVIDERS,
    DEFAULT_CUSTOMER_HISTORY_BLOCKED_DOMAINS,
    DEFAULT_CUSTOMER_HISTORY_FILTER_MIN_CUSTOMER_COUNT_FOR_PERSONALIZED_SELECTION_LEVEL1,
    DEFAULT_CUSTOMER_HISTORY_FILTER_MIN_CUSTOMER_COUNT_EXT_PERSONALIZED_SELECTION, DEFAULT_CUSTOMER_HISTORY_FILTER_MAX_CUSTOMER_AVG_DEFECT_EXT_PERSONALIZED_SELECTION
)

from typing import Union, Iterable

from attr import attrib, attrs
from pyspark.sql import DataFrame

from boba_python_utils.common_utils.iter_helper import iter__
from boba_python_utils.spark_utils.spark_functions.common import first_non_null
from boba_python_utils.production_utils.greenwich_data._constants.common_data_keys import NAME_PREFIX_CUSTOMER_PAIR_AVG, NAME_PREFIX_GLOBAL_PAIR_AVG

from boba_python_utils.production_utils.greenwich_data.constants import (
    NAME_PREFIX_GLOBAL_AVG,
    NAME_PREFIX_CUSTOMER_AVG,
    KEY_CPDR7,
    KEY_DEFECT,
    KEY_CPDR7_1,
    KEY_SESSION_DEFECT,
    KEY_SESSION_DEFECT7,
    KEY_SESSION_DEFECT7_1,
    KEY_CPDR6,
    KEY_SESSION_DEFECT6,
    NAME_PREFIX_SUM,
    NAME_SUFFIX_FIRST,
    NAME_SUFFIX_SECOND,
)
from boba_python_utils.production_utils.pdfs.index._constants.customer_history import (
    DEFAULT_CUSTOMER_HISTORY_FILTER_MIN_GLOBAL_COUNT,
    DEFAULT_CUSTOMER_HISTORY_FILTER_MAX_GLOBAL_AVG_DEFECT,
    DEFAULT_CUSTOMER_HISTORY_FILTER_MAX_CUSTOMER_AVG_DEFECT_FOR_GLOBAL_SELECTION,
    DEFAULT_CUSTOMER_HISTORY_FILTER_MIN_CUSTOMER_COUNT_FOR_PERSONALIZED_SELECTION_LEVEL2,
    DEFAULT_CUSTOMER_HISTORY_FILTER_MAX_CUSTOMER_AVG_DEFECT,
    DEFAULT_CUSTOMER_HISTORY_FILTER_MAX_GLOBAL_AVG_DEFECT_FOR_PERSONALIZED_SELECTION,
    DEFAULT_CUSTOMER_HISTORY_FILTER_MIN_GLOBAL_COUNT_EXT_PERSONALIZED_SELECTION,
    DEFAULT_CUSTOMER_HISTORY_FILTER_MAX_GLOBAL_AVG_DEFECT_EXT_PERSONALIZED_SELECTION,
    DEFAULT_CUSTOMER_HISTORY_FILTER_MAX_SIZE,
    DEFAULT_CUSTOMER_HISTORY_BLOCKED_PROVIDERS,
    DEFAULT_CUSTOMER_HISTORY_BLOCKED_DOMAINS,
    DEFAULT_CUSTOMER_HISTORY_FILTER_MIN_CUSTOMER_COUNT_FOR_PERSONALIZED_SELECTION_LEVEL1,
    DEFAULT_CUSTOMER_HISTORY_FILTER_MIN_CUSTOMER_COUNT_EXT_PERSONALIZED_SELECTION, DEFAULT_CUSTOMER_HISTORY_FILTER_MAX_CUSTOMER_AVG_DEFECT_EXT_PERSONALIZED_SELECTION
)

class CpdrConfig:
    # ! DEPRECATED after unified Greenwich

    def __init__(
            self,
            greenwich_version_label: str = 'unified',
            utterance_defect_colname: str = KEY_DEFECT,
            session_defect_colname: str = KEY_SESSION_DEFECT,
    ):
        self.greenwich_version_label = greenwich_version_label
        self.utterance_defect_colname = utterance_defect_colname
        self.session_defect_colname = session_defect_colname
        self.utterance_defect_first_colname = utterance_defect_colname + NAME_SUFFIX_FIRST
        self.utterance_defect_second_colname = utterance_defect_colname + NAME_SUFFIX_SECOND
        self.utterance_global_defect_first_colname = NAME_PREFIX_GLOBAL_AVG + utterance_defect_colname + NAME_SUFFIX_FIRST
        self.utterance_global_defect_second_colname = NAME_PREFIX_GLOBAL_AVG + utterance_defect_colname + NAME_SUFFIX_SECOND
        self.utterance_customer_defect_first_colname = NAME_PREFIX_CUSTOMER_AVG + utterance_defect_colname + NAME_SUFFIX_FIRST
        self.utterance_customer_defect_second_colname = NAME_PREFIX_CUSTOMER_AVG + utterance_defect_colname + NAME_SUFFIX_SECOND
        self.utterance_global_pair_defect_first_colname = NAME_PREFIX_GLOBAL_PAIR_AVG + utterance_defect_colname + NAME_SUFFIX_FIRST
        self.utterance_global_pair_defect_second_colname = NAME_PREFIX_GLOBAL_PAIR_AVG + utterance_defect_colname + NAME_SUFFIX_SECOND
        self.utterance_customer_pair_defect_first_colname = NAME_PREFIX_CUSTOMER_PAIR_AVG + utterance_defect_colname + NAME_SUFFIX_FIRST
        self.utterance_customer_pair_defect_second_colname = NAME_PREFIX_CUSTOMER_PAIR_AVG + utterance_defect_colname + NAME_SUFFIX_SECOND
        self.sum_utterance_defect_colname = NAME_PREFIX_SUM + utterance_defect_colname


@attrs(slots=True)
class CustomerHistoryFilterConfig:
    # region for global selection
    min_global_count = attrib(
        type=int, default=DEFAULT_CUSTOMER_HISTORY_FILTER_MIN_GLOBAL_COUNT
    )
    max_global_avg_defect = attrib(
        type=float, default=DEFAULT_CUSTOMER_HISTORY_FILTER_MAX_GLOBAL_AVG_DEFECT
    )
    max_customer_avg_defect_global_selection = attrib(
        type=float,
        default=DEFAULT_CUSTOMER_HISTORY_FILTER_MAX_CUSTOMER_AVG_DEFECT_FOR_GLOBAL_SELECTION
    )
    # endregion

    # for personalized selection
    min_customer_count_for_personalized_selection_level2 = attrib(
        type=int, default=DEFAULT_CUSTOMER_HISTORY_FILTER_MIN_CUSTOMER_COUNT_FOR_PERSONALIZED_SELECTION_LEVEL2
    )
    max_customer_avg_defect = attrib(
        type=float, default=DEFAULT_CUSTOMER_HISTORY_FILTER_MAX_CUSTOMER_AVG_DEFECT
    )
    max_global_avg_defect_personalized_selection = attrib(
        type=float,
        default=DEFAULT_CUSTOMER_HISTORY_FILTER_MAX_GLOBAL_AVG_DEFECT_FOR_PERSONALIZED_SELECTION
    )
    # endregion

    # region for extended personalized selection
    enable_extended_personalized_selection = attrib(type=bool, default=True)
    min_customer_count_extended_personalized_selection = attrib(
        type=int,
        default=DEFAULT_CUSTOMER_HISTORY_FILTER_MIN_CUSTOMER_COUNT_EXT_PERSONALIZED_SELECTION
    )
    max_customer_avg_defect_extended_personalized_selection = attrib(
        type=float, default=DEFAULT_CUSTOMER_HISTORY_FILTER_MAX_CUSTOMER_AVG_DEFECT_EXT_PERSONALIZED_SELECTION
    )
    min_global_count_extended_personalized_selection = attrib(
        type=int,
        default=DEFAULT_CUSTOMER_HISTORY_FILTER_MIN_GLOBAL_COUNT_EXT_PERSONALIZED_SELECTION
    )
    max_global_avg_defect_extended_personalized_selection = attrib(
        type=float,
        default=DEFAULT_CUSTOMER_HISTORY_FILTER_MAX_GLOBAL_AVG_DEFECT_EXT_PERSONALIZED_SELECTION
    )
    # endregion

    # region misc
    max_hist_size = attrib(type=int, default=DEFAULT_CUSTOMER_HISTORY_FILTER_MAX_SIZE)
    blocked_providers = attrib(type=set, default=DEFAULT_CUSTOMER_HISTORY_BLOCKED_PROVIDERS)
    blocked_domains_for_global_selection = attrib(
        type=set, default=DEFAULT_CUSTOMER_HISTORY_BLOCKED_DOMAINS
    )
    blocked_domains_for_personalized_selection = attrib(
        type=set, default=DEFAULT_CUSTOMER_HISTORY_BLOCKED_DOMAINS
    )
    # endregion

    # region legacy
    min_customer_count_for_personalized_selection_level1 = attrib(
        type=int, default=DEFAULT_CUSTOMER_HISTORY_FILTER_MIN_CUSTOMER_COUNT_FOR_PERSONALIZED_SELECTION_LEVEL1
    )
    # endregion


CPDR_UNIFIED_CONFIG = CpdrConfig(
    greenwich_version_label='unified',
    utterance_defect_colname=KEY_DEFECT,
    session_defect_colname=KEY_SESSION_DEFECT,
)
CPDR7_1_CONFIG = CpdrConfig(
    greenwich_version_label='7_0_1',
    utterance_defect_colname=KEY_CPDR7_1,
    session_defect_colname=KEY_SESSION_DEFECT7_1,
)
CPDR7_CONFIG = CpdrConfig(
    greenwich_version_label='7_0_0',
    utterance_defect_colname=KEY_CPDR7,
    session_defect_colname=KEY_SESSION_DEFECT7,
)
CPDR6_CONFIG = CpdrConfig(
    greenwich_version_label='6_0_0',
    utterance_defect_colname=KEY_CPDR6,
    session_defect_colname=KEY_SESSION_DEFECT6,
)


def solve_data_field_names(
        df_traffic: DataFrame,
        cpdr_config: CpdrConfig = None,
        backward_compatible_cpdr_configs: Union[CpdrConfig, Iterable[CpdrConfig]] = None
) -> DataFrame:
    """

    Solves CPDR column name backward compatibility issue for
    utterance CPD signal `cpdr_config.utterance_defect_colname`
    and session CPD signal `cpdr_config.session_defect_colname`.

    On 06/30/2022 we move to unified greenwich, and change CPD signal column name from 'cpdr7'
    to just 'defect', and 'session_defect7' to 'session_defect'; but when we load utterance data
    from before 06/30 and after 06/30, we would have two sets of CPD columns.
    This function merges the two column set as a single CPD column set defined in `cpdr_config`.

    TODO: in the future this function will be the general place to solve data field name conflict.

    Args:
        df_traffic: the utterance dataframe.
        cpdr_config: the current CPDR configuration.
        backward_compatible_cpdr_configs: CPDR configurations we

    Returns: the `df_traffic` dataframe with CPD signal columns renamed.

    """

    def _rename_defect_colnames(defect_field_name):
        nonlocal df_traffic
        _defect_col_names = [
            getattr(_cpdr_config, defect_field_name)
            for _cpdr_config in iter__(backward_compatible_cpdr_configs)
            if (_cpdr_config.greenwich_version_label != cpdr_config.greenwich_version_label and
                getattr(_cpdr_config, defect_field_name) in df_traffic.columns)
        ]
        if _defect_col_names:
            defect_col_names = _defect_col_names
            new_defect_colname = getattr(cpdr_config, defect_field_name)
            if new_defect_colname in df_traffic.columns:
                defect_col_names = [new_defect_colname] + _defect_col_names
            df_traffic = df_traffic.withColumn(
                new_defect_colname,
                first_non_null(*defect_col_names)
            ).drop(*_defect_col_names)

    if backward_compatible_cpdr_configs is not None:
        _rename_defect_colnames('utterance_defect_colname')
        _rename_defect_colnames('session_defect_colname')
        _rename_defect_colnames('sum_utterance_defect_colname')
        _rename_defect_colnames('utterance_defect_first_colname')
        _rename_defect_colnames('utterance_defect_second_colname')

    return df_traffic.drop('_corrupt_record')
