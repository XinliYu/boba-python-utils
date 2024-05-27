from typing import Iterable, Union

from pyspark.sql import DataFrame
from pyspark.sql.session import SparkSession

import boba_python_utils.spark_utils.common
import boba_python_utils.spark_utils.spark_functions.common
import boba_python_utils.production_utils.pdfs.constants as c
import boba_python_utils.spark_utils.spark_functions as F
import boba_python_utils.spark_utils as sparku
from boba_python_utils.general_utils.console_util import hprint_message
from boba_python_utils.general_utils.general import make_list_
from boba_python_utils.production_utils.greenwich_data.common import (
    get_provider_raw_request_traffic,
    NluMergerResultFilter,
    get_cpdr_signals_from_greenwich_nextgen, UtteranceIdMatchOptions, filter_raw_request_traffic, _solve_nlu_merger_result_filter,
)
from boba_python_utils.production_utils.pdfs.testsets.common import (
    save_turn_pairs_with_customer_history,
)


def _solve_filter(spark, filter, filter_format, filter_name):
    if filter:
        if isinstance(filter, str):
            filter = sparku.cache__(
                filter,
                spark=spark,
                input_format=filter_format,
                name=filter_name
            )
        elif isinstance(filter, (list, tuple)):
            filter = set(filter)
        elif not isinstance(filter, (set, DataFrame)):
            raise ValueError(f"unsupported {filter_name} filter '{filter}'")
        return filter


def build_traffic_dataset(
        spark: SparkSession,
        output_path: str,
        input_daily_aggregation,
        history_index_path: Union[str, Iterable[str]],
        history_index_format: str = 'parquet',
        nlu_merger_result_filter=NluMergerResultFilter.NonMergedPlusProvider,
        provider_filter=c.PROVIDER_NAME__PDFS,
        dfs_should_trigger_filter=None,
        sample_ratio=0.003,
        save_requests=True,
        save_history_index=True,
        save_combined=True,
        extra_filter=None,
        customer_id_filter: Union[str, DataFrame, Iterable[str]] = None,
        customer_id_filter_format: str = None,
        utterance_id_filter: Union[str, DataFrame, Iterable[str]] = None,
        utterance_id_filter_format: str = None,
        utterance_id_filter_match_option: Union[str, UtteranceIdMatchOptions] = UtteranceIdMatchOptions.FullUtteranceId,
        cache_option: boba_python_utils.spark_utils.common.CacheOptions = boba_python_utils.spark_utils.common.CacheOptions.IMMEDIATE,
        **kwargs
):
    # region STEP1: read pDFS raw traffic, and then either:
    #     1) subsample by `sample_ratio`; or
    #     2) extracts those of the specified `sample_utterance_ids`.
    utterance_id_filter = _solve_filter(
        spark=spark,
        filter=utterance_id_filter,
        filter_format=utterance_id_filter_format,
        filter_name='utterance_id_filter'
    )

    customer_id_filter = _solve_filter(
        spark=spark,
        filter=customer_id_filter,
        filter_format=customer_id_filter_format,
        filter_name='customer_id_filter'
    )

    df_traffic = sparku.solve_input(
        input_daily_aggregation,
        spark=spark,
        name='df_traffic'
    )
    has_dfs_should_trigger = c.KEY_DFS_SHOULD_TRIGGER in df_traffic.columns
    hprint_message('has_dfs_should_trigger', has_dfs_should_trigger)

    df_traffic = filter_raw_request_traffic(
        df_traffic,
        customer_id_filter=customer_id_filter,
        device_id_filter=None,
        utterance_id_filter=utterance_id_filter,
        provider_filter=provider_filter,
        should_trigger_filter=dfs_should_trigger_filter if has_dfs_should_trigger else None,
        nlu_merger_result_filter=nlu_merger_result_filter,
        utterance_id_match_option=utterance_id_filter_match_option,
        final_filter=None
    )

    if 0.0 < sample_ratio < 1.0:
        df_traffic_sample = df_traffic.sample(fraction=sample_ratio, seed=0)
    else:
        df_traffic_sample = df_traffic

    df_traffic_sample = sparku.cache__(
        df_traffic_sample,
        name='df_traffic_sample',
        cache_option=cache_option
    )
    # endregion

    # region STEP2: joins with global/customer stats

    df_traffic_sample_with_dummy = sparku.with_columns(
        df_traffic_sample,
        {
            c.KEY_REQUEST_SECOND: F.first_non_null(c.KEY_REPLACED_REQUEST, c.KEY_REQUEST),
            c.KEY_NLU_HYPOTHESIS_FIRST: F.first_non_null(
                c.KEY_ASR_HYPOTHESIS, c.KEY_HYPOTHESIS
            ),
            c.KEY_DOMAIN_SECOND: c.KEY_DOMAIN,  # dummy
            c.KEY_INTENT_SECOND: c.KEY_INTENT,  # dummy
            c.KEY_RESPONSE_SECOND: c.KEY_RESPONSE,  # dummy
        },
    )
    df_traffic_sample_with_dummy = sparku.rename(
        df_traffic_sample_with_dummy,
        {
            c.KEY_REQUEST: c.KEY_REQUEST_FIRST,
            c.KEY_HYPOTHESIS: c.KEY_NLU_HYPOTHESIS_SECOND,
            c.KEY_DOMAIN: c.KEY_DOMAIN_FIRST,
            c.KEY_INTENT: c.KEY_INTENT_FIRST,
            c.KEY_RESPONSE: c.KEY_RESPONSE_FIRST,
        },
    )

    if extra_filter is not None:
        df_traffic_sample_with_dummy = df_traffic_sample_with_dummy.where(extra_filter)

    df_traffic_sample_with_dummy = sparku.cache__(
        df_traffic_sample_with_dummy,
        name='df_traffic_sample_with_dummy',
        cache_option=cache_option,
        unpersist=df_traffic_sample,
    )

    # endregion

    # region STEP3: joins with history

    out = save_turn_pairs_with_customer_history(
        spark=spark,
        df_rephrase=df_traffic_sample_with_dummy,
        customer_history_index_dataframes_or_paths=history_index_path,
        output_paths=output_path,
        customer_history_index_format=history_index_format,
        save_requests=save_requests,
        save_history_index=save_history_index,
        save_combined=save_combined,
    )

    df_traffic_sample_with_dummy.unpersist()
    # endregion

    return out[0] if len(out) == 1 else out
