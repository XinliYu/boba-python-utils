from typing import Iterable, Union, List, Tuple

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.column import Column
from enum import Enum

from pyspark.sql.types import StringType

import boba_python_utils.spark_utils.common
import boba_python_utils.production_utils.greenwich_data.constants as c
import boba_python_utils.spark_utils as sparku
from boba_python_utils.general_utils.console_util import hprint_message, wprint_message
from boba_python_utils.spark_utils.common import repartition, CacheOptions
from boba_python_utils.spark_utils.typing import NameOrColumn
from boba_python_utils.production_utils.common.constants import SupportedLocales
from boba_python_utils.production_utils.common.device_type import (
    get_known_no_screen_non_tv_device_types,
    get_echo_show_device_types,
    get_device_external_name_udf
)
from boba_python_utils.production_utils.greenwich_data.filters import (  # noqa: F401,E501
    solve_filter,
)
from boba_python_utils.production_utils.greenwich_data.schemas import (
    SCHEMA_GREENWICH3,
    SCHEMA_GREENWICH_NEXTGEN,
)
from abi_python_commons.schemas.flare_metrics_schemas import flare_metrics_schema as SCHEMA_GREENWICH_UNIFIED
from boba_python_utils.production_utils.greenwich_data._common.greenwich_common_data_process import *  # noqa: F401,F403,E501
from boba_python_utils.production_utils.greenwich_data._common.greenwich_paths import *  # noqa: F401,F403,E501
from boba_python_utils.production_utils.greenwich_data._common.greenwich_configs import *  # noqa: F401,F403,E501
from boba_python_utils.production_utils.greenwich_data._common.greenwich_turn_pairs import *  # noqa: F401,F403,E501

from boba_python_utils.general_utils.general import make_set
import boba_python_utils.spark_utils.spark_functions as F
from boba_python_utils.production_utils.greenwich_data._common.greenwich_configs import (  # noqa: E501
    CpdrConfig,
)
from boba_python_utils.production_utils.greenwich_data._common.greenwich_common_data_process import (  # noqa: E501
    bool2str,
    extract_resolved_slots_udf,
    extract_merger_rule_udf,
    get_provider_udf,
    get_device_id_from_utterance_id_udf,
    get_device_type_from_utterance_id_udf,
    extract_hypothesis_from_asr_aus_interpretation_udf,
    extract_utterance_from_asr_aus_interpretation_udf,
    extract_session_defect_for_next_gen_udf,
    extract_defect_from_nextgen__,
    generate_hypothesis_udf,
)


# region raw data

class UtteranceIdMatchOptions(str, Enum):
    """
    FullUtteranceId: use the entire utterance id for matching; use this when the whole utterance id
        is available.
    HeadAndTailSection: use the head/tail sections of the utterance id for matching; use this
        when some middle parts of the utterance id is masked.
    HeadAndTailSectionWithoutTurnIndex: use the head/tail sections of the utterance id for matching,
        and for the tail section we ignore the turn index; use this when some middle parts of
        the utterance id is masked, and the turn index in the tail section is also not available.
    """
    FullUtteranceId = 'full'
    HeadAndTailSection = 'head_tail'
    HeadAndTailSectionWithoutTurnIndex = 'head_tail_no_turn_index'


def process_utterance_id_column_for_matching(uid_col, uid_match_option: UtteranceIdMatchOptions):
    if uid_match_option == UtteranceIdMatchOptions.FullUtteranceId:
        return uid_col
    elif uid_match_option == UtteranceIdMatchOptions.HeadAndTailSection:
        uid_col_for_match = F.concat_ws(
            ':',
            F.split(uid_col, r'\:').getItem(0),
            F.element_at(F.split(uid_col, r'\:'), -1)
        )

    elif uid_match_option == UtteranceIdMatchOptions.HeadAndTailSectionWithoutTurnIndex:
        uid_col_for_match = F.concat_ws(
            ':',
            F.split(uid_col, r'\:').getItem(0),
            F.split(F.element_at(F.split(uid_col, r'\:'), -1), '/').getItem(0)
        )
    else:
        raise ValueError(
            f"unrecognized 'utterance_id_match_option' argument {uid_match_option}"
        )
    return uid_col_for_match


def filter_raw_request_traffic(
        df,
        customer_id_filter,
        device_id_filter,
        utterance_id_filter,
        provider_filter,
        should_trigger_filter,
        nlu_merger_result_filter,
        final_filter,
        utterance_id_match_option: UtteranceIdMatchOptions = UtteranceIdMatchOptions.FullUtteranceId,
):
    nlu_merger_result_filter, provider_filter, final_filter = _solve_nlu_merger_result_filter(
        nlu_merger_result_filter, provider_filter, final_filter
    )
    if (
            should_trigger_filter is not None
            and should_trigger_filter != ''
            and should_trigger_filter != 'none'
    ):
        should_trigger_filter = bool2str(should_trigger_filter)
        should_trigger_filter = (F.col(c.KEY_DFS_SHOULD_TRIGGER) == should_trigger_filter)
        hprint_message('should_trigger_filter', should_trigger_filter)
        df = df.where(should_trigger_filter)

    if (
            nlu_merger_result_filter is not None
            and nlu_merger_result_filter != ''
            and nlu_merger_result_filter != 'none'
    ):
        nlu_merger_result_filter = bool2str(nlu_merger_result_filter)
        nlu_merger_result_filter = (F.col(c.KEY_NLU_MERGER_RESULT) == nlu_merger_result_filter)
        hprint_message('nlu_merger_result_filter', nlu_merger_result_filter)
        df = df.where(nlu_merger_result_filter)

    if (
            provider_filter is not None
            and provider_filter != ''
            and provider_filter != 'none'
    ):
        df = sparku.filter_by_inner_join_on_columns(
            df, df_filter=provider_filter, colnames=[c.KEY_PROVIDER_NAME],
            filter_name='provider_filter'
        )

    if (
            customer_id_filter is not None
            and customer_id_filter != ''
            and customer_id_filter != 'none'
    ):
        df = sparku.filter_by_inner_join_on_columns(
            df, df_filter=customer_id_filter, colnames=[c.KEY_CUSTOMER_ID],
            filter_name='customer_id_filter'
        )

    if (
            device_id_filter is not None
            and device_id_filter != ''
            and device_id_filter != 'none'
    ):
        df = sparku.filter_by_inner_join_on_columns(
            df, df_filter=device_id_filter, colnames=[c.KEY_DEVICE_ID],
            filter_name='device_id_filter'
        )

    if (
            utterance_id_filter is not None
            and utterance_id_filter != ''
            and utterance_id_filter != 'none'
    ):
        if utterance_id_match_option != UtteranceIdMatchOptions.FullUtteranceId:
            # This is the case when the utterance id is only partially available, e.g.
            # 'A1RABVCI4QCIKC:MASKED/52:43::TNIH_2V.89cd649f-363e-454c-91e0-80967c15b1eaZXV',
            # but it is still able to uniquely identify a customer query with its available parts;

            # This requires the head and tail section of the utterance id are available.

            uid_col_for_match = process_utterance_id_column_for_matching(
                uid_col=F.col(c.KEY_UTTERANCE_ID),
                uid_match_option=utterance_id_match_option
            )

            if (
                    c.KEY_UTTERANCE_ID not in utterance_id_filter.columns and
                    c.KEY_UTTERANCE_ID_GW_STYLE in utterance_id_filter.columns
            ):
                # specialized code to allow 'utteranceId' as the utternace id column in the filter
                utterance_id_filter = utterance_id_filter.withColumnRenamed(
                    c.KEY_UTTERANCE_ID_GW_STYLE, c.KEY_UTTERANCE_ID
                )

            KEY_TMP_UTTERANCEID = f'___{c.KEY_UTTERANCE_ID}'
            df = df.withColumn(
                KEY_TMP_UTTERANCEID,
                uid_col_for_match
            )
            utterance_id_filter = utterance_id_filter.select(
                uid_col_for_match.alias(KEY_TMP_UTTERANCEID)
            )

            df = sparku.filter_by_inner_join_on_columns(
                df, df_filter=utterance_id_filter,
                filter_name='utterance_id_filter'
            ).drop(KEY_TMP_UTTERANCEID)
        else:
            df = sparku.filter_by_inner_join_on_columns(
                df, df_filter=utterance_id_filter, colnames=[c.KEY_UTTERANCE_ID],
                filter_name='utterance_id_filter'
            )

    if final_filter is not None:
        df = df.where(final_filter)

    return df


class NluMergerResultFilter(str, Enum):
    """
    Specifies certain pre-defined NLU merger result filters for request traffic pulling.
    """

    # keeps traffic with merged rewrites
    Merged = 'merged'
    # keeps traffic with non-merged rewrites
    NonMerged = 'non_merged'
    # no filter based on NLU merger result
    NoFilter = 'no_filter'
    # a special filter that keeps the opportunity traffic related to the specified providers
    NonMergedPlusProvider = 'non_merged_plus_provider'


def _solve_nlu_merger_result_filter(nlu_merger_result_filter, providers_filter, final_filter):
    if (
            (nlu_merger_result_filter is None)
            or (nlu_merger_result_filter == '')
            or (nlu_merger_result_filter == 'none')
            or (nlu_merger_result_filter == NluMergerResultFilter.NoFilter)
    ):
        return None, providers_filter, final_filter
    elif (
            (nlu_merger_result_filter is True)
            or (nlu_merger_result_filter == 'true')
            or (nlu_merger_result_filter == NluMergerResultFilter.Merged)
    ):
        return 'true', providers_filter, final_filter
    elif (
            (nlu_merger_result_filter is False)
            or (nlu_merger_result_filter == 'false')
            or (nlu_merger_result_filter == NluMergerResultFilter.NonMerged)
    ):
        return 'false', providers_filter, final_filter
    elif nlu_merger_result_filter == NluMergerResultFilter.NonMergedPlusProvider:
        opportunity_filter = (
                                 F.col(c.KEY_PROVIDER_NAME).isin(make_set(providers_filter))
                             ) | (  # noqa: E126
                                     F.col(c.KEY_NLU_MERGER_RESULT) == 'false'
                             )
        if final_filter is None:
            final_filter = opportunity_filter
        else:
            final_filter = final_filter & opportunity_filter
        return None, None, final_filter
    else:
        raise ValueError("the 'nlu_merger_result_filter' is not recognized")


def _get_greenwich3_raw_request_traffic_columns(
        df: DataFrame,
        extra_aus_and_merger_data: bool
) -> List[Column]:
    cols = [
        F.uuid4().alias(c.KEY_UUID),
        F.col('signals.totalTurns').alias(c.KEY_NUM_TURNS),
        F.col(c.GW_KEY_CUSTOMER_ID).alias(c.KEY_CUSTOMER_ID),  # customer_id
        F.col(c.GW3_KEY_SESSION_ID).alias(c.KEY_SESSION_ID),  # session_id
        F.col(c.KEY_LOCALE),  # locale
        F.col(c.GW3_KEY_DIALOG_ID).alias(c.KEY_DIALOG_ID),  # dialog_id
        F.col(c.GW3_KEY_TURN_INDEX).alias(c.KEY_INDEX),  # index
        F.col(c.GW3_KEY_UTTERANCE_ID).alias(c.KEY_UTTERANCE_ID),  # utterance_id
        F.concat_ws(' ', F.col("turn.request")).alias(c.KEY_REQUEST),  # request
        F.concat_ws(' ', F.col("turn.replacedRequest")).alias(
            c.KEY_REPLACED_REQUEST
        ),  # replaced_request
        F.col(c.GW3_KEY_RESPONSE).alias(c.KEY_RESPONSE),  # response
        F.col(c.GW3_KEY_DOMAIN).alias(c.KEY_DOMAIN),  # domain
        F.col(c.GW3_KEY_INTENT).alias(c.KEY_INTENT),  # intent
        generate_hypothesis_udf(c.GW3_KEY_DOMAIN, c.GW3_KEY_INTENT, 'turn.tokenLabelText').alias(
            c.KEY_HYPOTHESIS
        ),  # hypothesis
        extract_resolved_slots_udf(F.col("turn.slots")).alias(c.KEY_RESOLVED_SLOTS),
        F.col(c.GW3_KEY_ASR_NBEST).alias(c.KEY_ASR_NBEST),
        F.col(c.GW3_KEY_NLU_MERGE_RESULT).alias(c.KEY_NLU_MERGER_RESULT),  # nluAusMergerResult
        extract_merger_rule_udf(F.col(c.GW3_KEY_NLU_MERGER_RULES)).alias(
            c.KEY_NLU_MERGER_RULE
        ),  # nluAusMergerRule
        extract_merger_rule_version_udf(F.col(c.GW3_KEY_NLU_MERGER_RULES)).alias(
            c.KEY_NLU_MERGER_RULE_VERSION
        ),  # nluAusMergerRuleVersion
        get_provider_udf(F.col(c.GW3_KEY_AUS_MERGER_RESULTS)).alias(
            c.KEY_PROVIDER_NAME
        ),  # providerName
        F.col(c.GW3_KEY_DFS_SOURCE).getItem(0).alias(c.KEY_DFS_SOURCE),  # DFS_SOURCE
        F.col(c.GW3_KEY_DFS_SCORE).getItem(0).alias(c.KEY_DFS_SCORE),  # DFS_score
        F.col(c.GW3_KEY_DFS_SCORE_BIN).getItem(0).alias(c.KEY_DFS_SCORE_BIN),  # DFS_score_bin
        F.col(c.GW3_KEY_DFS_SCORE_LATENCY).alias(c.KEY_DFS_LATENCY),  # DFS_latency
        F.col(c.GW3_KEY_NLU_MERGER_DETAIL_ASR_INTERPRETATION).alias(
            c.KEY_ASR_INTERPRETATION
        ),  # asrInterpretation; for extracting ASR hypothesis
        F.col(c.GW3_KEY_NLU_MERGER_DETAIL_AUS_INTERPRETATION).alias(
            c.KEY_AUS_INTERPRETATION
        ),  # ausInterpretation; for extracting AUS hypothesis/request
        F.col(c.GW3_KEY_TURN_DEFECT_BARGEIN).alias(c.KEY_DEFECT_BARGEIN),  # bargeIn
        F.col(c.GW3_KEY_TURN_DEFECT_REPHRASE).alias(c.KEY_DEFECT_REPHRASE),  # rephrase
        F.col(c.GW3_KEY_TURN_DEFECT_TERMINATION).alias(c.KEY_DEFECT_TERMINATION),  # termination
        F.col(c.GW3_KEY_TURN_DEFECT_SANDPAPER).alias(c.KEY_DEFECT_SANDPAPER),  # sandpaper
        F.col(c.GW3_KEY_TURN_DEFECT_UNHANDLED).alias(c.KEY_DEFECT_UNHANDLED),  # unhandled
        F.col(c.GW3_KEY_TURN_TIMESTAMP).alias(c.KEY_TIMESTAMP),  # timestamp
        F.col(c.GW3_KEY_TURN_DFS_SHOULD_TRIGGER).alias(c.KEY_DFS_SHOULD_TRIGGER),  # shouldTrigger
        F.col(c.GW3_KEY_TURN_IS_HOLDOUT).alias(c.KEY_IS_HOLDOUT),  # is_holdout
        F.col(c.GW3_KEY_WEBLAB).alias(c.KEY_WEBLAB),
    ]

    if extra_aus_and_merger_data:
        cols += [
            F.col(c.GW3_KEY_NLU_MERGE_DETAILED_RESULTS).alias(c.KEY_NLU_MERGER_DETAILED_RESULT),
            F.col(c.GW3_KEY_AUS_CRITICAL).alias(c.KEY_AUS_CRITICAL),
        ]

    return cols


def get_greenwich_raw_request_traffic_extra_columns():
    return {
        c.KEY_DEVICE_NO_SCREEN:
            F.when(
                F.col(c.KEY_DEVICE_TYPE).isin(get_known_no_screen_non_tv_device_types()),
                F.lit(True)
            ).otherwise(
                F.lit(None)
            ),
        c.KEY_DEVICE_IS_ECHO_SHOW:
            F.col(c.KEY_DEVICE_TYPE).isin(get_echo_show_device_types()),
        c.KEY_DEVICE_COMMON_NAME:
            get_device_external_name_udf(c.KEY_DEVICE_TYPE),

    }


def get_greenwich_raw_request_traffic_refined_columns():
    return {
        c.KEY_ASR_HYPOTHESIS:
            F.when(
                (F.col(c.KEY_NLU_MERGER_RESULT) == 'true'), F.col(c.KEY_ASR_HYPOTHESIS)
            ).otherwise(
                F.col(c.KEY_HYPOTHESIS)
            ),
        c.KEY_AUS_HYPOTHESIS: F.when(
            (F.col(c.KEY_NLU_MERGER_RESULT) == 'true'), F.col(c.KEY_HYPOTHESIS)
        ).otherwise(
            F.col(c.KEY_AUS_HYPOTHESIS)
        )
    }


def refine_greenwich_raw_request_traffic_columns(df_traffic: DataFrame) -> DataFrame:
    return sparku.with_columns(
        df_traffic,
        get_greenwich_raw_request_traffic_refined_columns()
    )


def add_greenwich_raw_request_traffic_extra_columns(df_traffic: DataFrame) -> DataFrame:
    return sparku.with_columns(
        df_traffic,
        get_greenwich_raw_request_traffic_extra_columns()
    )


def _get_greenwich_unified_raw_request_traffic_columns(
        df: DataFrame,
        extra_aus_and_merger_data: bool
) -> List[Column]:
    cols = [
        F.uuid4().alias(c.KEY_UUID),
        F.col(c.GW_KEY_CUSTOMER_ID).alias(c.KEY_CUSTOMER_ID),  # customer_id
        F.col(c.GW_KEY_PERSON_ID).alias(c.KEY_PERSON_ID),  # customer_id
        F.col(c.KEY_SESSION_ID),  # session_id
        F.col(c.GW_KEY_DEVICE_TYPE).alias(c.KEY_DEVICE_TYPE),  # device_type
        F.col(c.GW_KEY_DEVICE_ID).alias(c.KEY_DEVICE_ID),  # device_id
        F.col(c.GW_KEY_CLIENT_PROFILE).alias(c.KEY_CLIENT_PROFILE),  # client_profile
        F.col(c.KEY_LOCALE),  # locale
        F.col(c.GW_KEY_DIALOG_ID).alias(c.KEY_DIALOG_ID),  # dialog_id
        F.col('_index').alias(c.KEY_INDEX),  # index
        F.col(c.KEY_UTTERANCE_ID),  # utterance_id
        F.concat_ws(' ', F.col("request")).alias(c.KEY_REQUEST),  # request
        F.concat_ws(' ', F.col("replacedRequest")).alias(
            c.KEY_REPLACED_REQUEST
        ),  # replaced_request
        F.col('response2').alias(c.KEY_RESPONSE),  # response
        F.col(c.KEY_DOMAIN),  # domain
        F.col(c.KEY_INTENT),  # intent
        generate_hypothesis_udf(
            F.col(c.KEY_DOMAIN), F.col(c.KEY_INTENT), F.col('tokenLabelText')
        ).alias(
            c.KEY_HYPOTHESIS
        ),  # hypothesis
        extract_resolved_slots_udf(F.col("slots")).alias(c.KEY_RESOLVED_SLOTS),
        F.col(c.GW_KEY_ASR_NBEST).alias(c.KEY_ASR_NBEST),
        F.col(c.KEY_NLU_MERGER_RESULT),  # nluAusMergerResult
        extract_merger_rule_udf(F.col(c.GW_KEY_NLU_MERGER_RULES)).alias(
            c.KEY_NLU_MERGER_RULE
        ),  # nluAusMergerRule
        extract_merger_rule_version_udf(F.col(c.GW_KEY_NLU_MERGER_RULES)).alias(
            c.KEY_NLU_MERGER_RULE_VERSION
        ),  # nluAusMergerRuleVersion
        get_provider_udf(F.col(c.GW_KEY_AUS_MERGER_RESULTS)).alias(
            c.KEY_PROVIDER_NAME
        ),  # providerName
        F.col(c.GW_KEY_NLU_MERGER_DETAIL_ASR_INTERPRETATION).alias(
            c.KEY_ASR_INTERPRETATION
        ),  # asrInterpretation; for extracting ASR hypothesis
        F.col(c.GW_KEY_NLU_MERGER_DETAIL_AUS_INTERPRETATION).alias(
            c.KEY_AUS_INTERPRETATION
        ),  # ausInterpretation; for extracting AUS hypothesis/request
        F.col(c.GW_KEY_TURN_CPD).alias(c.KEY_DEFECT),  # defect (CPD)
        F.col(c.GW_KEY_TURN_CPD_SCORE).alias(c.KEY_CPD_SCORE),  # CPD score
        F.col(c.GW_KEY_TURN_CPD_VERSION).alias(c.KEY_CPD_VERSION),  # CPD score
        F.col(c.GW_KEY_TURN_DEFECT_BARGEIN).alias(c.KEY_DEFECT_BARGEIN),  # bargeIn
        F.col(c.GW_KEY_TURN_DEFECT_REPHRASE).alias(c.KEY_DEFECT_REPHRASE),  # rephrase
        F.col(c.GW_KEY_TURN_DEFECT_TERMINATION).alias(c.KEY_DEFECT_TERMINATION),  # termination
        F.col(c.GW_KEY_TURN_DEFECT_SANDPAPER).alias(c.KEY_DEFECT_SANDPAPER),  # sandpaper
        F.col(c.GW_KEY_TURN_DEFECT_UNHANDLED).alias(c.KEY_DEFECT_UNHANDLED),  # unhandled
        F.col(c.GW_KEY_SESSION_DEFECT).alias(c.KEY_SESSION_DEFECT),  # session defect
        F.col(c.GW_KEY_TURN_TIMESTAMP).alias(c.KEY_TIMESTAMP),  # timestamp
        F.col(c.GW_KEY_TURN_LOCAL_TIMESTAMP).alias(c.KEY_LOCAL_TIMESTAMP),  # local_timestamp
        F.col(c.GW_KEY_TURN_DFS_SHOULD_TRIGGER).cast(StringType()).alias(c.KEY_DFS_SHOULD_TRIGGER),  # shouldTrigger
        F.col(c.GW_KEY_TURN_IS_HOLDOUT).cast(StringType()).alias(c.KEY_IS_HOLDOUT),  # is_holdout
        F.col(c.GW_KEY_WEBLAB).alias(c.KEY_WEBLAB),  # weblab
        F.col(c.GW_KEY_VIDEO_DEVICE_ACTIVE_SESSION).alias(c.KEY_VIDEO_DEVICE_ACTIVE_SESSION),  # video_device_active_session
        F.col(c.GW_KEY_VIDEO_DEVICE_ENABLED).alias(c.KEY_VIDEO_DEVICE_ENABLED),  # video_device_enabled
        F.col(c.GW_KEY_MACAW_CUSTOMER_RESPONSE_TYPE).alias(c.KEY_MACAW_CUSTOMER_RESPONSE_TYPE),  # macaw_customer_response_type
        F.col(c.GW_KEY_IS_LLM_TRAFFIC).alias(c.KEY_IS_LLM_TRAFFIC),  # is_llm_traffic
        F.col(c.GW_KEY_LLM_TOKEN).alias(c.KEY_LLM_TOKEN),  # llm_token
        F.col(c.GW_KEY_LLM_SESSION_TOKEN).alias(c.KEY_LLM_SESSION_TOKEN),  # llm_session_token
        F.col(c.GW_KEY_ALEXA_STACK_TYPE).alias(c.KEY_ALEXA_STACK_TYPE),  # alexa_stack_type
        F.col(c.GW_KEY_ALEXA_STACK_CONFIG).alias(c.KEY_ALEXA_STACK_CONFIG),  # alexa_stack_config
        F.col(c.GW_KEY_LLM_PROPERTIES).alias(c.KEY_LLM_PROPERTIES),  # llm_properties
        F.col(c.GW_KEY_INFO_CATEGORY).alias(c.KEY_INFO_CATEGORY),  # info_category
        F.col(c.GW_KEY_CHILD_DIRECT_REQUEST).alias(c.KEY_CHILD_DIRECT_REQUEST)  # child_directed_request
    ]

    if sparku.has_col(df, c.GW_KEY_PIPELINES):
        cols += [
            F.col(c.GW_KEY_DFS_SOURCE).getItem(0).alias(c.KEY_DFS_SOURCE),  # DFS_SOURCE
            F.col(c.GW_KEY_DFS_SCORE).getItem(0).alias(c.KEY_DFS_SCORE),  # DFS_score
            F.col(c.GW_KEY_DFS_SCORE_BIN).getItem(0).alias(c.KEY_DFS_SCORE_BIN),  # DFS_score_bin
            F.col(c.GW_KEY_DFS_SCORE_LATENCY).alias(c.KEY_DFS_LATENCY),  # DFS_latency
        ]
    else:
        # TODO: one missing field
        cols += [
            get_dfs_source_udf(F.col(c.GW_KEY_AUS_MERGER_RESULTS)).alias(c.KEY_DFS_SOURCE),
            F.col(c.GW_KEY_DFS_SCORE2).getItem(0).alias(c.KEY_DFS_SCORE),  # DFS_score
            F.col(c.GW_KEY_DFS_SCORE_BIN2).getItem(0).alias(c.KEY_DFS_SCORE_BIN),  # DFS_score_bin
            # F.col(GW_KEY_DFS_SCORE_LATENCY2).alias(c.KEY_DFS_LATENCY),  # DFS_latency
        ]

    if extra_aus_and_merger_data:
        cols += [
            F.col(c.GW_KEY_NLU_MERGER_RULES).alias(c.KEY_NLU_MERGER_DETAILED_RESULT),
            F.col(c.GW_KEY_AUS_CRITICAL).alias(c.KEY_AUS_CRITICAL),
        ]

    return cols


def get_raw_request_traffic(
        spark: SparkSession,
        input_path_greenwich3: Union[str, Iterable[str]] = None,
        input_path_greenwich_nextgen: Union[str, Iterable[str]] = None,
        input_path_greenwich_unified: Union[str, Iterable[str]] = None,
        locale: Union[str, SupportedLocales] = SupportedLocales.EN_US,
        extra_aus_and_merger_data: bool = True,
        cpdr_configs: Iterable[CpdrConfig] = (CpdrConfig(),),  # ! DEPRECATED
        greenwich3_extra_cols=None,
        greenwichx_extra_cols=None,
        greenwich_unified_extra_cols=None,
        avoid_redundant_data=True,
        customer_id_filter: set = None,
        dfs_should_trigger_filter: Union[str, bool] = None,
        nlu_merger_result_filter: Union[str, bool, NluMergerResultFilter] = None,
        device_id_filter: set = None,
        utterance_id_filter: set = None,
        utterance_id_filter_match_option: UtteranceIdMatchOptions = UtteranceIdMatchOptions.FullUtteranceId,
        provider_filter=None,
        final_filter: Column = None,
        cache_option: CacheOptions = CacheOptions.IMMEDIATE,
) -> DataFrame:
    """
    Pulls raw customer request traffic
        of pre-defined columns for each customer request_second from Greenwich.
    It first pulls Greenwich3 data from path `greenwich3_input_path`,
        and then joins with Greenwich Nextgen (join by utterance id)
        for CPDR signals if `greenwichx_input_path` is specified.

    Args:
        spark: provides the SparkSession object for the data pulling.
        input_path_greenwich3: the path to the Greenwich3 input data.
        input_path_greenwich_nextgen: the path to the Greenwich Nextgen input data to
                join with the Greenwich3 data by utterance id.
            If this argument is None,
                the Greenwich3 data will not join with Greenwich Nextgen for defect signals.
        locale: only pulls data from this locale, e.g. 'en_US'.
            If this argument is None, then we pull data from all locales.
        extra_aus_and_merger_data: True to pull extra AUS and NLU merger data from
                'turn.weblabs_information', 'turn.ausCritical', 'turn.nluAusMergerDetailedResult'.
            Enabling this to obtain A/B experiment name, detailed DFS pipeline metadata,
                and detailed NLU merger rule informationo applied on each utterance.
        cpdr_configs: pull CPDR signals of these specified versions.
            Not effective is `greenwichx_input_path` is not provided.
        greenwich3_extra_cols: specifies extra columns to collect
                from Greenwich3 in addition to the pre-defined ones.
        greenwichx_extra_cols: specifies extra columns to collect
                from Greenwich NextGen in addition to the pre-defined ones.
        avoid_redundant_data: True to not saving certain redundant data;
                for example, ASR hypothesis will only be saved for merged rewrites,
                AUS utterance and hypothesis will only be saved for non-merged rewrites.
        dfs_should_trigger_filter: specifies True to only pull customer requests
                whose 'shouldTrigger' label is 'true' for the DFS pipeline;
            specifies False to only pull customer queries
                whose 'shouldTrigger' label is 'false' for the DFS pipeline;
            specifies None to pull from all traffic without this filtering.
        nlu_merger_result_filter: specifies True to
                only pull customer requests whose 'nluAusMergerResult' label is 'true';
            specifies False to
                only pull customer queries whose 'nluAusMergerResult' label is 'false';
            specifies None to pull from all traffic without this filtering.
        customer_id_filter: only selects request data of these specified customer IDs.
        device_id_filter: only selects request data of these specified device IDs.
        utterance_id_filter: only selects request data of these specified utterance IDs.
        provider_filter: only selects request data of these specified rewrite providers.
        final_filter: a filter applied on the final traffic dataframe before return.
        cache: True to enable caching during data pulling and return a cached dataframe.

    Returns: a dataframe of raw customer request traffic pulled from Greenwich.

    """

    use_unified_greenwich = bool(input_path_greenwich_unified)
    if use_unified_greenwich and (input_path_greenwich3 or input_path_greenwich_nextgen):
        raise ValueError(
            "either only use unified greenwich, or use greenwich3 and greenwich-nextgen"
        )

    # region STEP1: determine main data source and reads data
    if use_unified_greenwich:
        df = sparku.solve_input(
            input_path_greenwich_unified,
            spark=spark,
            input_format='json',
            schema=SCHEMA_GREENWICH_UNIFIED
        )
        _get_cols = _get_greenwich_unified_raw_request_traffic_columns
        _extra_cols = greenwich_unified_extra_cols
        _df_name = 'greenwich unified'
    else:
        df = (
            sparku.solve_input(
                input_path_greenwich3,
                spark=spark,
                input_format='json',
                schema=SCHEMA_GREENWICH3
            ).withColumn("turn", F.explode("turns"))
        )
        _get_cols = _get_greenwich3_raw_request_traffic_columns
        _extra_cols = greenwich3_extra_cols
        _df_name = 'greenwich3'
    # endregion

    # region STEP2: select main columns & filter by locale
    cols = _get_cols(df, extra_aus_and_merger_data)
    if _extra_cols:
        cols = [
            *cols,
            *(
                (F.col(_col) if isinstance(_col, str) else _col)
                for _col in _extra_cols
            )
        ]
    df = df.select(cols)

    if locale:
        df = df.where(F.col(c.KEY_LOCALE) == locale)

    if not use_unified_greenwich:
        df = df.withColumn(
            c.KEY_DEVICE_ID, get_device_id_from_utterance_id_udf(F.col(c.KEY_UTTERANCE_ID))
        ).withColumn(
            c.KEY_DEVICE_TYPE, get_device_type_from_utterance_id_udf(F.col(c.KEY_UTTERANCE_ID))
        )

    # endregion

    # region STEP3: extracts ASR hypothesis for merged rewrites, and AUS request/hypothesis for non-merged rewrites  # noqa: E501
    if avoid_redundant_data:
        if c.KEY_REPLACED_REQUEST in df.columns:
            df = df.withColumn(
                c.KEY_REPLACED_REQUEST,
                F.when(
                    F.col(c.KEY_NLU_MERGER_RESULT) == 'true', F.col(c.KEY_REPLACED_REQUEST)
                ).otherwise(F.lit(None)),
            )

        df = df.withColumn(
            c.KEY_ASR_HYPOTHESIS,
            F.when(
                F.col(c.KEY_NLU_MERGER_RESULT) == 'true',
                extract_hypothesis_from_asr_aus_interpretation_udf(c.KEY_ASR_INTERPRETATION),
            ).otherwise(F.lit(None)),
        )
        df = df.withColumn(
            c.KEY_AUS_HYPOTHESIS,
            F.when(
                F.col(c.KEY_NLU_MERGER_RESULT) == 'false',
                extract_hypothesis_from_asr_aus_interpretation_udf(c.KEY_AUS_INTERPRETATION),
            ).otherwise(F.lit(None)),
        )
        df = df.withColumn(
            c.KEY_AUS_REQUEST,
            F.when(
                F.col(c.KEY_NLU_MERGER_RESULT) == 'false',
                extract_utterance_from_asr_aus_interpretation_udf(c.KEY_AUS_INTERPRETATION),
            ).otherwise(F.lit(None)),
        )
    else:
        df = df.withColumn(
            c.KEY_ASR_HYPOTHESIS,
            extract_hypothesis_from_asr_aus_interpretation_udf(c.KEY_ASR_INTERPRETATION),
        )
        df = df.withColumn(
            c.KEY_AUS_HYPOTHESIS,
            extract_hypothesis_from_asr_aus_interpretation_udf(c.KEY_AUS_INTERPRETATION),
        )
        df = df.withColumn(
            c.KEY_AUS_REQUEST,
            extract_utterance_from_asr_aus_interpretation_udf(c.KEY_AUS_INTERPRETATION)
        )

    df = df.drop(c.KEY_ASR_INTERPRETATION).drop(c.KEY_AUS_INTERPRETATION)

    # endregion

    # region STEP4: join with greenwich nextgen for defect signals,
    #    if a nextgen input path is provided
    if use_unified_greenwich:
        # _df_tmp = sparku.cache__(
        #     df.select(c.KEY_SESSION_ID).groupBy(
        #         c.KEY_SESSION_ID
        #     ).agg(
        #         F.count('*').alias(c.KEY_NUM_TURNS)
        #     ),
        #     cache_option=cache_option,
        #     name=f'{_df_name} session sizes'
        # )
        # df = sparku.cache__(
        #     df.join(_df_tmp, [c.KEY_SESSION_ID], how='left'),
        #     name=f'{_df_name} joint with session size',
        #     cache_option=cache_option,
        #     unpersist=_df_tmp
        # )

        df = df.join(
            df.select(c.KEY_SESSION_ID).groupBy(
                c.KEY_SESSION_ID
            ).agg(
                F.count('*').alias(c.KEY_NUM_TURNS)
            ),
            [c.KEY_SESSION_ID],
            how='left'
        )
    elif input_path_greenwich_nextgen:
        _df_tmp = get_cpdr_signals_from_greenwich_nextgen(
            spark,
            input_path=input_path_greenwich_nextgen,
            locale=locale,
            cpdr_configs=cpdr_configs,
            extra_cols=greenwichx_extra_cols,
        ).drop(c.KEY_LOCALE, c.KEY_NUM_TURNS)

        df = sparku.cache__(
            df.join(_df_tmp, [c.KEY_UTTERANCE_ID], how='left'),
            name=f'{_df_name} joint with greenwich-nextgen',
            cache_option=cache_option,
            unpersist=_df_tmp
        )

    # endregion

    # region STEP4: filtering
    df = sparku.cache__(
        repartition(
            filter_raw_request_traffic(
                df,
                customer_id_filter=customer_id_filter,
                device_id_filter=device_id_filter,
                utterance_id_filter=utterance_id_filter,
                provider_filter=provider_filter,
                should_trigger_filter=dfs_should_trigger_filter,
                nlu_merger_result_filter=nlu_merger_result_filter,
                utterance_id_match_option=utterance_id_filter_match_option,
                final_filter=final_filter
            ),
            spark=spark
        ),
        name=f'{_df_name} (filtered)',
        cache_option=cache_option,
        unpersist=df
    )

    # endregion

    return df


def add_selected_session_turn_data_to_raw_request_traffic(
        df_traffic: DataFrame,
        previous_turns_only: bool = True,
        session_id_colname: str = c.KEY_SESSION_ID,
        num_turns_colname: str = c.KEY_NUM_TURNS,
        turn_index_colname: str = c.KEY_INDEX,
        selected_cols: Tuple[NameOrColumn, ...] = (
                c.KEY_REQUEST,
                c.KEY_AUS_REQUEST,
                c.KEY_PROVIDER_NAME,
                c.KEY_NLU_MERGER_RESULT,
                c.KEY_ASR_HYPOTHESIS,
                c.KEY_AUS_HYPOTHESIS,
                c.KEY_RESPONSE
        ),
        output_session_turn_data_colname: str = c.KEY_SESSION_TURNS,
        cache_option: CacheOptions = CacheOptions.IMMEDIATE,
) -> DataFrame:
    df_session_turns = sparku.cache__(
        sparku.fold(
            df_traffic.select(
                session_id_colname,
                turn_index_colname,
                num_turns_colname,
                *selected_cols
            ).where(
                F.col(num_turns_colname) > 1
            ),
            group_cols=[session_id_colname, num_turns_colname],
            fold_colname=output_session_turn_data_colname
        ),
        repartition=True,
        name='df_session_turns',
        cache_option=cache_option
    )

    sparku.show_counts(
        df_session_turns,
        (
                F.size(output_session_turn_data_colname) == F.col(num_turns_colname)
        ).alias('consistent_session_size')
    )

    df_traffic_with_session_turns = sparku.join_on_columns(
        df_traffic,
        df_session_turns.drop(num_turns_colname),
        [session_id_colname],
        how='left'
    )

    if previous_turns_only:
        df_traffic_with_session_turns = df_traffic_with_session_turns.withColumn(
            output_session_turn_data_colname,
            F.slice(
                F.col(output_session_turn_data_colname),
                1,
                F.col(turn_index_colname)
            )
        )

    df_traffic_with_session_turns = sparku.cache__(
        df_traffic_with_session_turns,
        name='df_traffic_with_session_turns',
        unpersist=df_session_turns,
        cache_option=cache_option
    )

    return df_traffic_with_session_turns


def get_cpdr_signals_from_greenwich_nextgen(
        spark: SparkSession,
        input_path: str,
        locale: Union[str, SupportedLocales] = None,
        cpdr_configs: Iterable[CpdrConfig] = (CpdrConfig(),),
        extra_cols=None,
) -> DataFrame:
    """
    Gets CPDR signals from Greenwich NextGen.

    Args:
        spark: provides the SparkSession object for the data pulling.
        input_path: the path to the Greenwich Nextgen data.
        locale: only pulls data from this locale, e.g. 'en_US'.
            If this argument is None, then we pull data from all locales.
        extra_cols: specifies extra columns to collect
                from Greenwich NextGen in addition to the CPDR signals.
        cpdr_configs: pull CPDR signals of these specified versions.
    Returns: a dataframe of CPDR signals pulled from Greenwich NextGen.

    """

    df = (
        spark.read.json(input_path, schema=SCHEMA_GREENWICH_NEXTGEN)  # noqa: E131
        .withColumn(c.KEY_NUM_TURNS, F.size(F.col('turns')))
        .withColumn('turn', F.explode('turns'))
    )

    select_cols = [
        F.col('turn._id').alias(c.KEY_UTTERANCE_ID),
        F.col(c.KEY_NUM_TURNS),  # num_turns
        F.col(c.KEY_LOCALE),
    ]

    if cpdr_configs is not None:
        for cpdr_config in cpdr_configs:
            select_cols.append(
                extract_session_defect_for_next_gen_udf(
                    F.col('session_signals'), version=cpdr_config.greenwich_version_label
                ).alias(cpdr_config.session_defect_colname)
            )
            select_cols.append(
                extract_defect_from_nextgen__(
                    'turn.signals.defect.versions', version=cpdr_config.greenwich_version_label
                ).alias(cpdr_config.utterance_defect_colname)
            )

    if extra_cols is not None:
        select_cols.extend(extra_cols)

    df = df.select(select_cols)

    if locale:
        df = df.where(F.col(c.KEY_LOCALE) == locale)

    return df


def get_provider_raw_request_traffic(
        spark: SparkSession,
        greenwich3_input_path: str,
        provider_names: Iterable[str],
        greenwichx_input_path: str = None,
        nlu_merger_result_filter: NluMergerResultFilter = None,
        locale: Union[str, SupportedLocales] = SupportedLocales.EN_US,
        dfs_should_trigger_filter=None,
        cache_option: boba_python_utils.spark_utils.common.CacheOptions = boba_python_utils.spark_utils.common.CacheOptions.IMMEDIATE,
        **kwargs
):
    """
    Pulls raw customer request traffic related to the specified rewrite providers from Greenwich.
    A convenient method that directly calls the `get_greenwich_raw_traffic` for data pulling,
        but making parameter `provider_names` required.
    """
    if not provider_names:
        raise ValueError("argument 'provider_names' must be non-empty")

    return get_raw_request_traffic(
        spark,
        input_path_greenwich3=greenwich3_input_path,
        input_path_greenwich_nextgen=greenwichx_input_path,
        locale=locale,
        nlu_merger_result_filter=nlu_merger_result_filter,
        dfs_should_trigger_filter=dfs_should_trigger_filter,
        provider_filter=provider_names,
        cache_option=cache_option,
        **kwargs
    )


def get_pdfs_raw_request_traffic(
        spark: SparkSession,
        greenwich3_input_path: str,
        greenwichx_input_path: str = None,
        nlu_merger_result_filter: NluMergerResultFilter = None,
        locale: Union[str, SupportedLocales] = SupportedLocales.EN_US,
        dfs_should_trigger_filter='true',
        **kwargs
):
    """
    Pulls raw customer request traffic related to pDFS from Greenwich.
    A convenient method that directly calls the `get_greenwich_raw_traffic` for data pulling,
        but adds a provider name filter and changes default parameter value
        of `dfs_should_trigger_filter` to 'true'.
    """
    return get_raw_request_traffic(
        spark,
        input_path_greenwich3=greenwich3_input_path,
        input_path_greenwich_nextgen=greenwichx_input_path,
        locale=locale,
        nlu_merger_result_filter=nlu_merger_result_filter,
        dfs_should_trigger_filter=dfs_should_trigger_filter,
        provider_filter={c.PROVIDER_NAME__PDFS},
        **kwargs
    )


def replace_request_with_merged_replaced_request(df_traffic: DataFrame) -> DataFrame:
    return df_traffic.withColumn(
        c.KEY_REQUEST,
        F.when(
            (
                    (F.col(c.KEY_NLU_MERGER_RESULT) == 'true') &
                    (F.col(c.KEY_REPLACED_REQUEST).isNotNull()) &
                    (F.col(c.KEY_REPLACED_REQUEST) != '')
            ),
            F.col(c.KEY_REPLACED_REQUEST)
        ).otherwise(F.col(c.KEY_REQUEST))
    )


class RequestWithMergedRewriteFilterOption(str, Enum):
    NoAction = 'use_request'
    DiscardIfWithMergedRewrites = 'non_merged_only'
    DiscardIfNotWithValidMergedRewrites = 'merged_only'
    UseReplacedRequestIfWithMergedRewrites = 'use_replaced_request'


def filter_traffic_with_merged_rewrites(
        df_traffic: DataFrame,
        option: RequestWithMergedRewriteFilterOption
) -> DataFrame:
    if option == RequestWithMergedRewriteFilterOption.UseReplacedRequestIfWithMergedRewrites:
        df_traffic = replace_request_with_merged_replaced_request(df_traffic)
    elif option == RequestWithMergedRewriteFilterOption.DiscardIfWithMergedRewrites:
        df_traffic = df_traffic.where(
            (F.col(c.KEY_NLU_MERGER_RESULT).isNull()) |
            (F.col(c.KEY_NLU_MERGER_RESULT) == 'false')
        )
    elif option == RequestWithMergedRewriteFilterOption.DiscardIfNotWithValidMergedRewrites:
        df_traffic = df_traffic.where(
            (F.col(c.KEY_REPLACED_REQUEST).isNotNull()) &
            (F.col(c.KEY_REPLACED_REQUEST) != '') &
            (F.col(c.KEY_NLU_MERGER_RESULT).isNotNull()) &
            (F.col(c.KEY_NLU_MERGER_RESULT) == 'true') &
            (F.col(c.KEY_REPLACED_REQUEST) != (F.col(c.KEY_REQUEST)))
        )
    return df_traffic


# endregion

# region daily utterance aggregation


def get_greenwich_request_aggregation(
        spark: SparkSession,
        greenwich3_input_path: str,
        greenwichx_input_path: str,
        locale: Union[str, SupportedLocales] = SupportedLocales.EN_US,
        group_cols: Iterable[str] = c.GROUP_KEYS__CUSTOMER_REQ_HYP_DOMAIN_PROVIDER,
        cpdr_config: CpdrConfig = None,
        agg_cols=c.KEYS_DEFECTIVE_SIGNALS,
        max_cols=(c.KEY_TIMESTAMP,),
        occurrence_list_col_name=c.KEY_OCCURRENCES,
        occurrence_cols=c.STRUCT_FIELDS__OCCURRENCES,
        keep_occurrence_defect_signals: bool = True,
        should_trigger_filter: Union[str, bool] = None,
        merger_result_filter: Union[str, bool] = None,
        cache: bool = False,
        **kwargs
):
    """
    Gets joint Greenwich aggregation from Greenwich3 and Greenwich Nextgen.
    The `greenwich3_input_path` and `greenwichx_input_path` must be for the same time range.
    Use this function to build daily aggregations or weekly aggregations.

    Args:
        spark: provides the SparkSession object for the data pulling.
        greenwich3_input_path: the path to the Greenwich3 input data.
        greenwichx_input_path: the path to the Greenwich Nextgen input data to
            join with the Greenwich3 data by utterance id.
        locale: only pulls data from this locale, e.g. 'en_US'.
        group_cols: group the request traffic by these columns.
            If this argument is None,
            then we simply save the flat raw request traffic without aggregations.
        agg_cols: performs 'sum' aggregations on these columns within each group.
        max_cols: performs 'max' aggregations on these columns within each group.
        occurrence_list_col_name: specifies a column name to
            collect a list of occurrences that saves certain data for each request in a group.
            If this argument is None, then we do not preserve data for each individual requeset.
        occurrence_cols: the columns for each occurrence in the occurrence list column.
        keep_occurrence_defect_signals: True to add all pre-defined defect signal columns to
            `occurrence_cols` even they are not specified in the argument.
        should_trigger_filter: see `get_greenwich_raw_traffic`.
        merger_result_filter: see `get_greenwich_raw_traffic`.
        kwargs: other named arguments that can be sent to `get_greenwich_raw_traffic` function.

    Returns:

    """
    # region STEP1: reads raw traffic from greenwich3/nextgen

    df_raw_request_traffic = get_raw_request_traffic(
        spark,
        input_path_greenwich3=greenwich3_input_path,
        cpdr_configs=[cpdr_config],
        dfs_should_trigger_filter=should_trigger_filter,
        nlu_merger_result_filter=merger_result_filter,
        locale=locale,
        input_path_greenwich_nextgen=greenwichx_input_path,
        # ! provides the nextgen path here so the extra defective signals are extracted
        cache=cache,
        **kwargs
    )

    # endregion

    # region STEP2: it is your decision whether aggregate the occurrences

    if group_cols is not None:
        # preserves binary raw defect scores
        if occurrence_list_col_name is not None:
            if occurrence_cols is None:
                raise ValueError(
                    "'occurrences_col_name' is set, but 'occurrence_list_cols' is not specified"
                )
            if keep_occurrence_defect_signals:
                cpdr_keys = [cpdr_config.utterance_defect_colname]
                occurrence_cols = tuple({*occurrence_cols, *cpdr_keys, *c.KEYS_DEFECTIVE_SIGNALS})
                if kwargs.get('extra_aus_and_merger_data', False):
                    occurrence_cols += c.KEYS_DETAILED_AUS_AND_MERGER_DATA
        else:
            occurrence_cols = None
        # make aggregation

        request_aggregation = sparku.aggregate(
            df_raw_request_traffic,
            group_cols=group_cols,
            sum_cols=agg_cols,  # the defect signals are summed in the aggregation
            max_cols=max_cols,
            collect_list_cols={occurrence_list_col_name: occurrence_cols}
            if occurrence_cols is not None
            else None,
        )
        if cache:
            request_aggregation = sparku.cache__(
                request_aggregation, name='request_aggregation', unpersist=df_raw_request_traffic
            )
    else:
        request_aggregation = df_raw_request_traffic

    # endregion

    return request_aggregation

# endregion
