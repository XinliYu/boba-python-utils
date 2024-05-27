import pprint
from collections import Mapping
from os import path

from pyspark.sql.session import SparkSession

import boba_python_utils.spark_utils.spark_functions as F
import boba_python_utils.spark_utils as sparku
import boba_python_utils.production_utils.pdfs.constants as c
from boba_python_utils.general_utils.console_util import hprint_message
from boba_python_utils.production_utils.greenwich_data.common import get_time_lag_udf, CpdrConfig


# TODO
# 1. write_df supports sub_dir
# 2. in-code doc

def build_context_data_from_daily_traffic_aggregation(
        spark: SparkSession,
        input_daily_traffic,
        cpdr_config: CpdrConfig,
        output_path: str,
        input_daily_traffic_format: str = None,
        min_num_turns=2,
        max_num_turns=20,
        max_query_time_elapsed_from_session_start=100,
        input_historical_cpdr=None,
        input_historical_cpdr_format: str = None,
        historical_cpdr_colname: str = None,
        min_query_defect=0.4,
        input_data_fields=None,
        input_rephrase_labels=None,
        input_rephrase_labels_format=None,
        debug_mode=False,
        max_allowed_inconsistent_session_size_ratio: float = 0.03
):
    """

    Filters applied,
        1. query filter by CPDR;

    Args:
        spark:
        input_daily_traffic:
        input_daily_traffic_format:
        cpdr_config:
        output_path:
        min_num_turns:
        max_num_turns:
            recommended 12 for training and 20 for inference
        max_query_time_elapsed_from_session_start:
        input_historical_cpdr:
        historical_cpdr_colname:
        min_query_defect:
        input_data_fields: selects these specified fields from the input daily traffic data;
            if not specified, a default set of fields will be selected.

    Returns:

    """
    # region STEP1: loads traffic data
    # The daily traffic data is flat where each row is a turn with a session id.

    if input_data_fields is None:
        input_data_fields = (
            c.KEY_UUID,
            c.KEY_CUSTOMER_ID,
            c.KEY_SESSION_ID,
            c.KEY_NUM_TURNS,
            c.KEY_INDEX,
            c.KEY_REQUEST,
            c.KEY_HYPOTHESIS,
            c.KEY_DOMAIN,
            c.KEY_RESPONSE,
            c.KEY_NLU_MERGER_RESULT,
            c.KEY_PROVIDER_NAME,
            c.KEY_REPLACED_REQUEST,
            cpdr_config.utterance_defect_colname,
            c.KEY_TIMESTAMP
        )

    cond_session_size_filter = (
            (F.col(c.KEY_NUM_TURNS) >= min_num_turns) &
            (F.col(c.KEY_NUM_TURNS) <= max_num_turns)
    )
    msg_session_size_filter = f'session size ' \
                              f'between {min_num_turns} inclusive ' \
                              f'and {max_num_turns} inclusive'

    # numbers for reference:
    #   en-US 2022 7days 1411764772
    #   en-GB 2022 120days 8074161659
    #   de-de 2022 120days 6484445501
    if callable(input_daily_traffic):
        df_daily_traffic = sparku.solve_input(
            input_daily_traffic,
            spark=spark,
            input_format=input_daily_traffic_format,
            verbose=debug_mode,
            where=cond_session_size_filter
        )
        _can_apply_session_size_filter_before_caching = (
                bool(input_historical_cpdr) or
                (historical_cpdr_colname in df_daily_traffic.columns)
        )
        if not _can_apply_session_size_filter_before_caching:
            raise ValueError(
                f"expects existence of query defect column '{historical_cpdr_colname}' "
                f"or provide 'input_historical_cpdr'"
            )
    else:
        df_daily_traffic = sparku.solve_input(
            input_daily_traffic,
            spark=spark,
            input_format=input_daily_traffic_format,
            verbose=debug_mode
        )  # loading data, no caching

        _can_apply_session_size_filter_before_caching = (
                bool(input_historical_cpdr) or
                (historical_cpdr_colname in df_daily_traffic.columns)
        )
        if _can_apply_session_size_filter_before_caching:
            # filter data to reduce caching size
            df_daily_traffic = df_daily_traffic.where(cond_session_size_filter)

    df_daily_traffic = sparku.cache__(
        df_daily_traffic,
        select=input_data_fields,
        name=f'df_daily_traffic'
             f'{"" if not _can_apply_session_size_filter_before_caching else f" ({msg_session_size_filter})"}'
    )  # real caching data

    df_rephrase_labels = None
    if input_rephrase_labels is not None and callable(input_rephrase_labels):
        df_rephrase_labels = sparku.cache__(
            input_rephrase_labels(df_daily_traffic).where(
                F.col(c.KEY_REQUEST_FIRST) != F.col(c.KEY_REQUEST_SECOND)
            ).distinct(),
            name='df_rephrase_labels'
        )

    # endregion

    # region STEP2: filter daily traffic sessions
    #   1) the session must contain a non-merged defective query;
    #   2) the number of turns in the session is within a specified range (
    #       this might have been applied earlier to reduce cache size);
    #   3)
    # The filtering is performed on the flatten data.
    cond_defective_non_merged_turn = (
            (F.col(cpdr_config.utterance_defect_colname) == 1.0) &
            (F.col(c.KEY_NLU_MERGER_RESULT) == 'false')
    )
    msg_filter = "session contains one defective non-merged query"

    cond_session_filter = cond_defective_non_merged_turn
    if not _can_apply_session_size_filter_before_caching:
        cond_session_filter = cond_session_filter & cond_session_size_filter
        msg_filter = f'{msg_filter}, {msg_session_size_filter}'

    df_daily_traffic_filtered = sparku.cache__(
        sparku.filter_by_inner_join_on_columns(
            df_daily_traffic,
            df_daily_traffic.where(cond_session_filter),
            [c.KEY_SESSION_ID],
            broadcast_join=False  # ! DO NOT use broadcasting; too many session ids
        ),
        name=f'df_daily_traffic_filtered ({msg_filter})',
        unpersist=df_daily_traffic
    )

    session_contains_rephrase_filter = sparku.prev_and_next(
        df_daily_traffic_filtered.select(
            c.KEY_SESSION_ID,
            c.KEY_INDEX,
            c.KEY_REQUEST,
            cpdr_config.utterance_defect_colname
        ),
        group_cols=c.KEY_SESSION_ID,
        order_cols=c.KEY_INDEX,
        null_next_indicator_col_name=c.KEY_REQUEST,
        keep_order_cols=True
    ).where(
        (F.col(cpdr_config.utterance_defect_first_colname) == 1) &
        (F.col(cpdr_config.utterance_defect_second_colname) == 0)
    )

    df_daily_traffic_filtered = sparku.cache__(
        sparku.filter_by_inner_join_on_columns(
            df_daily_traffic_filtered,
            session_contains_rephrase_filter,
            [c.KEY_SESSION_ID],
            broadcast_join=False  # ! DO NOT use broadcasting; too many session ids
        ),
        name='df_daily_traffic_filtered (session contains rephrases)',
        unpersist=df_daily_traffic_filtered
    )

    # endregion

    # region STEP3: group daily traffic by session

    df_daily_traffic_filtered_grouped_by_sessions, data_cnt = sparku.cache__(
        sparku.aggregate(
            df_daily_traffic_filtered.orderBy(c.KEY_SESSION_ID, c.KEY_INDEX),
            group_cols=[c.KEY_SESSION_ID],
            count_col='',
            collect_list_cols={
                'turns': [
                    c.KEY_INDEX,
                    c.KEY_REQUEST,
                    c.KEY_HYPOTHESIS,
                    c.KEY_RESPONSE,
                    cpdr_config.utterance_defect_colname
                ]
            }
        ),
        name='df_daily_traffic_filtered_grouped_by_sessions',
        return_count=True
    )

    # endregion

    # region STEP4: saves sessions with all defective queries

    KEY_FIRST_TURN_TIMESTAMP = f'first_turn_{c.KEY_TIMESTAMP}'

    df_all_defective_non_merged_request_first = sparku.rename(
        df_daily_traffic_filtered.where(cond_defective_non_merged_turn),
        {
            c.KEY_REQUEST: c.KEY_REQUEST_FIRST,
            c.KEY_INDEX: c.KEY_INDEX_FIRST
        }
    )

    df_daily_traffic_filtered_grouped_by_sessions_with_query, data_cnt = sparku.cache__(
        df_daily_traffic_filtered_grouped_by_sessions.join(
            df_all_defective_non_merged_request_first,
            [c.KEY_SESSION_ID]
        ),
        name='df_daily_traffic_filtered_grouped_by_sessions_with_query',
        return_count=True
    )

    inconsistent_session_size_cond = (F.col(c.KEY_NUM_TURNS) != F.size('turns'))
    inconsistent_session_size_cnt = df_daily_traffic_filtered_grouped_by_sessions_with_query.where(
        inconsistent_session_size_cond
    ).count()
    hprint_message('inconsistent_session_size_cnt', inconsistent_session_size_cnt)
    pprint.pprint(
        df_daily_traffic_filtered_grouped_by_sessions_with_query.where(
            inconsistent_session_size_cond
        ).head().asDict()
    )
    if inconsistent_session_size_cnt / data_cnt > max_allowed_inconsistent_session_size_ratio:
        raise ValueError(
            f"Too many sessions with inconsistent session size "
            f"({inconsistent_session_size_cnt} out of {data_cnt})"
        )
    pprint.pprint(
        df_daily_traffic_filtered_grouped_by_sessions_with_query.where(
            F.col(c.KEY_NUM_TURNS) > 2
        ).head().asDict()
    )

    sparku.write_df(
        df_daily_traffic_filtered_grouped_by_sessions_with_query,
        (output_path['all'] if isinstance(output_path, Mapping) else path.join(output_path, 'all')),
        num_files=512,
        compress=True
    )

    if input_rephrase_labels is not None:
        if df_rephrase_labels is None:
            df_rephrase_labels = sparku.cache__(
                input_rephrase_labels,
                input_format=input_rephrase_labels_format,
                spark=spark,
                name='df_rephrase_labels'
            )

        df_daily_traffic_filtered_overlap_with_labels = sparku.cache__(
            sparku.filter_by_inner_join_on_columns(
                df_daily_traffic_filtered.where(
                    F.col(cpdr_config.utterance_defect_colname) == 0
                ).select(
                    F.col(c.KEY_REQUEST).alias(c.KEY_REQUEST_SECOND),
                    F.col(c.KEY_INDEX).alias(c.KEY_INDEX_SECOND),
                    c.KEY_SESSION_ID
                ),
                df_rephrase_labels.drop(c.KEY_REQUEST_FIRST)
            ),
            name='df_daily_traffic_filtered_overlap_with_labels'
        )

        KEY_REQUEST_SECOND_LIST = f'{c.KEY_REQUEST_SECOND}_list'
        df_daily_traffic_filtered_joint_with_labels = sparku.cache__(
            sparku.filter_by_inner_join_on_columns(
                df_daily_traffic_filtered.where(
                    F.col(cpdr_config.utterance_defect_colname) == 1
                ),
                df_daily_traffic_filtered_overlap_with_labels,
                [c.KEY_SESSION_ID]
            ).select(
                F.col(c.KEY_REQUEST).alias(c.KEY_REQUEST_FIRST),
                F.col(c.KEY_INDEX).alias(c.KEY_INDEX_FIRST),
                c.KEY_SESSION_ID
            ).join(
                df_rephrase_labels.groupBy(
                    c.KEY_REQUEST_FIRST
                ).agg(
                    F.collect_list(c.KEY_REQUEST_SECOND).alias(KEY_REQUEST_SECOND_LIST)
                ),
                [c.KEY_REQUEST_FIRST]
            ),
            name='df_daily_traffic_filtered_joint_with_labels'
        )

        df_rephrase_labels = sparku.cache__(
            sparku.one_from_each_group(
                sparku.explode_as_flat_columns(df_daily_traffic_filtered_joint_with_labels, col_to_explode=KEY_REQUEST_SECOND_LIST, explode_colname_or_prefix=c.KEY_REQUEST_SECOND).join(
                    df_daily_traffic_filtered_overlap_with_labels,
                    [c.KEY_SESSION_ID, c.KEY_REQUEST_SECOND]
                ).where(
                    F.col(c.KEY_INDEX_FIRST) < F.col(c.KEY_INDEX_SECOND)
                ).select(
                    c.KEY_SESSION_ID,
                    c.KEY_REQUEST_FIRST,
                    c.KEY_REQUEST_SECOND,
                    c.KEY_INDEX_SECOND
                ),
                group_cols=[c.KEY_SESSION_ID, c.KEY_REQUEST_FIRST],
                order_cols=[c.KEY_INDEX_SECOND]
            ).where(
                F.col(c.KEY_REQUEST_FIRST) != F.col(c.KEY_REQUEST_SECOND)
            ),
            name='df_rephrase_labels (associated with sessions)',
            unpersist=(
                df_rephrase_labels,
                df_daily_traffic_filtered_overlap_with_labels,
                df_daily_traffic_filtered_joint_with_labels
            )
        )

        df_daily_traffic_filtered_grouped_by_sessions_with_query_and_label = sparku.cache__(
            df_daily_traffic_filtered_grouped_by_sessions_with_query.join(
                df_rephrase_labels,
                [c.KEY_SESSION_ID, c.KEY_REQUEST_FIRST]
            ),
            name='df_daily_traffic_filtered_grouped_by_sessions_with_query_and_label',
            unpersist=df_daily_traffic_filtered_grouped_by_sessions_with_query
        )

        df_daily_traffic_filtered_grouped_by_sessions_with_query_and_label.select(
            c.KEY_REQUEST_FIRST, c.KEY_REQUEST_SECOND, c.KEY_INDEX_SECOND, c.KEY_TURNS
        ).head().asDict()

        # 'all_labeled' is used for training in CRD
        sparku.write_df(
            df_daily_traffic_filtered_grouped_by_sessions_with_query_and_label,
            (path.join(output_path, 'all_labeled') if isinstance(output_path, str) else output_path['all_labeled']),
            num_files=512,
            compress=True,
            # unpersist=True
        )

    df_daily_traffic_filtered_grouped_by_sessions_with_query.unpersist()
    # endregion

    # region STEP4: saves sessions with filtered defective queries
    #  1) query CPDR is above certain threshold;
    #  2) within certain time from the first turn of the session.

    if max_query_time_elapsed_from_session_start:
        df_all_defective_non_merged_request_first = sparku.where(
            df=df_all_defective_non_merged_request_first.join(
                df_daily_traffic_filtered.where(F.col(c.KEY_INDEX) == 0).select(
                    c.KEY_SESSION_ID,
                    F.col(c.KEY_TIMESTAMP).alias(KEY_FIRST_TURN_TIMESTAMP)
                ),
                [c.KEY_SESSION_ID]
            ),
            cond=(
                    get_time_lag_udf(KEY_FIRST_TURN_TIMESTAMP, c.KEY_TIMESTAMP)
                    <= max_query_time_elapsed_from_session_start
            )
        )

    if min_query_defect is not None:
        if historical_cpdr_colname is None:
            historical_cpdr_colname = c.KEY_GLOBAL_AVG_DEFECT

        cond_query_filter = (F.col(historical_cpdr_colname) >= min_query_defect)

        if input_historical_cpdr:
            df_query_filter = sparku.cache__(
                input_historical_cpdr,
                spark=spark,
                input_format=input_historical_cpdr_format,
                name='df_query_filter',
                where=cond_query_filter
            )
        elif historical_cpdr_colname in df_all_defective_non_merged_request_first.columns:
            df_query_filter = sparku.cache__(
                df_daily_traffic.where(cond_query_filter),
                name='df_query_filter'
            )
        else:
            # numbers for reference:
            #   en-GB 2022 120 days 245323681
            #   de-DE 2022 120 days 168619516
            df_query_filter = sparku.cache__(
                df_daily_traffic.where(F.col(c.KEY_NLU_MERGER_RESULT) == 'false').groupBy(
                    c.KEY_REQUEST, c.KEY_DOMAIN
                ).agg(
                    F.avg(
                        cpdr_config.utterance_defect_colname
                    ).alias(
                        historical_cpdr_colname
                    )
                ).where(
                    cond_query_filter
                ),
                name='df_query_filter'
            )
        df_all_defective_non_merged_request_first = sparku.filter_by_inner_join_on_columns(
            df_all_defective_non_merged_request_first,
            df_query_filter.withColumnRenamed(c.KEY_REQUEST, c.KEY_REQUEST_FIRST),
            [c.KEY_REQUEST_FIRST, c.KEY_DOMAIN]
        )

        query_filtered_select_cols = [
            c.KEY_UUID,
            c.KEY_CUSTOMER_ID,
            c.KEY_SESSION_ID,
            c.KEY_NUM_TURNS,
            c.KEY_REQUEST_FIRST,
            F.col(c.KEY_HYPOTHESIS).alias(c.KEY_HYPOTHESIS_FIRST),
            c.KEY_INDEX_FIRST
        ]
        if historical_cpdr_colname in df_all_defective_non_merged_request_first.columns:
            query_filtered_select_cols.append(historical_cpdr_colname)

        df_all_defective_non_merged_request_first = \
            df_all_defective_non_merged_request_first.select(
                query_filtered_select_cols
            )

    df_daily_traffic_filtered_grouped_by_sessions_with_query = sparku.cache__(
        df_daily_traffic_filtered_grouped_by_sessions.join(
            df_all_defective_non_merged_request_first,
            [c.KEY_SESSION_ID]
        ),
        name='df_daily_traffic_filtered_grouped_by_sessions_with_query'
    )

    # 'query_filtered' is used for inference in CRD
    sparku.write_df(
        df_daily_traffic_filtered_grouped_by_sessions_with_query,
        (path.join(output_path, 'query_filtered') if isinstance(output_path, str) else output_path['query_filtered']),
        num_files=512,
        compress=True
    )

    if input_rephrase_labels is not None:
        sparku.write_df(
            df_daily_traffic_filtered_grouped_by_sessions_with_query.join(
                df_rephrase_labels,
                [c.KEY_SESSION_ID, c.KEY_REQUEST_FIRST]
            ),
            (path.join(output_path, 'query_filtered_labeled') if isinstance(output_path, str) else output_path['query_filtered_labeled']),
            num_files=512,
            compress=True
        )
        df_rephrase_labels.unpersist()

    df_daily_traffic_filtered_grouped_by_sessions_with_query.unpersist()
    df_daily_traffic_filtered.unpersist()
    df_daily_traffic_filtered_grouped_by_sessions.unpersist()

    # endregion
